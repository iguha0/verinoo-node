#!/usr/bin/env python3
"""Create a Stripe invoice from a Verinoo usage CSV export.

The CSV comes from ``GET /v1/usage/:address/csv`` and has the columns
``address,tasks,completed,pending,failed,feesCommitted,balance``. Billing is in
fiat; the chain's fee and balance fields are informational only. By default the
script prints a dry-run summary and makes no network calls. With ``--live`` it
finds or creates a Stripe customer, adds one invoice item and finalizes an
invoice without paying it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REQUIRED_COLUMNS = (
    "address",
    "tasks",
    "completed",
    "pending",
    "failed",
    "feesCommitted",
    "balance",
)
DEFAULT_RATE_CENTS = "0.5"
STRIPE_API_BASE = "https://api.stripe.com"
IDEMPOTENCY_PREFIX = "verinoo-invoice-"
MAX_IDEMPOTENCY_LENGTH = 255


class InvoiceError(Exception):
    """Malformed input or an invalid invoice request."""


class StripeError(Exception):
    """A Stripe API or transport failure."""


@dataclass
class UsageRow:
    address: str
    tasks: int
    completed: int
    pending: int
    failed: int
    fees_committed: int
    balance: int
    line: int


@dataclass
class UsageAggregate:
    address: str
    tasks: int = 0
    completed: int = 0
    pending: int = 0
    failed: int = 0
    fees_committed: int = 0
    balance: int = 0
    rows: int = 0

    def add(self, row: UsageRow) -> None:
        self.tasks += row.tasks
        self.completed += row.completed
        self.pending += row.pending
        self.failed += row.failed
        self.fees_committed += row.fees_committed
        self.balance += row.balance
        self.rows += 1


def _parse_rate(value: str) -> Decimal:
    try:
        rate = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"invalid rate {value!r}") from exc
    if not rate.is_finite() or rate < 0:
        raise argparse.ArgumentTypeError("rate must be a non-negative number of cents")
    if rate == 0:
        return Decimal(0)
    return rate


def format_rate(rate: Decimal) -> str:
    return format(rate.normalize(), "f")


def _parse_int(raw: str, column: str, line: int) -> int:
    text = (raw or "").strip()
    if not text:
        raise InvoiceError(f"line {line}: {column} is empty")
    try:
        value = int(text, 10)
    except ValueError as exc:
        raise InvoiceError(f"line {line}: {column} must be an integer, got {text!r}") from exc
    if value < 0:
        raise InvoiceError(f"line {line}: {column} must not be negative")
    return value


def parse_usage_csv(text: str) -> List[UsageRow]:
    if text.startswith("\ufeff"):
        text = text[1:]
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader)
    except StopIteration as exc:
        raise InvoiceError("CSV is empty; expected a header row") from exc

    header = [name.strip() for name in header]
    if not any(header):
        raise InvoiceError("CSV has an empty header row")

    seen = set()
    duplicates = []
    for name in header:
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise InvoiceError("CSV has duplicate column(s): " + ", ".join(sorted(set(duplicates))))

    missing = [name for name in REQUIRED_COLUMNS if name not in header]
    if missing:
        raise InvoiceError("CSV is missing required column(s): " + ", ".join(missing))

    positions = {name: header.index(name) for name in REQUIRED_COLUMNS}
    rows: List[UsageRow] = []
    for line, cells in enumerate(reader, start=2):
        if not cells or all(not cell.strip() for cell in cells):
            continue
        if len(cells) != len(header):
            raise InvoiceError(
                f"line {line}: expected {len(header)} columns, found {len(cells)}"
            )
        address = cells[positions["address"]].strip()
        if not address:
            raise InvoiceError(f"line {line}: address is empty")
        rows.append(
            UsageRow(
                address=address,
                tasks=_parse_int(cells[positions["tasks"]], "tasks", line),
                completed=_parse_int(cells[positions["completed"]], "completed", line),
                pending=_parse_int(cells[positions["pending"]], "pending", line),
                failed=_parse_int(cells[positions["failed"]], "failed", line),
                fees_committed=_parse_int(
                    cells[positions["feesCommitted"]], "feesCommitted", line
                ),
                balance=_parse_int(cells[positions["balance"]], "balance", line),
                line=line,
            )
        )
    return rows


def aggregate_usage(rows: List[UsageRow]) -> UsageAggregate:
    if not rows:
        raise InvoiceError("CSV contains no data rows")

    by_address: Dict[str, UsageAggregate] = {}
    order: List[str] = []
    for row in rows:
        aggregate = by_address.get(row.address)
        if aggregate is None:
            aggregate = UsageAggregate(address=row.address)
            by_address[row.address] = aggregate
            order.append(row.address)
        aggregate.add(row)

    if len(by_address) > 1:
        listed = ", ".join(order)
        raise InvoiceError(
            "CSV contains multiple tenant addresses "
            f"({listed}); invoice one tenant at a time"
        )
    return by_address[order[0]]


def load_usage_csv(path: Path) -> Tuple[UsageAggregate, bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        detail = exc.strerror or str(exc)
        raise InvoiceError(f"cannot read CSV {path}: {detail}") from None
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InvoiceError(f"CSV {path} is not valid UTF-8: {exc}") from None
    return aggregate_usage(parse_usage_csv(text)), raw


def compute_amount_cents(completed: int, rate: Decimal) -> int:
    if completed < 0:
        raise InvoiceError("completed tasks must not be negative")
    exact = Decimal(completed) * rate
    try:
        rounded = exact.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    except InvalidOperation as exc:
        raise InvoiceError("computed invoice amount is out of range") from exc
    return int(rounded)


def derive_idempotency_key(
    csv_bytes: bytes, tenant: str, rate: Decimal, currency: str
) -> str:
    digest = hashlib.sha256()
    digest.update(csv_bytes)
    digest.update(b"\x00")
    digest.update(tenant.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(format_rate(rate).encode("ascii"))
    digest.update(b"\x00")
    digest.update(currency.encode("ascii"))
    return IDEMPOTENCY_PREFIX + digest.hexdigest()


def _step_key(base: str, step: str) -> str:
    candidate = f"{base}-{step}"
    if len(candidate) <= MAX_IDEMPOTENCY_LENGTH:
        return candidate
    return hashlib.sha256(candidate.encode("utf-8")).hexdigest()


def format_money(amount_cents: int, currency: str) -> str:
    value = (Decimal(amount_cents) / Decimal(100)).quantize(Decimal("0.01"))
    if currency == "usd":
        return f"${value:.2f} USD"
    return f"{value:.2f} {currency.upper()}"


def default_description(tenant: str, usage: UsageAggregate, rate: Decimal) -> str:
    return (
        f"Verinoo usage for {tenant} ({usage.address}): "
        f"{usage.completed} completed tasks at {format_rate(rate)} cents/task"
    )


def _stripe_error_message(body: str) -> str:
    try:
        payload = json.loads(body)
    except ValueError:
        return body.strip()[:300] or "no response body"
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    return body.strip()[:300] or "no response body"


def _redact(text: str, secret: str) -> str:
    if secret and secret in text:
        return text.replace(secret, "[redacted]")
    return text


class StripeClient:
    def __init__(
        self,
        secret_key: str,
        base_url: str = STRIPE_API_BASE,
        timeout: float = 30.0,
    ) -> None:
        if not secret_key:
            raise StripeError("STRIPE_SECRET_KEY is empty")
        self._secret_key = secret_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, str]] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, object]:
        url = self._base_url + path
        data: Optional[bytes] = None
        headers = {
            "Authorization": f"Bearer {self._secret_key}",
            "Accept": "application/json",
            "User-Agent": "verinoo-stripe-invoice/1.0",
        }
        if method == "GET":
            query = urllib.parse.urlencode(params or {})
            if query:
                separator = "&" if "?" in url else "?"
                url = f"{url}{separator}{query}"
        else:
            data = urllib.parse.urlencode(params or {}).encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            if idempotency_key:
                headers["Idempotency-Key"] = idempotency_key

        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            message = _redact(_stripe_error_message(response_body), self._secret_key)
            safe_path = path.split("?", 1)[0]
            raise StripeError(
                _redact(
                    f"Stripe {method} {safe_path} failed with HTTP {exc.code}: {message}",
                    self._secret_key,
                )
            ) from None
        except urllib.error.URLError as exc:
            raise StripeError(
                _redact(f"Stripe request failed: {exc.reason}", self._secret_key)
            ) from None

        try:
            payload = json.loads(body)
        except ValueError as exc:
            raise StripeError("Stripe returned a non-JSON response") from exc
        if not isinstance(payload, dict):
            raise StripeError("Stripe returned an unexpected response")
        return payload

    def get(self, path: str, params: Optional[Dict[str, str]] = None) -> Dict[str, object]:
        return self._request("GET", path, params)

    def post(
        self,
        path: str,
        params: Optional[Dict[str, str]] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, object]:
        return self._request("POST", path, params, idempotency_key)


def find_customer(
    client: StripeClient, email: Optional[str], name: str
) -> Optional[Dict[str, object]]:
    if email:
        payload = client.get("/v1/customers", {"email": email, "limit": "1"})
    else:
        escaped = name.replace("\\", "\\\\").replace('"', '\\"')
        payload = client.get(
            "/v1/customers/search", {"query": f'name:"{escaped}"', "limit": "1"}
        )
    data = payload.get("data") or []
    if not isinstance(data, list) or not data:
        return None
    first = data[0]
    return first if isinstance(first, dict) else None


def _customer_params(tenant: str, email: Optional[str]) -> Dict[str, str]:
    params = {
        "name": tenant,
        "description": f"Verinoo tenant {tenant}",
        "metadata[tenant]": tenant,
    }
    if email:
        params["email"] = email
    return params


def _invoice_item_params(
    customer_id: str,
    amount_cents: int,
    currency: str,
    description: str,
    tenant: str,
    usage: UsageAggregate,
    rate: Decimal,
    csv_sha256: str,
) -> Dict[str, str]:
    return {
        "customer": customer_id,
        "amount": str(amount_cents),
        "currency": currency,
        "description": description,
        "metadata[tenant]": tenant,
        "metadata[address]": usage.address,
        "metadata[completed_tasks]": str(usage.completed),
        "metadata[rate_cents]": format_rate(rate),
        "metadata[csv_sha256]": csv_sha256,
    }


def _invoice_params(
    customer_id: str,
    description: str,
    tenant: str,
    usage: UsageAggregate,
    csv_sha256: str,
    collection_method: str,
) -> Dict[str, str]:
    params = {
        "customer": customer_id,
        "collection_method": collection_method,
        "auto_advance": "false",
        "pending_invoice_items_behavior": "include",
        "description": description,
        "metadata[tenant]": tenant,
        "metadata[address]": usage.address,
        "metadata[completed_tasks]": str(usage.completed),
        "metadata[csv_sha256]": csv_sha256,
    }
    if collection_method == "send_invoice":
        params["days_until_due"] = "30"
    return params


def print_dry_run(
    tenant: str,
    usage: UsageAggregate,
    rate: Decimal,
    currency: str,
    amount_cents: int,
    description: str,
    customer_email: Optional[str],
    idempotency_key: str,
    csv_sha256: str,
    out,
) -> None:
    print("Dry run: no Stripe API calls were made.", file=out)
    print("", file=out)
    print(f"Tenant:            {tenant}", file=out)
    print(f"Address:           {usage.address}", file=out)
    print(f"CSV rows:          {usage.rows} (aggregated by address)", file=out)
    print(f"CSV sha256:        {csv_sha256}", file=out)
    print(f"Tasks:             {usage.tasks}", file=out)
    print(f"Completed:         {usage.completed}", file=out)
    print(f"Pending:           {usage.pending}", file=out)
    print(f"Failed:            {usage.failed}", file=out)
    print(f"feesCommitted:     {usage.fees_committed} (informational, not billed)", file=out)
    print(f"balance:           {usage.balance} (informational, not billed)", file=out)
    print(f"Rate:              {format_rate(rate)} cents per completed task", file=out)
    print(f"Amount:            {format_money(amount_cents, currency)} ({amount_cents} cents)", file=out)
    print(f"Currency:          {currency}", file=out)
    print(f"Customer email:    {customer_email or '(none)'}", file=out)
    print(f"Description:       {description}", file=out)
    print(f"Idempotency key:   {idempotency_key}", file=out)


def run_live(
    secret_key: str,
    tenant: str,
    usage: UsageAggregate,
    rate: Decimal,
    currency: str,
    amount_cents: int,
    description: str,
    customer_email: Optional[str],
    base_key: str,
    csv_sha256: str,
    out,
) -> int:
    if amount_cents <= 0:
        raise InvoiceError("computed invoice amount is 0 cents; nothing to invoice")

    client = StripeClient(secret_key)

    customer = find_customer(client, customer_email, tenant)
    if customer is None:
        customer = client.post(
            "/v1/customers",
            _customer_params(tenant, customer_email),
            _step_key(base_key, "customer"),
        )
    customer_id = customer.get("id")
    if not customer_id:
        raise StripeError("Stripe returned a customer without an id")

    customer_email = customer.get("email") or customer_email
    collection_method = "send_invoice" if customer_email else "charge_automatically"

    item = client.post(
        "/v1/invoiceitems",
        _invoice_item_params(
            str(customer_id),
            amount_cents,
            currency,
            description,
            tenant,
            usage,
            rate,
            csv_sha256,
        ),
        _step_key(base_key, "invoice-item"),
    )
    item_id = item.get("id")
    if not item_id:
        raise StripeError("Stripe returned an invoice item without an id")

    invoice = client.post(
        "/v1/invoices",
        _invoice_params(
            str(customer_id),
            description,
            tenant,
            usage,
            csv_sha256,
            collection_method,
        ),
        _step_key(base_key, "invoice"),
    )
    invoice_id = invoice.get("id")
    if not invoice_id:
        raise StripeError("Stripe returned an invoice without an id")

    quoted_id = urllib.parse.quote(str(invoice_id), safe="")
    finalized = client.post(
        f"/v1/invoices/{quoted_id}/finalize",
        {"auto_advance": "false"},
        _step_key(base_key, "finalize"),
    )

    print("Live invoice created and finalized; it was not paid.", file=out)
    print("", file=out)
    print(f"Tenant:            {tenant}", file=out)
    print(f"Customer:          {customer_id}", file=out)
    print(f"Invoice item:      {item_id}", file=out)
    print(f"Invoice:           {invoice_id}", file=out)
    print(f"Status:            {finalized.get('status', 'unknown')}", file=out)
    print(f"Collection:        {collection_method} (automatic collection off)", file=out)
    print(f"Amount:            {format_money(amount_cents, currency)} ({amount_cents} cents)", file=out)
    hosted = finalized.get("hosted_invoice_url")
    if hosted:
        print(f"Hosted invoice:    {hosted}", file=out)
    print(f"Idempotency key:   {base_key}", file=out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a Stripe invoice from a Verinoo usage CSV export.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="usage CSV from GET /v1/usage/:address/csv",
    )
    parser.add_argument(
        "--tenant",
        required=True,
        help="tenant name used for the customer and invoice metadata",
    )
    parser.add_argument(
        "--rate-cents",
        type=_parse_rate,
        default=Decimal(DEFAULT_RATE_CENTS),
        metavar="CENTS",
        help="cents per completed task; fractional cents are allowed",
    )
    parser.add_argument(
        "--currency",
        default="usd",
        help="three-letter currency code passed to Stripe",
    )
    parser.add_argument(
        "--customer-email",
        default=None,
        help="email used to find or create the Stripe customer",
    )
    parser.add_argument(
        "--description",
        default=None,
        help="invoice and invoice-item description",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="print the invoice summary without calling Stripe (default)",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="create and finalize the invoice; requires STRIPE_SECRET_KEY",
    )
    parser.add_argument(
        "--idempotency-key",
        default=None,
        help="base Idempotency-Key; defaults to a hash of the CSV, tenant, rate and currency",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None, stdout=None, stderr=None) -> int:
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    parser = build_parser()
    args = parser.parse_args(argv)

    tenant = args.tenant.strip()
    if not tenant:
        print("error: --tenant must not be empty", file=err)
        return 2

    currency = args.currency.strip().lower()
    if len(currency) != 3 or any(c not in "abcdefghijklmnopqrstuvwxyz" for c in currency):
        print("error: --currency must be a three-letter ISO code such as usd", file=err)
        return 2

    secret_key = ""
    if args.live:
        secret_key = os.environ.get("STRIPE_SECRET_KEY", "").strip()
        if not secret_key:
            print(
                "error: --live requires STRIPE_SECRET_KEY in the environment",
                file=err,
            )
            return 2

    try:
        usage, raw = load_usage_csv(args.csv)
        rate = args.rate_cents
        amount_cents = compute_amount_cents(usage.completed, rate)
        csv_sha256 = hashlib.sha256(raw).hexdigest()
        description = args.description or default_description(tenant, usage, rate)
        base_key = args.idempotency_key or derive_idempotency_key(
            raw, tenant, rate, currency
        )

        if args.live:
            return run_live(
                secret_key,
                tenant,
                usage,
                rate,
                currency,
                amount_cents,
                description,
                args.customer_email,
                base_key,
                csv_sha256,
                out,
            )

        print_dry_run(
            tenant,
            usage,
            rate,
            currency,
            amount_cents,
            description,
            args.customer_email,
            base_key,
            csv_sha256,
            out,
        )
        return 0
    except InvoiceError as exc:
        print(f"error: {exc}", file=err)
        return 2
    except StripeError as exc:
        print(f"error: {exc}", file=err)
        return 1


if __name__ == "__main__":
    sys.exit(main())
