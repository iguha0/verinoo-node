import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
import urllib.parse
from decimal import Decimal
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stripe_invoice as invoice


HEADER = "address,tasks,completed,pending,failed,feesCommitted,balance\n"


class FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class FakeStripe:
    def __init__(self):
        self.requests = []

    def __call__(self, request, timeout=None):
        parsed = urllib.parse.urlparse(request.full_url)
        body = request.data.decode("utf-8") if request.data else ""
        headers = {key.lower(): value for key, value in request.headers.items()}
        self.requests.append(
            {
                "method": request.get_method(),
                "path": parsed.path,
                "query": urllib.parse.parse_qs(parsed.query, keep_blank_values=True),
                "params": urllib.parse.parse_qs(body, keep_blank_values=True),
                "headers": headers,
            }
        )

        method = request.get_method()
        if method == "GET" and parsed.path == "/v1/customers":
            return FakeResponse({"data": []})
        if method == "GET" and parsed.path == "/v1/customers/search":
            return FakeResponse({"data": [{"id": "cus_search"}]})
        if method == "POST" and parsed.path == "/v1/customers":
            return FakeResponse({"id": "cus_test"})
        if method == "POST" and parsed.path == "/v1/invoiceitems":
            return FakeResponse({"id": "ii_test"})
        if method == "POST" and parsed.path == "/v1/invoices":
            return FakeResponse({"id": "in_test", "status": "draft"})
        if method == "POST" and parsed.path == "/v1/invoices/in_test/finalize":
            return FakeResponse(
                {
                    "id": "in_test",
                    "status": "open",
                    "hosted_invoice_url": "https://invoice.stripe.test/in_test",
                }
            )
        raise AssertionError(f"unexpected Stripe request: {method} {parsed.path}")


class CsvParsingTests(unittest.TestCase):
    def test_parses_and_aggregates_rows_for_one_address(self):
        text = (
            HEADER
            + "ai_tenant,10,8,1,1,100,50\n"
            + "ai_tenant,5,4,0,1,20,10\n"
        )
        rows = invoice.parse_usage_csv(text)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].line, 2)
        self.assertEqual(rows[1].line, 3)

        usage = invoice.aggregate_usage(rows)
        self.assertEqual(usage.address, "ai_tenant")
        self.assertEqual(usage.tasks, 15)
        self.assertEqual(usage.completed, 12)
        self.assertEqual(usage.pending, 1)
        self.assertEqual(usage.failed, 2)
        self.assertEqual(usage.fees_committed, 120)
        self.assertEqual(usage.balance, 60)
        self.assertEqual(usage.rows, 2)

    def test_accepts_utf8_bom_and_crlf(self):
        text = "\ufeff" + HEADER.replace("\n", "\r\n") + "ai_tenant,2,1,1,0,5,0\r\n"
        usage = invoice.aggregate_usage(invoice.parse_usage_csv(text))
        self.assertEqual(usage.completed, 1)

    def test_rejects_empty_csv(self):
        with self.assertRaises(invoice.InvoiceError):
            invoice.parse_usage_csv("")

    def test_rejects_missing_columns(self):
        text = "address,tasks,completed,pending,failed,feesCommitted\nai,1,1,0,0,0\n"
        with self.assertRaises(invoice.InvoiceError) as caught:
            invoice.parse_usage_csv(text)
        self.assertIn("balance", str(caught.exception))

    def test_rejects_duplicate_columns(self):
        text = "address,address,tasks,completed,pending,failed,feesCommitted,balance\nai,ai,1,1,0,0,0,0\n"
        with self.assertRaises(invoice.InvoiceError) as caught:
            invoice.parse_usage_csv(text)
        self.assertIn("duplicate", str(caught.exception).lower())

    def test_rejects_no_data_rows(self):
        with self.assertRaises(invoice.InvoiceError) as caught:
            invoice.aggregate_usage(invoice.parse_usage_csv(HEADER))
        self.assertIn("no data rows", str(caught.exception))

    def test_rejects_malformed_numbers(self):
        text = HEADER + "ai,one,0,0,0,0,0\n"
        with self.assertRaises(invoice.InvoiceError) as caught:
            invoice.parse_usage_csv(text)
        self.assertIn("tasks", str(caught.exception))

    def test_rejects_negative_numbers(self):
        text = HEADER + "ai,1,-1,0,0,0,0\n"
        with self.assertRaises(invoice.InvoiceError) as caught:
            invoice.parse_usage_csv(text)
        self.assertIn("negative", str(caught.exception))

    def test_rejects_wrong_column_count(self):
        text = HEADER + "ai,1,1,0,0,0\n"
        with self.assertRaises(invoice.InvoiceError) as caught:
            invoice.parse_usage_csv(text)
        self.assertIn("columns", str(caught.exception))

    def test_rejects_multiple_addresses(self):
        text = HEADER + "ai_one,1,1,0,0,0,0\nai_two,1,1,0,0,0,0\n"
        with self.assertRaises(invoice.InvoiceError) as caught:
            invoice.aggregate_usage(invoice.parse_usage_csv(text))
        self.assertIn("multiple tenant addresses", str(caught.exception))

    def test_load_usage_csv_reads_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "usage.csv"
            path.write_text(HEADER + "ai,3,2,1,0,7,0\n", encoding="utf-8")
            usage, raw = invoice.load_usage_csv(path)
            self.assertEqual(usage.completed, 2)
            self.assertEqual(raw, path.read_bytes())


class AmountTests(unittest.TestCase):
    def test_rate_totals(self):
        cases = [
            (1000, "0.5", 500),
            (1000, "0.05", 50),
            (3, "0.5", 2),
            (1, "0.5", 1),
            (7, "2", 14),
            (0, "0.5", 0),
        ]
        for completed, rate, expected in cases:
            with self.subTest(completed=completed, rate=rate):
                self.assertEqual(
                    invoice.compute_amount_cents(completed, Decimal(rate)), expected
                )

    def test_format_money(self):
        self.assertEqual(invoice.format_money(500, "usd"), "$5.00 USD")
        self.assertEqual(invoice.format_money(50, "usd"), "$0.50 USD")
        self.assertEqual(invoice.format_money(123, "eur"), "1.23 EUR")


class IdempotencyTests(unittest.TestCase):
    def test_key_is_stable_and_canonicalises_rate(self):
        raw = (HEADER + "ai,2,2,0,0,0,0\n").encode("utf-8")
        first = invoice.derive_idempotency_key(raw, "acme", Decimal("0.5"), "usd")
        second = invoice.derive_idempotency_key(raw, "acme", Decimal("0.50"), "usd")
        self.assertEqual(first, second)
        self.assertTrue(first.startswith(invoice.IDEMPOTENCY_PREFIX))

    def test_key_changes_with_csv_tenant_rate_and_currency(self):
        raw = (HEADER + "ai,2,2,0,0,0,0\n").encode("utf-8")
        base = invoice.derive_idempotency_key(raw, "acme", Decimal("0.5"), "usd")
        self.assertNotEqual(
            base,
            invoice.derive_idempotency_key(raw + b"\n", "acme", Decimal("0.5"), "usd"),
        )
        self.assertNotEqual(
            base,
            invoice.derive_idempotency_key(raw, "other", Decimal("0.5"), "usd"),
        )
        self.assertNotEqual(
            base,
            invoice.derive_idempotency_key(raw, "acme", Decimal("1"), "usd"),
        )
        self.assertNotEqual(
            base,
            invoice.derive_idempotency_key(raw, "acme", Decimal("0.5"), "eur"),
        )


class CommandTests(unittest.TestCase):
    def _csv(self, directory, body="ai_tenant,10,8,1,1,100,50\n"):
        path = Path(directory) / "usage.csv"
        path.write_text(HEADER + body, encoding="utf-8")
        return path

    def test_dry_run_makes_no_network_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._csv(directory)
            out = io.StringIO()
            err = io.StringIO()
            with mock.patch.object(invoice.urllib.request, "urlopen") as urlopen:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = invoice.main(["--csv", str(path), "--tenant", "acme"])
            urlopen.assert_not_called()
            self.assertEqual(code, 0)
            self.assertEqual(err.getvalue(), "")
            self.assertIn("Dry run", out.getvalue())
            self.assertIn("$0.04 USD", out.getvalue())

    def test_live_without_key_fails_before_network(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._csv(directory)
            env = dict(os.environ)
            env.pop("STRIPE_SECRET_KEY", None)
            out = io.StringIO()
            err = io.StringIO()
            with mock.patch.dict(os.environ, env, clear=True):
                with mock.patch.object(invoice.urllib.request, "urlopen") as urlopen:
                    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                        code = invoice.main(
                            ["--csv", str(path), "--tenant", "acme", "--live"]
                        )
            urlopen.assert_not_called()
            self.assertNotEqual(code, 0)
            self.assertIn("STRIPE_SECRET_KEY", err.getvalue())
            self.assertNotIn("sk_", out.getvalue() + err.getvalue())

    def test_live_flow_uses_expected_stripe_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._csv(
                directory, "ai_tenant,1200,1000,150,50,1234,5678\n"
            )
            fake = FakeStripe()
            out = io.StringIO()
            err = io.StringIO()
            env = dict(os.environ)
            env["STRIPE_SECRET_KEY"] = "sk_test_example"
            with mock.patch.dict(os.environ, env, clear=True):
                with mock.patch.object(
                    invoice.urllib.request, "urlopen", side_effect=fake
                ):
                    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                        code = invoice.main(
                            [
                                "--csv",
                                str(path),
                                "--tenant",
                                "acme",
                                "--customer-email",
                                "billing@example.test",
                                "--live",
                            ]
                        )
            self.assertEqual(code, 0, err.getvalue())
            self.assertEqual(
                [request["method"] for request in fake.requests],
                ["GET", "POST", "POST", "POST", "POST"],
            )
            self.assertEqual(
                [request["path"] for request in fake.requests],
                [
                    "/v1/customers",
                    "/v1/customers",
                    "/v1/invoiceitems",
                    "/v1/invoices",
                    "/v1/invoices/in_test/finalize",
                ],
            )
            self.assertFalse(
                any("/pay" in request["path"] for request in fake.requests)
            )

            by_path = {
                (request["method"], request["path"]): request
                for request in fake.requests
            }
            self.assertEqual(
                by_path[("GET", "/v1/customers")]["query"]["email"],
                ["billing@example.test"],
            )
            self.assertEqual(
                by_path[("GET", "/v1/customers")]["query"]["limit"], ["1"]
            )
            self.assertEqual(
                by_path[("POST", "/v1/customers")]["params"]["name"], ["acme"]
            )
            self.assertEqual(
                by_path[("POST", "/v1/customers")]["params"]["metadata[tenant]"],
                ["acme"],
            )
            self.assertEqual(
                by_path[("POST", "/v1/invoiceitems")]["params"]["amount"], ["500"]
            )
            self.assertEqual(
                by_path[("POST", "/v1/invoiceitems")]["params"]["currency"], ["usd"]
            )
            self.assertEqual(
                by_path[("POST", "/v1/invoiceitems")]["params"]["customer"],
                ["cus_test"],
            )
            self.assertEqual(
                by_path[("POST", "/v1/invoices")]["params"]["collection_method"],
                ["send_invoice"],
            )
            self.assertEqual(
                by_path[("POST", "/v1/invoices")]["params"]["days_until_due"], ["30"]
            )
            self.assertEqual(
                by_path[("POST", "/v1/invoices")]["params"]["auto_advance"], ["false"]
            )
            self.assertEqual(
                by_path[("POST", "/v1/invoices")]["params"][
                    "pending_invoice_items_behavior"
                ],
                ["include"],
            )

            raw = path.read_bytes()
            base = invoice.derive_idempotency_key(
                raw, "acme", Decimal("0.5"), "usd"
            )
            for request in fake.requests:
                self.assertEqual(
                    request["headers"]["authorization"], "Bearer sk_test_example"
                )
            for request in fake.requests[1:]:
                self.assertIn("idempotency-key", request["headers"])
            self.assertTrue(
                by_path[("POST", "/v1/customers")]["headers"][
                    "idempotency-key"
                ].startswith(base)
            )
            self.assertTrue(
                by_path[("POST", "/v1/invoiceitems")]["headers"][
                    "idempotency-key"
                ].startswith(base)
            )
            self.assertTrue(
                by_path[("POST", "/v1/invoices")]["headers"][
                    "idempotency-key"
                ].startswith(base)
            )
            self.assertTrue(
                by_path[("POST", "/v1/invoices/in_test/finalize")]["headers"][
                    "idempotency-key"
                ].startswith(base)
            )
            self.assertEqual(
                by_path[("POST", "/v1/invoices/in_test/finalize")]["params"][
                    "auto_advance"
                ],
                ["false"],
            )
            self.assertIn("not paid", out.getvalue())

    def test_live_finds_customer_by_name_without_email(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._csv(directory)
            fake = FakeStripe()
            env = dict(os.environ)
            env["STRIPE_SECRET_KEY"] = "sk_test_example"
            with mock.patch.dict(os.environ, env, clear=True):
                with mock.patch.object(
                    invoice.urllib.request, "urlopen", side_effect=fake
                ):
                    with contextlib.redirect_stdout(io.StringIO()):
                        code = invoice.main(
                            ["--csv", str(path), "--tenant", "acme", "--live"]
                        )
            self.assertEqual(code, 0)
            self.assertEqual(fake.requests[0]["path"], "/v1/customers/search")
            self.assertIn("acme", fake.requests[0]["query"]["query"][0])
            self.assertEqual(fake.requests[1]["path"], "/v1/invoiceitems")
            self.assertEqual(
                fake.requests[1]["params"]["customer"], ["cus_search"]
            )
            self.assertEqual(fake.requests[1]["params"]["amount"], ["4"])
            self.assertEqual(
                fake.requests[2]["params"]["collection_method"],
                ["charge_automatically"],
            )
            self.assertNotIn("days_until_due", fake.requests[2]["params"])

    def test_api_error_does_not_leak_secret_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._csv(directory)
            secret = "sk_test_secret_value"
            body = json.dumps(
                {"error": {"message": f"Invalid API Key provided: {secret}"}}
            ).encode("utf-8")
            error = urllib.error.HTTPError(
                "https://api.stripe.com/v1/customers",
                401,
                "Unauthorized",
                {},
                io.BytesIO(body),
            )
            env = dict(os.environ)
            env["STRIPE_SECRET_KEY"] = secret
            out = io.StringIO()
            err = io.StringIO()
            with mock.patch.dict(os.environ, env, clear=True):
                with mock.patch.object(
                    invoice.urllib.request, "urlopen", side_effect=error
                ):
                    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                        code = invoice.main(
                            ["--csv", str(path), "--tenant", "acme", "--live"]
                        )
            self.assertEqual(code, 1)
            self.assertNotIn(secret, out.getvalue() + err.getvalue())
            self.assertIn("[redacted]", err.getvalue())


if __name__ == "__main__":
    unittest.main()
