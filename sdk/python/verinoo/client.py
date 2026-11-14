"""Verinoo Python SDK.

Transaction identity matches the Rust node exactly::

    txId = sha256_hex(canonical_json({data, from, nonce, to, value}))
    signature = Ed25519(txId UTF-8 bytes), hex encoded

Inference input is encrypted to the validator network X25519 key with
HKDF-SHA256 and ChaCha20-Poly1305. The operator can decrypt it; this is
encryption to the network, not end-to-end privacy from the operator.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

DEFAULT_BASE_URL = "https://verinoo.com"
DEFAULT_MODEL = "intent-classifier-v1"
DEFAULT_MAX_FEE = 100
INPUT_KEY_INFO = b"verinoo-input-v1"

_HEX_RE = re.compile(r"^[0-9a-fA-F]*$")
_SEED_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class VerinooError(Exception):
    """HTTP/API error carrying the status code and decoded JSON body."""

    def __init__(
        self,
        message: str,
        status: Optional[int] = None,
        body: Any = None,
        body_text: str = "",
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.body = body
        self.body_text = body_text
        self.cause = cause


def _utf8(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    raise TypeError("value must be str or bytes")


def sha256_hex(value: Any) -> str:
    return hashlib.sha256(_utf8(value)).hexdigest()


def _canonical_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def canonical_json(value: Any) -> str:
    """Return Verinoo canonical JSON.

    Object keys are sorted recursively, arrays keep order, output has no
    whitespace, and only integers in Rust's i64/u64 range are accepted.
    """
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _canonical_string(value)
    if isinstance(value, int):
        if value < -(2**63) or value > 2**64 - 1:
            raise ValueError(
                f"number out of range for canonical encoding: {value}"
            )
        return str(value)
    if isinstance(value, float):
        raise ValueError(f"float not allowed in canonical data: {value}")
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(canonical_json(item) for item in value) + "]"
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise TypeError("canonical JSON object keys must be strings")
        parts = []
        for key in sorted(value):
            parts.append(
                f"{_canonical_string(key)}:{canonical_json(value[key])}"
            )
        return "{" + ",".join(parts) + "}"
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def address_from_public_key(public_key_hex: str) -> str:
    public_key = bytes.fromhex(public_key_hex)
    if len(public_key) != 32:
        raise ValueError("public key must be 32 bytes")
    return "ai_" + hashlib.sha256(public_key).hexdigest()[:40]


def input_commitment(plaintext: str) -> str:
    return sha256_hex(plaintext)


class Wallet:
    """An Ed25519 signing identity.

    ``repr`` deliberately omits private key material.
    """

    __slots__ = ("address", "public_key", "private_key", "signing_key")

    def __init__(
        self,
        address: str,
        public_key: str,
        private_key: str,
        signing_key: Ed25519PrivateKey,
    ) -> None:
        self.address = address
        self.public_key = public_key
        self.private_key = private_key
        self.signing_key = signing_key

    def __repr__(self) -> str:
        return f"Wallet(address={self.address!r})"

    def to_dict(self) -> dict[str, str]:
        """Return the Rust CLI key-file shape (contains the private key)."""
        return {
            "publicKey": self.public_key,
            "privateKey": self.private_key,
            "address": self.address,
        }


def _decode_seed(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("privateKey is empty")
        if _SEED_HEX_RE.fullmatch(text):
            raw = bytes.fromhex(text)
        else:
            try:
                raw = base64.b64decode(text, validate=True)
            except Exception:
                raise ValueError(
                    "privateKey must be base64 or 64-character hex"
                ) from None
    else:
        raise TypeError(
            "privateKey must be a base64/hex string, bytes, or Ed25519PrivateKey"
        )
    if len(raw) < 32:
        raise ValueError("Ed25519 private key must be at least 32 bytes")
    return raw[:32]


def _read_wallet_source(source: Any) -> dict[str, Any]:
    if isinstance(source, Ed25519PrivateKey):
        return {"privateKey": source}
    if isinstance(source, (bytes, bytearray, memoryview)):
        return {"privateKey": bytes(source)}
    if isinstance(source, str):
        text = source.strip()
        if text.startswith("{"):
            return json.loads(text)
        if os.path.isfile(text):
            with open(text, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return {"privateKey": text}
    if isinstance(source, Wallet):
        return {
            "privateKey": source.signing_key,
            "publicKey": source.public_key,
            "address": source.address,
        }
    if isinstance(source, dict):
        return source
    raise TypeError(
        "wallet must be a key-file path, wallet dict, base64 key, or "
        "Ed25519PrivateKey"
    )


def load_wallet(source: Any) -> Wallet:
    """Load a wallet from the Rust CLI JSON shape, dict, path, or raw key."""
    wallet = _read_wallet_source(source)
    private_key = wallet.get("privateKey")
    signing_key = (
        private_key
        if isinstance(private_key, Ed25519PrivateKey)
        else Ed25519PrivateKey.from_private_bytes(_decode_seed(private_key))
    )
    seed = signing_key.private_bytes(
        Encoding.Raw, PrivateFormat.Raw, NoEncryption()
    )
    public_key = signing_key.public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw
    ).hex()
    address = address_from_public_key(public_key)

    declared_public = wallet.get("publicKey")
    if declared_public is not None and str(declared_public).lower() != public_key:
        raise ValueError("wallet publicKey does not match privateKey")
    declared_address = wallet.get("address")
    if declared_address is not None and declared_address != address:
        raise ValueError(
            f"wallet address {declared_address} does not match privateKey "
            f"address {address}"
        )

    return Wallet(
        address=address,
        public_key=public_key,
        private_key=base64.b64encode(seed).decode("ascii"),
        signing_key=signing_key,
    )


def generate_wallet() -> dict[str, str]:
    """Generate a wallet in the same shape as the Rust CLI key file."""
    signing_key = Ed25519PrivateKey.generate()
    seed = signing_key.private_bytes(
        Encoding.Raw, PrivateFormat.Raw, NoEncryption()
    )
    public_key = signing_key.public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw
    ).hex()
    return {
        "publicKey": public_key,
        "privateKey": base64.b64encode(seed).decode("ascii"),
        "address": address_from_public_key(public_key),
    }


def build_signed_tx(
    signing_key: Ed25519PrivateKey,
    from_address: str,
    data: Any,
    nonce: int,
    to: str = "",
    value: int = 0,
    gas_limit: Optional[int] = None,
    gas_price: Optional[int] = None,
) -> dict[str, Any]:
    """Build a signed transaction envelope."""
    payload: dict[str, Any] = {
        "data": data,
        "from": from_address,
        "nonce": nonce,
        "to": to,
        "value": value,
    }
    if gas_limit is not None:
        payload["gasLimit"] = gas_limit
    if gas_price is not None:
        payload["gasPrice"] = gas_price

    tx_id = sha256_hex(canonical_json(payload))
    signature = signing_key.sign(tx_id.encode("utf-8")).hex()
    public_key = signing_key.public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw
    ).hex()

    envelope: dict[str, Any] = {
        "txId": tx_id,
        "from": from_address,
        "to": to,
        "value": value,
        "nonce": nonce,
        "data": data,
        "publicKey": public_key,
        "signature": signature,
    }
    if gas_limit is not None:
        envelope["gasLimit"] = gas_limit
    if gas_price is not None:
        envelope["gasPrice"] = gas_price
    return envelope


def build_inference_data(
    from_address: str,
    input_text: str,
    model: str = DEFAULT_MODEL,
    max_fee: int = DEFAULT_MAX_FEE,
    deadline: int = 0,
    encrypted_input: Optional[dict[str, str]] = None,
    commitment: Optional[str] = None,
) -> dict[str, Any]:
    """Assemble the ``submitInference`` TxData (encrypted or plaintext)."""
    inner: dict[str, Any] = {
        "requester": from_address,
        "targetModel": model,
        "inputCommitment": commitment
        if commitment is not None
        else input_commitment(input_text),
        "maxFee": max_fee,
        "deadline": deadline,
        "verificationType": "optimistic",
    }
    if encrypted_input is not None:
        inner["encryptedInput"] = encrypted_input
    else:
        inner["input"] = input_text
    return {"type": "submitInference", "data": inner}


def encrypt_input(
    plaintext: str,
    network_public_key_hex: str,
    ephemeral_private_key: Optional[X25519PrivateKey] = None,
    nonce: Optional[bytes] = None,
) -> dict[str, Any]:
    """Encrypt input to the validator network X25519 public key.

    Returns ``{"inputCommitment": hex, "encryptedInput": {...}}`` where
    ``encryptedInput`` has exactly ``ephemeralPublicKey`` (hex), ``nonce``
    (hex), and ``ciphertext`` (base64, includes the Poly1305 tag).
    """
    plaintext_bytes = _utf8(plaintext)
    network_raw = bytes.fromhex(network_public_key_hex)
    if len(network_raw) != 32:
        raise ValueError("network public key must be 32 bytes")
    network_public = X25519PublicKey.from_public_bytes(network_raw)

    ephemeral = ephemeral_private_key or X25519PrivateKey.generate()
    ephemeral_public = ephemeral.public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw
    )
    shared = ephemeral.exchange(network_public)
    key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=network_raw + ephemeral_public,
        info=INPUT_KEY_INFO,
    ).derive(shared)

    nonce_bytes = nonce if nonce is not None else os.urandom(12)
    if len(nonce_bytes) != 12:
        raise ValueError("ChaCha20-Poly1305 nonce must be 12 bytes")

    commitment = input_commitment(plaintext)
    ciphertext = ChaCha20Poly1305(key).encrypt(
        nonce_bytes, plaintext_bytes, bytes.fromhex(commitment)
    )

    return {
        "inputCommitment": commitment,
        "encryptedInput": {
            "ephemeralPublicKey": ephemeral_public.hex(),
            "nonce": nonce_bytes.hex(),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        },
    }


class Client:
    """Synchronous Verinoo HTTP client using urllib (no requests)."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        body: Any = None,
        api_key: Optional[str] = None,
        raw: bool = False,
    ) -> Any:
        url = f"{self.base_url}{path}"
        headers = {"Accept": "*/*" if raw else "application/json"}
        payload = None
        if body is not None:
            payload = json.dumps(
                body, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            headers["Content-Type"] = "application/json"
        key = api_key if api_key is not None else self.api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"

        request = urllib.request.Request(
            url, data=payload, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout
            ) as response:
                text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            parsed = None
            try:
                parsed = json.loads(text) if text else None
            except json.JSONDecodeError:
                parsed = None
            raise VerinooError(
                f"HTTP {exc.code}: {text}",
                status=exc.code,
                body=parsed,
                body_text=text,
                cause=exc,
            ) from None
        except OSError as exc:
            raise VerinooError(
                f"request failed: {exc}", cause=exc
            ) from None

        if raw:
            return text
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def models(self) -> Any:
        return self._request("GET", "/v1/models")

    def input_key(self) -> Any:
        return self._request("GET", "/v1/input-key")

    inputKey = input_key

    def faucet(self, address: str) -> Any:
        return self._request("POST", "/faucet", body={"address": address})

    def account(self, address: str) -> Any:
        return self._request("GET", f"/account/{urllib.parse.quote(address)}")

    def task(self, task_id: str) -> Any:
        return self._request(
            "GET", f"/v1/tasks/{urllib.parse.quote(task_id)}"
        )

    def wait_for_task(
        self, task_id: str, timeout: float = 120.0, interval: float = 1.0
    ) -> Any:
        """Poll until the task is executed in a block.

        A 404 before inclusion is expected: the task id exists only once a
        block carries the transaction.
        """
        deadline = time.monotonic() + timeout
        last_error: Optional[VerinooError] = None
        while time.monotonic() < deadline:
            try:
                view = self.task(task_id)
                status = view.get("task", {}).get("status", "")
                if status in ("completed", "failed"):
                    return view
            except VerinooError as exc:
                if exc.status != 404:
                    last_error = exc
            time.sleep(interval)
        if last_error is not None:
            raise last_error
        raise VerinooError(f"task {task_id} was not executed within {timeout}s")

    def verify(self, task_id: str) -> Any:
        return self._request(
            "GET", f"/v1/verify/{urllib.parse.quote(task_id)}"
        )

    def usage(self, address: str, api_key: Optional[str] = None) -> Any:
        return self._request(
            "GET",
            f"/v1/usage/{urllib.parse.quote(address)}",
            api_key=api_key,
        )

    def usage_csv(self, address: str, api_key: Optional[str] = None) -> str:
        return self._request(
            "GET",
            f"/v1/usage/{urllib.parse.quote(address)}/csv",
            api_key=api_key,
            raw=True,
        )

    usageCsv = usage_csv

    def infer(
        self,
        private_key: Any,
        input: str,
        model: str = DEFAULT_MODEL,
        max_fee: int = DEFAULT_MAX_FEE,
        api_key: Optional[str] = None,
        encrypt: bool = True,
        deadline: Optional[int] = None,
        nonce: Optional[int] = None,
    ) -> Any:
        """Sign and submit an inference transaction.

        ``private_key`` accepts a wallet dict/file path, base64 seed, or an
        ``Ed25519PrivateKey``. Encryption is on by default. If the network
        advertises no input key, or the caller explicitly asks for plaintext
        and the network does not require encryption, the legacy top-level
        ``input`` field is sent instead of ``encryptedInput``.
        """
        if not isinstance(input, str):
            raise TypeError("input must be a string")
        wallet = load_wallet(private_key)
        account = self.account(wallet.address) or {}
        tx_nonce = (
            nonce
            if nonce is not None
            else int(account.get("nonce", 0) or 0) + 1
        )
        tx_deadline = (
            deadline if deadline is not None else int(time.time()) + 3600
        )

        key_info: Optional[dict[str, Any]] = None
        try:
            key_info = self.input_key()
        except VerinooError as exc:
            if exc.status != 404:
                raise

        network_public_key = (key_info or {}).get("publicKey")
        encrypted_input = None
        commitment = input_commitment(input)
        if network_public_key and (
            encrypt or (key_info or {}).get("required")
        ):
            encrypted = encrypt_input(input, network_public_key)
            encrypted_input = encrypted["encryptedInput"]
            commitment = encrypted["inputCommitment"]

        data = build_inference_data(
            from_address=wallet.address,
            input_text=input,
            model=model,
            max_fee=max_fee,
            deadline=tx_deadline,
            encrypted_input=encrypted_input,
            commitment=commitment,
        )
        envelope = build_signed_tx(
            signing_key=wallet.signing_key,
            from_address=wallet.address,
            data=data,
            nonce=tx_nonce,
        )
        return self._request(
            "POST", "/v1/infer", body=envelope, api_key=api_key
        )
