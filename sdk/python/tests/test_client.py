import base64
import io
import json
import urllib.error

import pytest
from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
)

from verinoo import (
    Client,
    VerinooError,
    address_from_public_key,
    build_inference_data,
    build_signed_tx,
    canonical_json,
    encrypt_input,
    generate_wallet,
    load_wallet,
    sha256_hex,
)

TRANSFER = {
    "from": "ai_c0334217fc39036aad7cd9cb15123f3f02d9ff40",
    "to": "ai_db63de2209ed3acb318ae00d5fe44037f5742813",
    "value": 250,
    "nonce": 1,
    "data": {"type": "transfer", "data": {}},
}
TRANSFER_CANONICAL = (
    '{"data":{"data":{},"type":"transfer"},"from":'
    '"ai_c0334217fc39036aad7cd9cb15123f3f02d9ff40","nonce":1,"to":'
    '"ai_db63de2209ed3acb318ae00d5fe44037f5742813","value":250}'
)
TRANSFER_TX_ID = (
    "c98970ee550228b926be07b5a863f1dfac6daff55c318d771abaa02684217c17"
)

SEED = bytes(range(32))
SEED_B64 = base64.b64encode(SEED).decode("ascii")
EXPECTED_ADDRESS = "ai_56475aa75463474c0285df5dbf2bcab73da65135"
EXPECTED_PUBLIC_KEY = (
    "03a107bff3ce10be1d70dd18e74bc09967e4d6309ba50d5f1ddc8664125531b8"
)
EXPECTED_CANONICAL = (
    '{"data":{"data":{"deadline":1700000000,"input":"hello world",'
    '"inputCommitment":'
    '"b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",'
    '"maxFee":100,"requester":"ai_56475aa75463474c0285df5dbf2bcab73da65135",'
    '"targetModel":"intent-classifier-v1","verificationType":"optimistic"},'
    '"type":"submitInference"},"from":'
    '"ai_56475aa75463474c0285df5dbf2bcab73da65135","nonce":3,"to":"","value":0}'
)
EXPECTED_TX_ID = (
    "fd2318e902c051cfffadf7a20c5da21160f53fabcbd4f100d8c5f903fb6563d8"
)
EXPECTED_SIGNATURE = (
    "ebeb325f38e74831eaeda317484c313ea95913e3a0bd99b9004c5331c1874c61"
    "5cbc27c751b795d2e415a8d23b496dec35fda8e0b66f5a8d46c3340547a9bc0c"
)


def inference_vector():
    wallet = load_wallet({"privateKey": SEED_B64})
    data = build_inference_data(
        from_address=wallet.address,
        input_text="hello world",
        model="intent-classifier-v1",
        max_fee=100,
        deadline=1700000000,
    )
    return wallet, data


def test_canonical_json_matches_rust_canonical_encoding():
    reordered = {
        "nonce": 1,
        "value": 250,
        "to": TRANSFER["to"],
        "from": TRANSFER["from"],
        "data": {"data": {}, "type": "transfer"},
    }
    assert canonical_json(reordered) == TRANSFER_CANONICAL
    assert sha256_hex(canonical_json(reordered)) == TRANSFER_TX_ID

    assert (
        canonical_json({"z": [{"b": 2, "a": 1}, "x"], "a": {"d": 4, "c": 3}})
        == '{"a":{"c":3,"d":4},"z":[{"a":1,"b":2},"x"]}'
    )
    assert canonical_json({"s": "hello world"}) == '{"s":"hello world"}'
    assert canonical_json({"b": True, "n": None}) == '{"b":true,"n":null}'

    with pytest.raises(ValueError, match="float"):
        canonical_json({"n": 1.5})
    with pytest.raises(ValueError, match="out of range"):
        canonical_json({"n": 2**64})


def test_tx_id_and_signature_vector_are_stable():
    wallet, data = inference_vector()
    assert wallet.address == EXPECTED_ADDRESS
    assert wallet.public_key == EXPECTED_PUBLIC_KEY

    envelope = build_signed_tx(
        signing_key=wallet.signing_key,
        from_address=wallet.address,
        data=data,
        nonce=3,
    )

    assert envelope["txId"] == EXPECTED_TX_ID
    assert envelope["signature"] == EXPECTED_SIGNATURE
    assert envelope["publicKey"] == EXPECTED_PUBLIC_KEY
    assert "gasLimit" not in envelope
    assert "gasPrice" not in envelope
    assert (
        canonical_json(
            {
                "data": data,
                "from": wallet.address,
                "nonce": 3,
                "to": "",
                "value": 0,
            }
        )
        == EXPECTED_CANONICAL
    )

    public_key = Ed25519PublicKey.from_public_bytes(
        bytes.fromhex(envelope["publicKey"])
    )
    public_key.verify(
        bytes.fromhex(envelope["signature"]),
        envelope["txId"].encode("utf-8"),
    )
    with pytest.raises(InvalidSignature):
        public_key.verify(
            bytes.fromhex(envelope["signature"]),
            ("00" * 32).encode("utf-8"),
        )


def test_wallet_generation_and_rust_cli_key_file_loading(tmp_path):
    generated = generate_wallet()
    assert generated["address"].startswith("ai_")
    assert len(generated["address"]) == 43
    assert len(generated["publicKey"]) == 64
    assert len(base64.b64decode(generated["privateKey"])) == 32
    assert address_from_public_key(generated["publicKey"]) == generated["address"]

    path = tmp_path / "key.json"
    path.write_text(json.dumps(generated, indent=2), encoding="utf-8")
    loaded = load_wallet(str(path))
    assert loaded.address == generated["address"]
    assert loaded.public_key == generated["publicKey"]
    assert loaded.to_dict() == generated

    with pytest.raises(ValueError, match="does not match"):
        load_wallet(
            {
                **generated,
                "address": "ai_" + "00" * 20,
            }
        )


def test_input_encryption_shape_commitment_and_round_trip():
    network = X25519PrivateKey.generate()
    network_public = network.public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw
    )

    result = encrypt_input("hello world", network_public.hex())
    commitment = result["inputCommitment"]
    encrypted = result["encryptedInput"]

    assert set(encrypted) == {"ephemeralPublicKey", "nonce", "ciphertext"}
    assert len(bytes.fromhex(encrypted["ephemeralPublicKey"])) == 32
    assert len(bytes.fromhex(encrypted["nonce"])) == 12
    assert commitment == sha256_hex("hello world")
    ciphertext = base64.b64decode(encrypted["ciphertext"])
    assert len(ciphertext) == len(b"hello world") + 16

    ephemeral_public = X25519PublicKey.from_public_bytes(
        bytes.fromhex(encrypted["ephemeralPublicKey"])
    )
    shared = network.exchange(ephemeral_public)
    key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=network_public + bytes.fromhex(encrypted["ephemeralPublicKey"]),
        info=b"verinoo-input-v1",
    ).derive(shared)
    plaintext = ChaCha20Poly1305(key).decrypt(
        bytes.fromhex(encrypted["nonce"]),
        ciphertext,
        bytes.fromhex(commitment),
    )
    assert plaintext == b"hello world"

    with pytest.raises(InvalidTag):
        ChaCha20Poly1305(key).decrypt(
            bytes.fromhex(encrypted["nonce"]),
            ciphertext,
            b"\x00" * 32,
        )


def test_infer_uses_legacy_input_when_node_has_no_key(monkeypatch):
    client = Client(base_url="https://node.test", api_key="tenant-key")
    wallet = load_wallet({"privateKey": SEED_B64})
    calls = []

    def fake_request(method, path, body=None, api_key=None, raw=False):
        calls.append(
            {
                "method": method,
                "path": path,
                "body": body,
                "api_key": api_key if api_key is not None else client.api_key,
                "raw": raw,
            }
        )
        if path.startswith("/account/"):
            return {"balance": 1_000_000, "nonce": 4}
        if path == "/v1/input-key":
            raise VerinooError(
                'HTTP 404: {"error":"not found"}',
                status=404,
                body={"error": "not found"},
            )
        if path == "/v1/infer":
            return {"accepted": True, "taskId": "task-1"}
        raise AssertionError(f"unexpected {path}")

    monkeypatch.setattr(client, "_request", fake_request)
    result = client.infer(wallet, "hi")
    assert result["taskId"] == "task-1"

    post = next(call for call in calls if call["path"] == "/v1/infer")
    envelope = post["body"]
    assert envelope["nonce"] == 5
    assert envelope["data"]["type"] == "submitInference"
    assert envelope["data"]["data"]["input"] == "hi"
    assert "encryptedInput" not in envelope["data"]["data"]
    assert post["api_key"] == "tenant-key"


def test_infer_encrypts_when_network_key_is_advertised(monkeypatch):
    network = X25519PrivateKey.generate()
    network_public = network.public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw
    )
    client = Client(base_url="https://node.test")
    wallet = load_wallet({"privateKey": SEED_B64})
    calls = []

    def fake_request(method, path, body=None, api_key=None, raw=False):
        calls.append({"path": path, "body": body})
        if path.startswith("/account/"):
            return {"balance": 1, "nonce": 0}
        if path == "/v1/input-key":
            return {
                "publicKey": network_public.hex(),
                "required": False,
                "algorithm": "x25519-hkdf-sha256-chacha20poly1305",
            }
        if path == "/v1/infer":
            return {"accepted": True}
        raise AssertionError(f"unexpected {path}")

    monkeypatch.setattr(client, "_request", fake_request)
    client.infer(wallet, "secret text", max_fee=100)

    post = next(call for call in calls if call["path"] == "/v1/infer")
    inner = post["body"]["data"]["data"]
    assert "input" not in inner
    assert set(inner["encryptedInput"]) == {
        "ephemeralPublicKey",
        "nonce",
        "ciphertext",
    }
    assert inner["inputCommitment"] == sha256_hex("secret text")
    assert post["body"]["nonce"] == 1


def test_infer_honors_required_encryption_when_plaintext_requested(monkeypatch):
    network = X25519PrivateKey.generate()
    network_public = network.public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw
    )
    client = Client(base_url="https://node.test")
    wallet = load_wallet({"privateKey": SEED_B64})
    calls = []

    def fake_request(method, path, body=None, api_key=None, raw=False):
        calls.append({"path": path, "body": body})
        if path.startswith("/account/"):
            return {"balance": 1, "nonce": 0}
        if path == "/v1/input-key":
            return {
                "publicKey": network_public.hex(),
                "required": True,
                "algorithm": "x25519-hkdf-sha256-chacha20poly1305",
            }
        if path == "/v1/infer":
            return {"accepted": True}
        raise AssertionError(f"unexpected {path}")

    monkeypatch.setattr(client, "_request", fake_request)
    client.infer(wallet, "must encrypt", encrypt=False)

    post = next(call for call in calls if call["path"] == "/v1/infer")
    inner = post["body"]["data"]["data"]
    assert "input" not in inner
    assert "encryptedInput" in inner


def test_errors_carry_http_status_and_json_body(monkeypatch):
    def raise_http_error(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "Bad Request",
            None,
            io.BytesIO(b'{"error":"nope"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", raise_http_error)
    client = Client(base_url="https://node.test")
    with pytest.raises(VerinooError) as info:
        client.models()
    assert info.value.status == 400
    assert info.value.body == {"error": "nope"}
    assert "HTTP 400:" in str(info.value)
