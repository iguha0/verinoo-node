# Verinoo Python SDK

Synchronous client for the Verinoo inference API using only
[`cryptography`](https://cryptography.io/) and `urllib.request` (no `requests`
dependency). Requires Python >= 3.9.

## Install

```bash
cd sdk/python
python3 -m pip install -e .
```

## Quickstart

```python
from verinoo import Client, generate_wallet

wallet = generate_wallet()  # {"address", "publicKey", "privateKey"(base64)}
client = Client(base_url="https://verinoo.com")

# 1. Fund the address (10 000 testnet units; 60 s cooldown per address).
client.faucet(wallet["address"])

# 2. Submit an inference transaction.
submitted = client.infer(
    private_key=wallet,
    input="my parcel was never delivered",
    model="intent-classifier-v1",  # default
    max_fee=100,                   # default
)
task_id = submitted["taskId"]

# 3. Poll the task until the block executes it.
task = client.wait_for_task(task_id)  # 404 before inclusion is expected
print(task["task"].get("resultOutput"))

# 4. Independently re-run and compare against the on-chain hash.
check = client.verify(task_id)
print(check["verified"], check.get("output"))
```

`private_key` in `infer` accepts a wallet dict, a Rust-CLI key-file path, a
base64/hex seed, or an `Ed25519PrivateKey`. The wallet file shape is the same
as `verinoo-node wallet generate`:

```json
{
  "publicKey": "<64 hex chars>",
  "privateKey": "<base64 of the 32-byte Ed25519 seed>",
  "address": "ai_<40 hex chars>"
}
```

Address derivation is `ai_` + the first 20 bytes of
`sha256(ed25519_public_key_bytes)`, hex encoded, matching the node.

## Transaction semantics

- `txId = sha256_hex(canonical_json({data, from, nonce, to, value}))`
- Canonical JSON sorts object keys recursively, keeps array order, emits no
  whitespace, and accepts integers only.
- The Ed25519 signature is over the UTF-8 bytes of the `txId` string and is
  hex encoded.
- Inference envelopes use `to: ""`, `value: 0`, and
  `nonce = account.nonce + 1`.
- `deadline = int(time.time()) + 3600`.
- `gasLimit` / `gasPrice` are omitted unless explicitly supplied.

## Input encryption

By default `infer` fetches `GET /v1/input-key` and encrypts the input to the
validator network's X25519 public key:

1. Generate an ephemeral X25519 keypair.
2. `shared = X25519(ephemeral_private, network_public)`.
3. `key = HKDF-SHA256(ikm=shared, salt=network_public || ephemeral_public,
   info=b"verinoo-input-v1", length=32)`.
4. `nonce = 12 random bytes`.
5. `ciphertext = ChaCha20-Poly1305(key, nonce, plaintext,
   aad=sha256(plaintext))`; the 16-byte tag is appended and the result is
   base64.
6. Send `encryptedInput = {ephemeralPublicKey: hex, nonce: hex,
   ciphertext: base64}` and `inputCommitment = sha256_hex(plaintext)`.

**This encrypts input to the validator network key, not end-to-end from the
operator.** The validator that runs the model can decrypt the input. It is not
a substitute for client-side handling of sensitive data.

If the endpoint returns no `publicKey`, or `required` is `false` and the caller
passes `encrypt=False`, the legacy top-level `input` string is sent instead.
If `required` is `true`, input stays encrypted even when `encrypt=False`.
A `404` from `/v1/input-key` is treated as "no key advertised" so the SDK also
works against nodes that predate the encryption endpoint.

## API

```python
client = Client(base_url="https://verinoo.com", api_key=None, timeout=30.0)
```

| Method | Endpoint |
|---|---|
| `models()` | `GET /v1/models` |
| `input_key()` / `inputKey()` | `GET /v1/input-key` |
| `faucet(address)` | `POST /faucet` |
| `account(address)` | `GET /account/:address` |
| `infer(private_key, input, ...)` | `POST /v1/infer` |
| `task(id)` / `wait_for_task(id)` | `GET /v1/tasks/:id`, polling until executed |
| `verify(id)` | `GET /v1/verify/:id` |
| `usage(address, api_key=None)` | `GET /v1/usage/:address` |
| `usage_csv(address, api_key=None)` / `usageCsv(...)` | `GET /v1/usage/:address/csv` |

API keys are sent as `Authorization: Bearer <key>`. Errors raise
`VerinooError` with `.status`, `.body` (parsed JSON), and `.body_text`.

## Tests

```bash
cd sdk/python
python3 -m pytest
```

The suite covers canonical JSON against the repo conformance fixture, a fixed
transaction vector shared with the TypeScript SDK, Ed25519 signing
verification, wallet key-file loading, encryption shape/commitment, a full
X25519 + HKDF + ChaCha20-Poly1305 round-trip, and plaintext/encrypted `infer`
selection.
