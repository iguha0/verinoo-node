# Verinoo client SDKs

Both clients implement the exact Rust-node transaction semantics:

- canonical JSON (sorted keys, no whitespace, integers only)
- `txId = sha256_hex(canonical_json({data, from, nonce, to, value}))`
- Ed25519 signature over the UTF-8 `txId`, hex encoded
- inference input encrypted to the network X25519 key returned by
  `GET /v1/input-key` (X25519 + HKDF-SHA256 + ChaCha20-Poly1305)
- faucet, account nonce lookup, task polling, verification and usage

| SDK | Requirements | Test |
|---|---|---|
| [`typescript`](typescript) | Node >= 20, no npm install | `cd typescript && node --test` |
| [`python`](python) | Python 3.9+, `cryptography` | `cd python && python3 -m pytest` |

The base URL defaults to `https://verinoo.com`. Use a node you run yourself for
private models.

**Privacy:** the input is encrypted to the validator network key. API and block
readers see ciphertext, but every validator (including the operator) can
decrypt. See [`../docs/privacy.md`](../docs/privacy.md).
