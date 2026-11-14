# Verinoo TypeScript SDK

Zero-dependency ESM client for the Verinoo inference API. Requires Node >= 20
(built-in `fetch`, WebCrypto-grade `node:crypto` primitives). No `npm install`
is needed to use or test it.

## Quickstart

```js
import { Client, generateWallet } from './src/index.js';

const wallet = generateWallet(); // {address, publicKey, privateKey(base64)}
const client = new Client({ baseUrl: 'https://verinoo.com' });

// 1. Fund the address (10 000 testnet units; 60 s cooldown per address).
await client.faucet(wallet.address);

// 2. Submit an inference transaction.
const { taskId } = await client.infer({
  privateKey: wallet,
  input: 'my parcel was never delivered',
  model: 'intent-classifier-v1', // default
  maxFee: 100,                   // default
});

// 3. Wait until the block executes it (404 before inclusion is expected).
const task = await client.waitForTask(taskId);
console.log(task.task.resultOutput);

// 4. Independently re-run and compare against the on-chain hash.
const check = await client.verify(taskId);
console.log(check.verified, check.output);
```

`privateKey` in `infer` accepts a wallet object, a Rust-CLI key-file path,
a base64/hex seed, or a Node `KeyObject`. The wallet file shape is the same as
`verinoo-node wallet generate`:

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

- `txId = sha256_hex(canonicalJson({data, from, nonce, to, value}))`
- Canonical JSON sorts object keys recursively, keeps array order, emits no
  whitespace, and accepts integers only.
- The Ed25519 signature is over the UTF-8 bytes of the `txId` string and is
  hex encoded.
- Inference envelopes use `to: ""`, `value: 0`, and
  `nonce = account.nonce + 1`.
- `deadline = floor(Date.now() / 1000) + 3600`.
- `gasLimit` / `gasPrice` are omitted unless explicitly supplied.

## Input encryption

By default `infer` fetches `GET /v1/input-key` and encrypts the input to the
validator network's X25519 public key:

1. Generate an ephemeral X25519 keypair.
2. `shared = X25519(ephemeral_private, network_public)`.
3. `key = HKDF-SHA256(ikm=shared, salt=network_public || ephemeral_public,
   info="verinoo-input-v1", length=32)`.
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
passes `encrypt: false`, the legacy top-level `input` string is sent instead.
If `required` is `true`, input stays encrypted even when `encrypt: false`.
A `404` from `/v1/input-key` is treated as "no key advertised" so the SDK also
works against nodes that predate the encryption endpoint.

## API

```js
const client = new Client({ baseUrl, apiKey, timeoutMs, fetch });
```

| Method | Endpoint |
|---|---|
| `models()` | `GET /v1/models` |
| `inputKey()` | `GET /v1/input-key` |
| `faucet(address)` | `POST /faucet` |
| `account(address)` | `GET /account/:address` |
| `infer(options)` | `POST /v1/infer` |
| `task(id)` | `GET /v1/tasks/:id` |
| `verify(id)` | `GET /v1/verify/:id` |
| `usage(address, apiKey?)` | `GET /v1/usage/:address` |
| `usageCsv(address, apiKey?)` | `GET /v1/usage/:address/csv` |

API keys are sent as `Authorization: Bearer <key>`. Errors throw
`VerinooError` with `.status`, `.body` (parsed JSON), and `.bodyText`.

## Tests

```bash
cd sdk/typescript
node --test
```

The suite covers canonical JSON against the repo conformance fixture, a fixed
transaction vector shared with the Python SDK, Ed25519 signing verification,
wallet key-file loading, encryption shape/commitment, a full X25519 +
HKDF + ChaCha20-Poly1305 round-trip, and plaintext/encrypted `infer` selection.
