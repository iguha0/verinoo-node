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

Decoder tasks can pin sampling in the signed transaction; the node, every
validator and the receipt then use the same values:

```js
await client.infer({
  privateKey: wallet,
  input: 'Once upon a time',
  model: 'tiny-gpt-v1',
  sampling: { temperatureMilli: 800, topK: 40, topPMilli: 900, seed: 7 },
});
```

Omitting `sampling` keeps the model's default. Supplying it sets every field:
omitted members are zero, so `sampling: {}` means greedy rather than inheriting
model defaults. Only decoder models accept an override; consensus validation
rejects classifier and out-of-range/unknown sampling (the RPC also rejects it
early), and the receipt only adds `sampling`/`samplingHash` when an override is
present.

Node seeds are `u64` (0 to 18446744073709551615). JavaScript numbers are
IEEE-754 doubles, so values above `2^53 - 1` (9007199254740991) lose precision
before the transaction is signed. Use seeds at or below `2^53 - 1` from this
SDK; the node itself accepts any `u64`.

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
- `sampling` is optional and, when present, becomes part of the canonical
  transaction data and therefore the `txId` / task id.
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

## Batch inference

`inferBatch(requests, options)` signs one independent inference transaction per
request and submits them together to `POST /v1/infer/batch`. Batches must
contain 1 to 32 requests; other sizes are rejected with a `TypeError` before
any network call. The concurrency limit is exported as `MAX_BATCH_SIZE`.

```js
const batch = await client.inferBatch(
  [
    { input: 'first prompt' },
    { input: 'second prompt', model: 'tiny-gpt-v1', maxFee: 200 },
  ],
  { privateKey: wallet, maxFee: 100 },
);
// { accepted, rejected, results: [...] }

for (const result of batch.results) {
  if (result.accepted) console.log(result.index, result.taskId);
  else console.log(result.index, result.error);
}
```

Each `requests` entry is either an input string or an object with `input` plus
optional `model`, `maxFee`, `encrypt`, `deadline`, `nonce`, and `sampling`
overrides. The options object supplies defaults. Signing, input encryption,
auth, and input-key discovery are identical to `infer`. When an item omits
`nonce`, the SDK signs consecutive nonces starting at `account.nonce + 1` (or
the explicit `nonce` option), because batch items execute as separate
transactions from the same account.

The endpoint returns HTTP 200 when at least one item is accepted. Partial
failure is reported per item: `{"index":1,"accepted":false,"error":"...",
"txId":"..."}`. Inspect each `results` entry; the SDK does not throw for
rejected items. Transport, auth, and non-2xx failures still throw
`VerinooError`.

## Task event streams

`streamTask(taskId, handlers, options)` consumes
`GET /v1/tasks/:task_id/events` (`text/event-stream`) and invokes handlers as
events arrive. The stream carries `status` and the final committed `result`
followed by `done` (and optionally `timeout`). This is final-only streaming:
there is no token-level preview, and consensus commits the whole result at
once.

```js
const controller = new AbortController();
const outcome = await client.streamTask(
  taskId,
  {
    signal: controller.signal,
    onStatus: (data) => console.log('status', data.status),
    onResult: (data) => console.log('result', data.resultOutput),
    onDone: (data) => console.log('done', data),
    onTimeout: (data) => console.log('stream timeout', data),
    onEvent: (event, data, id) => console.log('event', event),
  },
  { apiKey: 'tenant-key' }, // optional; defaults to the client key
);
// { taskId, events: [{event, data, id?}], done, timedOut }
```

`onStatus`, `onResult`, `onDone`, and `onTimeout` receive the decoded JSON
payload. `onEvent(event, data, id)` fires first for every event. The parser
accepts CRLF and LF line endings, joins multi-line `data:` fields with
newlines, and ignores comments/heartbeats (`: ...`). `done` terminates the
stream and resolves the returned promise. Pass an `AbortSignal` as `signal` to
cancel; the promise rejects with the abort error (`AbortError`). Non-2xx
responses throw `VerinooError` with `.status`, `.body`, and `.bodyText`. SSE
data is decoded as JSON; non-JSON data is returned as a string.

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
| `inferBatch(requests, options)` | `POST /v1/infer/batch` |
| `streamTask(id, handlers, options)` | `GET /v1/tasks/:id/events` (SSE) |
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
HKDF + ChaCha20-Poly1305 round-trip, plaintext/encrypted `infer` selection,
and offline local-server tests for `inferBatch` and `streamTask` SSE parsing,
heartbeats, CRLF/LF frames, and abort handling.
