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
auth, and input-key discovery are identical to `infer`.

Nonces are assigned in request order from `account.nonce + 1` (or the explicit
`nonce` option). An item with an explicit `nonce` is used as-is, and the next
implicit item continues from it (`prev_nonce + 1`); implicit items never
restart from the batch base after an explicit item. For example, with account
nonce `4` and items `[implicit, {nonce: 20}, implicit]` the signed nonces are
`[5, 20, 21]`. Pass explicit nonces for every item if you need a different
sequence; the SDK does not reorder or fill gaps.

The endpoint returns HTTP 200 after per-item processing even when every item is
rejected. Only request-level failures (malformed body, empty or oversized
batch, auth) return a non-2xx status. Partial failure is reported per item:
`{"index":1,"accepted":false,"error":"...","txId":"..."}`. Inspect each
`results` entry; the SDK does not throw for rejected items. Transport, auth,
and other non-2xx failures still throw `VerinooError`.

A rejected middle item does not stop later items from being submitted, but it
does leave later nonces from the same sender invalid: the node accepted no
nonce for that item, so every subsequent same-sender item in the batch is out
of sequence and will be rejected too unless it was signed with the correct
continuation nonce. The SDK does not auto-repair or re-sign sequences after a
partial failure; resubmit the affected items with fresh nonces from the
account's current state.

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
    onResult: (data) => console.log('result', data.task.resultOutput),
    onDone: (data) => console.log('done', data),
    onTimeout: (data) => console.log('stream timeout', data),
    onEvent: (event, data, id) => console.log('event', event),
  },
  { apiKey: 'tenant-key', timeoutMs: 65_000 },
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

The `result` event payload is `{taskId, task, receipt}`. The output is nested
on the redacted task (`data.task.resultOutput`), not a flat `data.resultOutput`.
The task view has `input`/`encryptedInput` removed and replaced by `inputHash`
and `encrypted`; `receipt` is `null` until a successful task has a receipt.
`done` carries `{taskId}`.

`streamTask` also enforces an overall client-side deadline. `timeoutMs`
defaults to `65_000` (exported as `DEFAULT_STREAM_TIMEOUT_MS`), just above the
node's 60 s SSE cap. When the deadline passes, the request is aborted and the
promise rejects with a `VerinooError` whose message is
`task <id> stream timed out after <timeoutMs>ms` and whose `.timedOut` is
`true`. Set `timeoutMs: 0` to disable the client deadline. A caller-supplied
`signal` still takes precedence and rejects with the abort error.

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
and offline local-server tests for `inferBatch` nonce sequencing (including
mixed explicit/implicit nonces and all-rejected batches), the real SSE result
shape, heartbeats, CRLF/LF frames, stream abort handling, and the overall
stream timeout.
