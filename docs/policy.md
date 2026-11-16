# Request policy

`verinoo-policy` is a small, deterministic policy engine for plaintext
inference inputs and optional output views. It performs no network calls, loads
no models, and uses no randomness or clock, so the same policy and input always
produce the same result.

It is **not ML moderation** and **not a PII guarantee**. It only enforces the
terms, regular expressions and identifier redactions that an operator
configures. It cannot understand intent or discover every sensitive value.

## API

```rust
use verinoo_policy::{InputKind, Policy, PolicyError};

let policy = Policy::default();
let checked = policy.check_input("gpt-2", "hello")?;
println!("{}", checked.text);

let policy = Policy::from_json(r#"{
    "maxInputBytes": 4096,
    "allowedModels": ["gpt-2"],
    "blockedTerms": ["confidential"],
    "blockedPatterns": ["(?i)account\\s+number"],
    "redact": { "email": true, "phone": true, "ssn": true },
    "requirePlaintext": true
}"#)?;

let checked = policy.check_input("gpt-2", "mail alice@example.com")?;
assert_eq!(checked.text, "mail [EMAIL]");
```

Public surface:

- `Policy::default()` — permissive defaults.
- `Policy::from_json(&str) -> Result<Policy, PolicyError>` — parse and validate.
- `Policy::from_config(PolicyConfig) -> Result<Policy, PolicyError>` — validate
  an in-memory config.
- `Policy::check_input(model, input) -> Result<SanitizedInput, PolicyError>` —
  checks a plaintext input.
- `Policy::check_input_kind(model, input, InputKind) -> Result<SanitizedInput, PolicyError>` —
  the same checks with an explicit caller label. `check_input` is
  `check_input_kind(..., InputKind::Plaintext)`.
- `Policy::check_output(model, output) -> Result<SanitizedOutput, PolicyError>` —
  a **non-consensus view helper** for displaying an output.
- `Policy::config() -> &PolicyConfig` — the validated, normalized config.
- `SanitizedInput` / `SanitizedOutput` contain `text` and the list of applied
  `RedactionKind` values (`Email`, `Phone`, `Ssn`).

Stable error variants: `TooLong`, `ModelNotAllowed`, `BlockedTerm`,
`BlockedPattern`, `EncryptedInput`, and `InvalidConfig`. Errors never contain
the raw blocked text.

## Config

All fields are optional; `{}` is the permissive default.

```json
{
  "maxInputBytes": 8192,
  "allowedModels": [],
  "blockedTerms": [],
  "blockedPatterns": [],
  "redact": { "email": false, "phone": false, "ssn": false },
  "requirePlaintext": false
}
```

- `maxInputBytes` — byte length limit, checked before any regex work. Must be
  between 1 and 16 MiB. Default `8192`.
- `allowedModels` — exact model ids; empty means every model is allowed.
- `blockedTerms` — case-insensitive substring matches. Empty terms are
  rejected, and matches are never echoed.
- `blockedPatterns` — Rust `regex` patterns. Invalid patterns, empty patterns,
  and patterns that match the empty string (for example `.*`) are rejected at
  load time. Config lists and entries are size-capped.
- `redact` — replaces matches with the stable placeholders `[EMAIL]`,
  `[PHONE]`, and `[SSN]`. Redaction is idempotent: re-checking sanitized text
  does not match the placeholders again.
- `requirePlaintext` — when true, payloads marked `InputKind::Encrypted` or
  `InputKind::Opaque` are rejected with `EncryptedInput`.

Unknown config fields are rejected so a typo cannot silently disable a rule.

## Plaintext-only limitation

The policy can only inspect plaintext. Encrypted payloads arrived from
`verinoo-crypto` as ciphertext and opaque payloads are caller-defined blobs;
searching or rewriting either would corrupt them. With `requirePlaintext`
false they pass through untouched; with it true they are rejected. A node-side
policy sees only the plaintext it is given and cannot inspect ciphertext.
Encryption does not make the operator blind: validators hold the network key
(see `docs/privacy.md`).

## Output view semantics

`check_output` is for display only. Its result must never replace the inference
output, result hash, receipt, task state, or any other committed data. Output
policy is a local view: applying it on one node and not another must not change
consensus. It also does not enforce the input byte limit, because
`maxInputBytes` describes requests, not generated responses.

## Integration plan

1. **API boundary before transaction creation.** After decryption or plaintext
   receipt, call `check_input_kind` before building the inference transaction.
   Reject the request with the stable error reason; do not include blocked text
   in API responses, logs, or metrics.
2. **SDK pre-encryption.** Clients run `check_input` on plaintext before
   encrypting with `verinoo-crypto`, so policy failures happen before signing
   or paying fees. SDKs still treat server-side policy as authoritative.
3. **Never change the committed result.** Call `check_output` only when
   rendering a response or explorer view. The transaction, execution result,
   result hash, receipt, and verification path keep the raw output. Peers must
   not reject or re-execute a result because a local output view differs.
4. **No consensus dependency.** Keep policy config node-local. Changing config
   must not alter block validity, state transitions, or receipt commitments.
