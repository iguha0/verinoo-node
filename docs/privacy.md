# Input privacy — threat model and operation

Inference inputs must be readable by every validator because execution happens
during block application. They must not be readable by API clients, block
explorers, or anyone who has a task id. This document describes exactly what the
current design protects and what it does not.

## What is protected

- The client encrypts the input to a network X25519 public key before signing.
- The transaction, block and task state contain only ciphertext, an ephemeral
  public key, a nonce and `inputCommitment = sha256(plaintext)`.
- `GET /v1/tasks/:id` redacts both `input` and `encryptedInput`; it returns
  `inputHash`, `encrypted: true`, the result and the receipt.
- `GET /v1/verify/:id` decrypts server-side and returns only the recomputed
  output and hashes.
- A node without the network key cannot execute encrypted tasks. It leaves them
  pending and logs the reason; it does not guess or commit a result.

## What is not protected

- **Not operator-blind.** Every validator, including the operator, holds the
  network private key and can decrypt. This is encryption to the validator set,
  not TEE- or ZK-level confidentiality.
- **Metadata is public.** Requester address, model id, fee, deadline, task id,
  block height, output and receipt are public. The output itself can reveal
  information about the input.
- **Pre-upgrade data.** Tasks and blocks created before encryption was enabled
  contain plaintext and stay public forever. Start a fresh chain for a pilot if
  old data matters.
- **No forward secrecy.** Rotating the network key makes old ciphertext
  undecryptable, but a stolen key decrypts every task encrypted to it.
- **No access control on ciphertext.** Anyone can download the ciphertext; the
  security rests on the key.

## Scheme

1. Client fetches `GET /v1/input-key`:
   `{"publicKey":"<hex>","required":bool,"algorithm":"x25519-hkdf-sha256-chacha20poly1305"}`.
2. Client generates an ephemeral X25519 keypair.
3. `shared = X25519(ephemeral_private, network_public)`.
4. `key = HKDF-SHA256(shared, salt = network_public || ephemeral_public,
   info = "verinoo-input-v1", 32)`.
5. `nonce = 12 random bytes`;
   `ciphertext = ChaCha20-Poly1305(key, nonce, plaintext, aad = inputCommitment bytes)`.
6. The transaction carries
   `encryptedInput = {ephemeralPublicKey, nonce, ciphertext}` and
   `inputCommitment`; it does not carry `input`.

Validators decrypt with the network secret, verify `sha256(plaintext) ==
inputCommitment`, and only then execute. A wrong key or tampered ciphertext is
a no-op: the task stays pending and a warning is logged.

## Operating the key

```bash
# 32 random bytes, hex encoded; the SAME value on every validator
openssl rand -hex 32
# /etc/verinoo/node.env
VNN_INPUT_KEY=<64 hex chars>
# reject any plaintext inference submission (recommended for paid deployments)
VNN_REQUIRE_ENCRYPTED_INPUT=true
```

- Back up the key with the database. Without it, encrypted tasks cannot be
  re-executed or verified.
- Rotate only after draining pending encrypted tasks, then restart every
  validator with the new key. There is no online rotation or re-encryption.
- A validator set with different keys will fail to execute the same encrypted
  task and diverge. Treat the key as part of the network configuration, like
  the model set.
- Leave `VNN_INPUT_KEY` unset only for a public demo with non-sensitive text.

## What this unlocks commercially

It removes the "inputs are public on-chain" blocker for regulated workflows
(support tickets, claims, KYC documents) and allows a pilot to require
encryption. It does **not** make the operator blind: a regulated buyer who
cannot trust the operator still needs a TEE or a ZK proof, which this product
does not have.
