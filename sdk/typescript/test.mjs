import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createDecipheriv,
  createPublicKey,
  diffieHellman,
  generateKeyPairSync,
  hkdfSync,
  verify,
} from 'node:crypto';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import Client, {
  VerinooError,
  addressFromPublicKey,
  buildInferenceData,
  buildSignedTx,
  canonicalJson,
  encryptInput,
  generateWallet,
  loadWallet,
  sha256Hex,
} from './src/index.js';

const ED25519_SPKI_PREFIX = Buffer.from('302a300506032b6570032100', 'hex');
const X25519_SPKI_PREFIX = Buffer.from('302a300506032b656e032100', 'hex');

const TRANSFER = {
  from: 'ai_c0334217fc39036aad7cd9cb15123f3f02d9ff40',
  to: 'ai_db63de2209ed3acb318ae00d5fe44037f5742813',
  value: 250,
  nonce: 1,
  data: { type: 'transfer', data: {} },
};
const TRANSFER_CANONICAL =
  '{"data":{"data":{},"type":"transfer"},"from":"ai_c0334217fc39036aad7cd9cb15123f3f02d9ff40","nonce":1,"to":"ai_db63de2209ed3acb318ae00d5fe44037f5742813","value":250}';
const TRANSFER_TX_ID =
  'c98970ee550228b926be07b5a863f1dfac6daff55c318d771abaa02684217c17';

const SEED = Buffer.from(
  '000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f',
  'hex',
);
const EXPECTED_ADDRESS = 'ai_56475aa75463474c0285df5dbf2bcab73da65135';
const EXPECTED_PUBLIC_KEY =
  '03a107bff3ce10be1d70dd18e74bc09967e4d6309ba50d5f1ddc8664125531b8';
const EXPECTED_CANONICAL =
  '{"data":{"data":{"deadline":1700000000,"input":"hello world","inputCommitment":"b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9","maxFee":100,"requester":"ai_56475aa75463474c0285df5dbf2bcab73da65135","targetModel":"intent-classifier-v1","verificationType":"optimistic"},"type":"submitInference"},"from":"ai_56475aa75463474c0285df5dbf2bcab73da65135","nonce":3,"to":"","value":0}';
const EXPECTED_TX_ID =
  'fd2318e902c051cfffadf7a20c5da21160f53fabcbd4f100d8c5f903fb6563d8';
const EXPECTED_SIGNATURE =
  'ebeb325f38e74831eaeda317484c313ea95913e3a0bd99b9004c5331c1874c615cbc27c751b795d2e415a8d23b496dec35fda8e0b66f5a8d46c3340547a9bc0c';

function inferenceVector() {
  const wallet = loadWallet({ privateKey: SEED.toString('base64') });
  const data = buildInferenceData({
    from: wallet.address,
    input: 'hello world',
    model: 'intent-classifier-v1',
    maxFee: 100,
    deadline: 1700000000,
  });
  return { wallet, data };
}

function ed25519PublicKey(hex) {
  return createPublicKey({
    key: Buffer.concat([ED25519_SPKI_PREFIX, Buffer.from(hex, 'hex')]),
    format: 'der',
    type: 'spki',
  });
}

test('canonical JSON matches Rust canonical encoding', () => {
  const reordered = {
    nonce: 1,
    value: 250,
    to: TRANSFER.to,
    from: TRANSFER.from,
    data: { data: {}, type: 'transfer' },
  };
  assert.equal(canonicalJson(reordered), TRANSFER_CANONICAL);
  assert.equal(sha256Hex(canonicalJson(reordered)), TRANSFER_TX_ID);

  assert.equal(
    canonicalJson({ z: [{ b: 2, a: 1 }, 'x'], a: { d: 4, c: 3 } }),
    '{"a":{"c":3,"d":4},"z":[{"a":1,"b":2},"x"]}',
  );
  assert.equal(canonicalJson({ s: 'hello world' }), '{"s":"hello world"}');
  assert.throws(() => canonicalJson({ n: 1.5 }), /float|out-of-range/);
  assert.throws(
    () => canonicalJson({ n: Number.MAX_SAFE_INTEGER + 1 }),
    /float|out-of-range/,
  );
});

test('tx id and signature vector are stable', () => {
  const { wallet, data } = inferenceVector();
  assert.equal(wallet.address, EXPECTED_ADDRESS);
  assert.equal(wallet.publicKey, EXPECTED_PUBLIC_KEY);

  const envelope = buildSignedTx({
    signingKey: wallet.signingKey,
    from: wallet.address,
    to: '',
    value: 0,
    nonce: 3,
    data,
  });

  assert.equal(envelope.txId, EXPECTED_TX_ID);
  assert.equal(envelope.signature, EXPECTED_SIGNATURE);
  assert.equal(
    canonicalJson({
      data,
      from: wallet.address,
      nonce: 3,
      to: '',
      value: 0,
    }),
    EXPECTED_CANONICAL,
  );
  assert.equal(envelope.publicKey, EXPECTED_PUBLIC_KEY);
  assert.equal(envelope.gasLimit, undefined);
  assert.equal(envelope.gasPrice, undefined);

  assert.ok(
    verify(
      null,
      Buffer.from(envelope.txId, 'utf8'),
      ed25519PublicKey(envelope.publicKey),
      Buffer.from(envelope.signature, 'hex'),
    ),
  );
  assert.equal(
    verify(
      null,
      Buffer.from('00'.repeat(32), 'utf8'),
      ed25519PublicKey(envelope.publicKey),
      Buffer.from(envelope.signature, 'hex'),
    ),
    false,
  );
});

test('wallet generation and Rust-CLI key-file loading', () => {
  const generated = generateWallet();
  assert.match(generated.address, /^ai_[0-9a-f]{40}$/);
  assert.match(generated.publicKey, /^[0-9a-f]{64}$/);
  assert.equal(Buffer.from(generated.privateKey, 'base64').length, 32);
  assert.equal(
    addressFromPublicKey(generated.publicKey),
    generated.address,
  );

  const dir = mkdtempSync(join(tmpdir(), 'verinoo-ts-'));
  const path = join(dir, 'key.json');
  writeFileSync(path, JSON.stringify(generated, null, 2));
  const loaded = loadWallet(path);
  assert.equal(loaded.address, generated.address);
  assert.equal(loaded.publicKey, generated.publicKey);

  assert.throws(
    () =>
      loadWallet({
        ...generated,
        address: 'ai_0000000000000000000000000000000000000000',
      }),
    /does not match/,
  );
});

test('input encryption shape, commitment and round-trip', () => {
  const network = generateKeyPairSync('x25519');
  const networkPublic = Buffer.from(
    network.publicKey.export({ type: 'spki', format: 'der' }),
  ).subarray(-32);

  const { inputCommitment, encryptedInput } = encryptInput(
    'hello world',
    networkPublic.toString('hex'),
  );

  assert.deepEqual(Object.keys(encryptedInput).sort(), [
    'ciphertext',
    'ephemeralPublicKey',
    'nonce',
  ]);
  assert.match(encryptedInput.ephemeralPublicKey, /^[0-9a-f]{64}$/);
  assert.match(encryptedInput.nonce, /^[0-9a-f]{24}$/);
  assert.equal(inputCommitment, sha256Hex('hello world'));
  assert.equal(
    Buffer.from(encryptedInput.ciphertext, 'base64').length,
    Buffer.byteLength('hello world') + 16,
  );

  const ephemeralPublic = createPublicKey({
    key: Buffer.concat([
      X25519_SPKI_PREFIX,
      Buffer.from(encryptedInput.ephemeralPublicKey, 'hex'),
    ]),
    format: 'der',
    type: 'spki',
  });
  const shared = diffieHellman({
    privateKey: network.privateKey,
    publicKey: ephemeralPublic,
  });
  const key = Buffer.from(
    hkdfSync(
      'sha256',
      shared,
      Buffer.concat([
        networkPublic,
        Buffer.from(encryptedInput.ephemeralPublicKey, 'hex'),
      ]),
      Buffer.from('verinoo-input-v1', 'utf8'),
      32,
    ),
  );
  const ciphertext = Buffer.from(encryptedInput.ciphertext, 'base64');
  const decipher = createDecipheriv(
    'chacha20-poly1305',
    key,
    Buffer.from(encryptedInput.nonce, 'hex'),
    { authTagLength: 16 },
  );
  decipher.setAAD(Buffer.from(inputCommitment, 'hex'), {
    plaintextLength: ciphertext.length - 16,
  });
  decipher.setAuthTag(ciphertext.subarray(-16));
  const plaintext = Buffer.concat([
    decipher.update(ciphertext.subarray(0, -16)),
    decipher.final(),
  ]);
  assert.equal(plaintext.toString('utf8'), 'hello world');

  const bad = createDecipheriv(
    'chacha20-poly1305',
    key,
    Buffer.from(encryptedInput.nonce, 'hex'),
    { authTagLength: 16 },
  );
  bad.setAAD(Buffer.alloc(32), { plaintextLength: ciphertext.length - 16 });
  bad.setAuthTag(ciphertext.subarray(-16));
  assert.throws(() => {
    bad.update(ciphertext.subarray(0, -16));
    bad.final();
  });
});

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

function fakeNode({ inputKey, account = { balance: 1_000_000, nonce: 4 } }) {
  const calls = [];
  const fetch = async (url, init = {}) => {
    const path = new URL(url).pathname;
    calls.push({ path, init });
    if (path.startsWith('/account/')) return jsonResponse(account);
    if (path === '/v1/input-key') {
      if (inputKey === null) {
        return jsonResponse({ error: 'not found' }, 404);
      }
      return jsonResponse(inputKey);
    }
    if (path === '/v1/infer') {
      const body = JSON.parse(init.body);
      return jsonResponse({
        accepted: true,
        txId: body.txId,
        taskId: 'task-1',
      });
    }
    return jsonResponse({ error: `unexpected ${path}` }, 500);
  };
  return { calls, fetch };
}

test('infer sends legacy input when the node has no input key', async () => {
  const { calls, fetch } = fakeNode({ inputKey: null });
  const client = new Client({
    baseUrl: 'https://node.test/',
    fetch,
    apiKey: 'tenant-key',
  });
  const wallet = loadWallet({ privateKey: SEED.toString('base64') });

  const result = await client.infer({ privateKey: wallet, input: 'hi' });
  assert.equal(result.taskId, 'task-1');

  const post = calls.find((call) => call.path === '/v1/infer');
  const envelope = JSON.parse(post.init.body);
  assert.equal(envelope.nonce, 5);
  assert.equal(envelope.data.type, 'submitInference');
  assert.equal(envelope.data.data.input, 'hi');
  assert.equal(envelope.data.data.encryptedInput, undefined);
  assert.equal(envelope.data.data.deadline >= Math.floor(Date.now() / 1000), true);
  assert.equal(post.init.headers.authorization, 'Bearer tenant-key');
});

test('infer encrypts when a network key is advertised', async () => {
  const network = generateKeyPairSync('x25519');
  const networkPublic = Buffer.from(
    network.publicKey.export({ type: 'spki', format: 'der' }),
  ).subarray(-32);
  const { calls, fetch } = fakeNode({
    inputKey: {
      publicKey: networkPublic.toString('hex'),
      required: false,
      algorithm: 'x25519-hkdf-sha256-chacha20poly1305',
    },
  });
  const client = new Client({ baseUrl: 'https://node.test', fetch });
  const wallet = loadWallet({ privateKey: SEED.toString('base64') });

  await client.infer({ privateKey: wallet, input: 'secret text' });
  const post = calls.find((call) => call.path === '/v1/infer');
  const inner = JSON.parse(post.init.body).data.data;
  assert.equal(inner.input, undefined);
  assert.deepEqual(Object.keys(inner.encryptedInput).sort(), [
    'ciphertext',
    'ephemeralPublicKey',
    'nonce',
  ]);
  assert.equal(inner.inputCommitment, sha256Hex('secret text'));
});

test('infer honors required encryption when plaintext is requested', async () => {
  const network = generateKeyPairSync('x25519');
  const networkPublic = Buffer.from(
    network.publicKey.export({ type: 'spki', format: 'der' }),
  ).subarray(-32);
  const { calls, fetch } = fakeNode({
    inputKey: {
      publicKey: networkPublic.toString('hex'),
      required: true,
      algorithm: 'x25519-hkdf-sha256-chacha20poly1305',
    },
  });
  const client = new Client({ baseUrl: 'https://node.test', fetch });
  const wallet = loadWallet({ privateKey: SEED.toString('base64') });

  await client.infer({
    privateKey: wallet,
    input: 'must encrypt',
    encrypt: false,
  });
  const post = calls.find((call) => call.path === '/v1/infer');
  const inner = JSON.parse(post.init.body).data.data;
  assert.equal(inner.input, undefined);
  assert.ok(inner.encryptedInput);
});

test('errors carry HTTP status and JSON body', async () => {
  const fetch = async () => jsonResponse({ error: 'nope' }, 400);
  const client = new Client({ baseUrl: 'https://node.test', fetch });
  await assert.rejects(
    () => client.models(),
    (error) => {
      assert.ok(error instanceof VerinooError);
      assert.equal(error.status, 400);
      assert.deepEqual(error.body, { error: 'nope' });
      assert.match(error.message, /HTTP 400: .*nope/);
      return true;
    },
  );
});
