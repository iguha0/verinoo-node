import {
  KeyObject,
  createCipheriv,
  createHash,
  createPrivateKey,
  createPublicKey,
  diffieHellman,
  generateKeyPairSync,
  hkdfSync,
  randomBytes,
  sign,
} from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';

export const DEFAULT_BASE_URL = 'https://verinoo.com';
export const DEFAULT_MODEL = 'intent-classifier-v1';
export const DEFAULT_MAX_FEE = 100;
export const INPUT_KEY_INFO = 'verinoo-input-v1';

const ED25519_PKCS8_PREFIX = Buffer.from(
  '302e020100300506032b657004220420',
  'hex',
);
const ED25519_SPKI_PREFIX = Buffer.from('302a300506032b6570032100', 'hex');
const X25519_SPKI_PREFIX = Buffer.from('302a300506032b656e032100', 'hex');

const HEX_RE = /^[0-9a-fA-F]*$/;

export class VerinooError extends Error {
  constructor(message, { status = null, body = null, bodyText = '', cause } = {}) {
    super(message);
    this.name = 'VerinooError';
    this.status = status;
    this.body = body;
    this.bodyText = bodyText;
    if (cause !== undefined) this.cause = cause;
  }
}

export function sha256Hex(value) {
  const bytes = Buffer.isBuffer(value)
    ? value
    : value instanceof Uint8Array
      ? Buffer.from(value)
      : Buffer.from(String(value), 'utf8');
  return createHash('sha256').update(bytes).digest('hex');
}

export function canonicalJson(value) {
  return writeCanonical(value);
}

function writeCanonical(value) {
  if (value === null) return 'null';
  const type = typeof value;
  if (type === 'boolean') return value ? 'true' : 'false';
  if (type === 'string') return JSON.stringify(value);
  if (type === 'number') {
    if (!Number.isSafeInteger(value)) {
      throw new TypeError(
        `float or out-of-range number not allowed in canonical data: ${value}`,
      );
    }
    return String(value);
  }
  if (type === 'bigint') {
    if (value < -(2n ** 63n) || value > 2n ** 64n - 1n) {
      throw new TypeError(
        `number out of range for canonical encoding: ${value}`,
      );
    }
    return value.toString();
  }
  if (Array.isArray(value)) {
    return `[${value.map(writeCanonical).join(',')}]`;
  }
  if (type === 'object') {
    const keys = Object.keys(value).sort((a, b) =>
      Buffer.compare(Buffer.from(a, 'utf8'), Buffer.from(b, 'utf8')),
    );
    const parts = keys.map((key) => {
      const child = value[key];
      if (child === undefined) {
        throw new TypeError(`undefined value for canonical key: ${key}`);
      }
      return `${JSON.stringify(key)}:${writeCanonical(child)}`;
    });
    return `{${parts.join(',')}}`;
  }
  throw new TypeError(`unsupported canonical value: ${type}`);
}

function bytesFromHex(hex, expectedLength, label) {
  if (typeof hex !== 'string' || hex.length % 2 !== 0 || !HEX_RE.test(hex)) {
    throw new TypeError(`${label} must be hexadecimal`);
  }
  const bytes = Buffer.from(hex, 'hex');
  if (expectedLength !== undefined && bytes.length !== expectedLength) {
    throw new TypeError(
      `${label} must be ${expectedLength} bytes, got ${bytes.length}`,
    );
  }
  return bytes;
}

function rawPublicKeyBytes(keyObject, expectedType) {
  const publicKey =
    keyObject.type === 'public' ? keyObject : createPublicKey(keyObject);
  if (expectedType && publicKey.asymmetricKeyType !== expectedType) {
    throw new TypeError(
      `expected ${expectedType} key, got ${publicKey.asymmetricKeyType}`,
    );
  }
  const der = publicKey.export({ type: 'spki', format: 'der' });
  return Buffer.from(der).subarray(-32);
}

export function addressFromPublicKey(publicKeyHex) {
  const bytes = bytesFromHex(publicKeyHex, 32, 'public key');
  const digest = createHash('sha256').update(bytes).digest('hex');
  return `ai_${digest.slice(0, 40)}`;
}

function privateKeyFromSeed(seed) {
  if (!Buffer.isBuffer(seed) && !(seed instanceof Uint8Array)) {
    throw new TypeError('seed must be a Buffer');
  }
  const bytes = Buffer.from(seed);
  if (bytes.length < 32) {
    throw new TypeError('Ed25519 private key must be at least 32 bytes');
  }
  return createPrivateKey({
    key: Buffer.concat([ED25519_PKCS8_PREFIX, bytes.subarray(0, 32)]),
    format: 'der',
    type: 'pkcs8',
  });
}

function seedFromPrivateKey(privateKey) {
  if (!(privateKey instanceof KeyObject) || privateKey.type !== 'private') {
    throw new TypeError('signingKey must be a private KeyObject');
  }
  if (privateKey.asymmetricKeyType !== 'ed25519') {
    throw new TypeError(
      `expected an Ed25519 key, got ${privateKey.asymmetricKeyType}`,
    );
  }
  const der = privateKey.export({ type: 'pkcs8', format: 'der' });
  return Buffer.from(der).subarray(-32);
}

function decodeSeed(value) {
  if (Buffer.isBuffer(value) || value instanceof Uint8Array) {
    return Buffer.from(value).subarray(0, 32);
  }
  if (typeof value !== 'string') {
    throw new TypeError('privateKey must be a base64/hex string or Buffer');
  }
  const text = value.trim();
  if (text.length === 0) throw new TypeError('privateKey is empty');
  if (/^[0-9a-fA-F]{64}$/.test(text)) return Buffer.from(text, 'hex');
  const decoded = Buffer.from(text, 'base64');
  if (decoded.length < 32) {
    throw new TypeError('privateKey must decode to at least 32 bytes');
  }
  return decoded.subarray(0, 32);
}

function readWalletSource(source) {
  if (source instanceof KeyObject) return { privateKey: source };
  if (Buffer.isBuffer(source) || source instanceof Uint8Array) {
    return { privateKey: source };
  }
  if (typeof source === 'string') {
    const text = source.trim();
    if (text.startsWith('{')) return JSON.parse(text);
    if (existsSync(text)) return JSON.parse(readFileSync(text, 'utf8'));
    return { privateKey: text };
  }
  if (source && typeof source === 'object') return source;
  throw new TypeError(
    'wallet must be a key file path, wallet object, KeyObject, or base64 key',
  );
}

export function loadWallet(source) {
  const wallet = readWalletSource(source);
  const signingKey =
    wallet.privateKey instanceof KeyObject
      ? wallet.privateKey
      : privateKeyFromSeed(decodeSeed(wallet.privateKey));
  const seed = seedFromPrivateKey(signingKey);
  const publicKey = rawPublicKeyBytes(signingKey, 'ed25519').toString('hex');
  const address = addressFromPublicKey(publicKey);

  if (wallet.publicKey !== undefined) {
    const expected = String(wallet.publicKey).toLowerCase();
    if (expected !== publicKey) {
      throw new TypeError('wallet publicKey does not match privateKey');
    }
  }
  if (wallet.address !== undefined && wallet.address !== address) {
    throw new TypeError(
      `wallet address ${wallet.address} does not match privateKey address ${address}`,
    );
  }

  return {
    address,
    publicKey,
    privateKey: seed.toString('base64'),
    signingKey,
  };
}

export function generateWallet() {
  const { publicKey, privateKey } = generateKeyPairSync('ed25519');
  const seed = seedFromPrivateKey(privateKey);
  const publicKeyHex = rawPublicKeyBytes(publicKey, 'ed25519').toString('hex');
  return {
    publicKey: publicKeyHex,
    privateKey: seed.toString('base64'),
    address: addressFromPublicKey(publicKeyHex),
  };
}

export function buildSignedTx({
  signingKey,
  from,
  to = '',
  value = 0,
  nonce,
  data,
  gasLimit,
  gasPrice,
}) {
  const payload = { data, from, nonce, to, value };
  if (gasLimit !== undefined) payload.gasLimit = gasLimit;
  if (gasPrice !== undefined) payload.gasPrice = gasPrice;

  const canonical = canonicalJson(payload);
  const txId = sha256Hex(canonical);
  const signature = sign(
    null,
    Buffer.from(txId, 'utf8'),
    signingKey,
  ).toString('hex');

  const envelope = {
    txId,
    from,
    to,
    value,
    nonce,
    data,
    publicKey: rawPublicKeyBytes(signingKey, 'ed25519').toString('hex'),
    signature,
  };
  if (gasLimit !== undefined) envelope.gasLimit = gasLimit;
  if (gasPrice !== undefined) envelope.gasPrice = gasPrice;
  return envelope;
}

export function inputCommitment(plaintext) {
  return sha256Hex(Buffer.from(plaintext, 'utf8'));
}

export function encryptInput(plaintext, networkPublicKeyHex, options = {}) {
  const plaintextBytes = Buffer.from(plaintext, 'utf8');
  const networkRaw = bytesFromHex(
    networkPublicKeyHex,
    32,
    'network public key',
  );
  const networkPublic = createPublicKey({
    key: Buffer.concat([X25519_SPKI_PREFIX, networkRaw]),
    format: 'der',
    type: 'spki',
  });

  const ephemeralPrivate =
    options.ephemeralPrivateKey ?? generateKeyPairSync('x25519').privateKey;
  const ephemeralRaw = rawPublicKeyBytes(ephemeralPrivate, 'x25519');
  const shared = diffieHellman({
    privateKey: ephemeralPrivate,
    publicKey: networkPublic,
  });

  const key = Buffer.from(
    hkdfSync(
      'sha256',
      shared,
      Buffer.concat([networkRaw, ephemeralRaw]),
      Buffer.from(INPUT_KEY_INFO, 'utf8'),
      32,
    ),
  );
  const nonce = options.nonce ? Buffer.from(options.nonce) : randomBytes(12);
  if (nonce.length !== 12) {
    throw new TypeError('ChaCha20-Poly1305 nonce must be 12 bytes');
  }

  const commitment = inputCommitment(plaintext);
  const cipher = createCipheriv('chacha20-poly1305', key, nonce, {
    authTagLength: 16,
  });
  cipher.setAAD(Buffer.from(commitment, 'hex'), {
    plaintextLength: plaintextBytes.length,
  });
  const ciphertext = Buffer.concat([
    cipher.update(plaintextBytes),
    cipher.final(),
    cipher.getAuthTag(),
  ]);

  return {
    inputCommitment: commitment,
    encryptedInput: {
      ephemeralPublicKey: ephemeralRaw.toString('hex'),
      nonce: nonce.toString('hex'),
      ciphertext: ciphertext.toString('base64'),
    },
  };
}

export function buildInferenceData({
  from,
  input,
  model = DEFAULT_MODEL,
  maxFee = DEFAULT_MAX_FEE,
  deadline,
  encryptedInput = null,
  commitment = inputCommitment(input),
}) {
  const inner = {
    requester: from,
    targetModel: model,
    inputCommitment: commitment,
    maxFee,
    deadline,
    verificationType: 'optimistic',
  };
  if (encryptedInput) {
    inner.encryptedInput = encryptedInput;
  } else {
    inner.input = input;
  }
  return { type: 'submitInference', data: inner };
}

export class Client {
  constructor(options = {}) {
    if (typeof options === 'string') options = { baseUrl: options };
    this.baseUrl = String(options.baseUrl ?? DEFAULT_BASE_URL).replace(
      /\/+$/,
      '',
    );
    this.apiKey = options.apiKey;
    this.fetchImpl = options.fetch ?? globalThis.fetch;
    this.timeoutMs = options.timeoutMs ?? 30_000;
    if (typeof this.fetchImpl !== 'function') {
      throw new TypeError('global fetch is required (Node >= 20)');
    }
  }

  async _request(method, path, { body, apiKey, raw = false } = {}) {
    const url = `${this.baseUrl}${path}`;
    const headers = { accept: raw ? '*/*' : 'application/json' };
    if (body !== undefined) headers['content-type'] = 'application/json';
    const key = apiKey ?? this.apiKey;
    if (key) headers.authorization = `Bearer ${key}`;

    const init = {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    };
    if (
      this.timeoutMs > 0 &&
      typeof AbortSignal !== 'undefined' &&
      typeof AbortSignal.timeout === 'function'
    ) {
      init.signal = AbortSignal.timeout(this.timeoutMs);
    }

    let response;
    try {
      response = await this.fetchImpl(url, init);
    } catch (error) {
      throw new VerinooError(`request failed: ${error.message}`, {
        cause: error,
      });
    }

    const text = await response.text();
    if (!response.ok) {
      let parsed = null;
      try {
        parsed = text ? JSON.parse(text) : null;
      } catch {
        parsed = null;
      }
      throw new VerinooError(`HTTP ${response.status}: ${text}`, {
        status: response.status,
        body: parsed,
        bodyText: text,
      });
    }
    if (raw) return text;
    if (!text) return null;
    try {
      return JSON.parse(text);
    } catch {
      return text;
    }
  }

  models() {
    return this._request('GET', '/v1/models');
  }

  inputKey() {
    return this._request('GET', '/v1/input-key');
  }

  faucet(address) {
    return this._request('POST', '/faucet', { body: { address } });
  }

  account(address) {
    return this._request(
      'GET',
      `/account/${encodeURIComponent(address)}`,
    );
  }

  task(id) {
    return this._request('GET', `/v1/tasks/${encodeURIComponent(id)}`);
  }

  /**
   * Poll until the task is executed in a block. A 404 before inclusion is
   * expected: the task id exists only once a block carries the transaction.
   */
  async waitForTask(id, { timeoutMs = 120000, intervalMs = 1000 } = {}) {
    const deadline = Date.now() + timeoutMs;
    let lastError;
    while (Date.now() < deadline) {
      try {
        const view = await this.task(id);
        const status = view?.task?.status;
        if (status === 'completed' || status === 'failed') return view;
      } catch (err) {
        if (!(err instanceof VerinooError && err.status === 404)) {
          lastError = err;
        }
      }
      await new Promise((resolve) => setTimeout(resolve, intervalMs));
    }
    throw (
      lastError ??
      new VerinooError(`task ${id} was not executed within ${timeoutMs}ms`)
    );
  }

  verify(id) {
    return this._request('GET', `/v1/verify/${encodeURIComponent(id)}`);
  }

  usage(address, apiKey) {
    return this._request(
      'GET',
      `/v1/usage/${encodeURIComponent(address)}`,
      { apiKey },
    );
  }

  usageCsv(address, apiKey) {
    return this._request(
      'GET',
      `/v1/usage/${encodeURIComponent(address)}/csv`,
      { apiKey, raw: true },
    );
  }

    async infer({
    privateKey,
    input,
    model = DEFAULT_MODEL,
    maxFee = DEFAULT_MAX_FEE,
    apiKey,
    encrypt = true,
    deadline,
    nonce,
  } = {}) {
    if (typeof input !== 'string') {
      throw new TypeError('input must be a string');
    }
    const wallet = loadWallet(privateKey);
    const account = (await this.account(wallet.address)) ?? {};
    const txNonce = nonce ?? Number(account.nonce ?? 0) + 1;
    const txDeadline =
      deadline ?? Math.floor(Date.now() / 1000) + 3_600;

    let keyInfo = null;
    try {
      keyInfo = await this.inputKey();
    } catch (error) {
      if (!(error instanceof VerinooError) || error.status !== 404) {
        throw error;
      }
    }

    const networkPublicKey = keyInfo?.publicKey;
    let encrypted = null;
    let commitment = inputCommitment(input);
    if (networkPublicKey) {
      if (encrypt || keyInfo.required) {
        const result = encryptInput(input, networkPublicKey);
        encrypted = result.encryptedInput;
        commitment = result.inputCommitment;
      }
    }

    const data = buildInferenceData({
      from: wallet.address,
      input,
      model,
      maxFee,
      deadline: txDeadline,
      encryptedInput: encrypted,
      commitment,
    });
    const envelope = buildSignedTx({
      signingKey: wallet.signingKey,
      from: wallet.address,
      to: '',
      value: 0,
      nonce: txNonce,
      data,
    });
    return this._request('POST', '/v1/infer', { body: envelope, apiKey });
  }
}

export default Client;
