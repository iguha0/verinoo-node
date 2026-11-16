import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';

import Client, {
  DEFAULT_STREAM_TIMEOUT_MS,
  MAX_BATCH_SIZE,
  VerinooError,
  loadWallet,
} from './src/index.js';

const SEED = Buffer.from(
  '000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f',
  'hex',
);

function listen(handler) {
  return new Promise((resolve, reject) => {
    const server = createServer(handler);
    server.on('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      resolve({ server, baseUrl: `http://127.0.0.1:${port}` });
    });
  });
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', (chunk) => chunks.push(chunk));
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

function sendJson(res, body, status = 200) {
  const text = JSON.stringify(body);
  res.writeHead(status, {
    'content-type': 'application/json',
    'content-length': Buffer.byteLength(text),
    connection: 'close',
  });
  res.end(text);
}

async function closeServer(server) {
  await new Promise((resolve) => server.close(resolve));
}

test('inferBatch signs one transaction per request and returns results', async () => {
  const calls = [];
  let batchBody;
  const { server, baseUrl } = await listen(async (req, res) => {
    const url = new URL(req.url, 'http://localhost');
    calls.push({
      method: req.method,
      path: url.pathname,
      headers: req.headers,
    });
    if (url.pathname.startsWith('/account/')) {
      sendJson(res, { balance: 1_000_000, nonce: 4 });
      return;
    }
    if (url.pathname === '/v1/input-key') {
      sendJson(res, { error: 'not found' }, 404);
      return;
    }
    if (url.pathname === '/v1/infer/batch') {
      batchBody = JSON.parse(await readBody(req));
      sendJson(res, {
        accepted: 2,
        rejected: 1,
        results: [
          {
            index: 0,
            accepted: true,
            txId: batchBody.transactions[0].txId,
            taskId: 'task-0',
          },
          {
            index: 1,
            accepted: true,
            txId: batchBody.transactions[1].txId,
            taskId: 'task-1',
          },
          {
            index: 2,
            accepted: false,
            error: 'model not allowed',
            txId: 'bad',
          },
        ],
      });
      return;
    }
    sendJson(res, { error: `unexpected ${url.pathname}` }, 500);
  });

  try {
    const client = new Client({ baseUrl, apiKey: 'tenant-key' });
    const wallet = loadWallet({ privateKey: SEED.toString('base64') });
    const response = await client.inferBatch(
      [
        { input: 'first' },
        { input: 'second', model: 'tiny-gpt-v1', maxFee: 7 },
        'third',
      ],
      { privateKey: wallet, maxFee: 11 },
    );

    assert.equal(response.accepted, 2);
    assert.equal(response.rejected, 1);
    assert.deepEqual(
      response.results.map((item) => item.index),
      [0, 1, 2],
    );
    assert.equal(response.results[0].taskId, 'task-0');
    assert.equal(response.results[2].accepted, false);
    assert.equal(response.results[2].error, 'model not allowed');

    const post = calls.find((call) => call.path === '/v1/infer/batch');
    assert.equal(post.method, 'POST');
    assert.equal(post.headers.authorization, 'Bearer tenant-key');
    assert.equal(batchBody.transactions.length, 3);

    for (const tx of batchBody.transactions) {
      assert.match(tx.txId, /^[0-9a-f]{64}$/);
      assert.match(tx.signature, /^[0-9a-f]{128}$/);
      assert.match(tx.publicKey, /^[0-9a-f]{64}$/);
      assert.equal(tx.data.type, 'submitInference');
      assert.equal(tx.to, '');
      assert.equal(tx.value, 0);
    }

    const [first, second, third] = batchBody.transactions;
    assert.deepEqual(
      batchBody.transactions.map((tx) => tx.nonce),
      [5, 6, 7],
    );
    assert.equal(first.data.data.input, 'first');
    assert.equal(first.data.data.maxFee, 11);
    assert.equal(second.data.data.targetModel, 'tiny-gpt-v1');
    assert.equal(second.data.data.maxFee, 7);
    assert.equal(third.data.data.input, 'third');
    assert.equal(new Set(batchBody.transactions.map((tx) => tx.txId)).size, 3);
  } finally {
    await closeServer(server);
  }
});

test('inferBatch continues implicit nonces from the last explicit nonce', async () => {
  let batchBody;
  const { server, baseUrl } = await listen(async (req, res) => {
    const url = new URL(req.url, 'http://localhost');
    if (url.pathname.startsWith('/account/')) {
      sendJson(res, { balance: 1_000_000, nonce: 4 });
      return;
    }
    if (url.pathname === '/v1/input-key') {
      sendJson(res, { error: 'not found' }, 404);
      return;
    }
    if (url.pathname === '/v1/infer/batch') {
      batchBody = JSON.parse(await readBody(req));
      sendJson(res, {
        accepted: batchBody.transactions.length,
        rejected: 0,
        results: batchBody.transactions.map((tx, index) => ({
          index,
          accepted: true,
          txId: tx.txId,
          taskId: `task-${index}`,
        })),
      });
      return;
    }
    sendJson(res, { error: `unexpected ${url.pathname}` }, 500);
  });

  try {
    const client = new Client({ baseUrl });
    const wallet = loadWallet({ privateKey: SEED.toString('base64') });
    await client.inferBatch(
      [
        'implicit-first',
        { input: 'explicit-mid', nonce: 20 },
        'implicit-after',
        { input: 'explicit-later', nonce: 30 },
        'implicit-last',
      ],
      { privateKey: wallet },
    );

    assert.deepEqual(
      batchBody.transactions.map((tx) => tx.nonce),
      [5, 20, 21, 30, 31],
    );
  } finally {
    await closeServer(server);
  }
});

test('inferBatch resolves with per-item errors when all items are rejected', async () => {
  const { server, baseUrl } = await listen(async (req, res) => {
    const url = new URL(req.url, 'http://localhost');
    if (url.pathname.startsWith('/account/')) {
      sendJson(res, { balance: 1_000_000, nonce: 4 });
      return;
    }
    if (url.pathname === '/v1/input-key') {
      sendJson(res, { error: 'not found' }, 404);
      return;
    }
    if (url.pathname === '/v1/infer/batch') {
      const body = JSON.parse(await readBody(req));
      sendJson(res, {
        accepted: 0,
        rejected: body.transactions.length,
        results: body.transactions.map((tx, index) => ({
          index,
          accepted: false,
          error: 'policy rejected',
          txId: tx.txId,
        })),
      });
      return;
    }
    sendJson(res, { error: `unexpected ${url.pathname}` }, 500);
  });

  try {
    const client = new Client({ baseUrl });
    const wallet = loadWallet({ privateKey: SEED.toString('base64') });
    const response = await client.inferBatch(['first', 'second'], {
      privateKey: wallet,
    });

    assert.equal(response.accepted, 0);
    assert.equal(response.rejected, 2);
    assert.deepEqual(
      response.results.map((result) => result.accepted),
      [false, false],
    );
    assert.deepEqual(
      response.results.map((result) => result.error),
      ['policy rejected', 'policy rejected'],
    );
  } finally {
    await closeServer(server);
  }
});

test('inferBatch rejects batches outside 1..32 before any request', async () => {
  const client = new Client({
    baseUrl: 'http://127.0.0.1:1',
    fetch: async () => {
      throw new Error('network should not be used');
    },
  });
  const wallet = loadWallet({ privateKey: SEED.toString('base64') });

  await assert.rejects(
    () => client.inferBatch([], { privateKey: wallet }),
    /between 1 and 32 requests/,
  );
  const tooMany = Array.from(
    { length: MAX_BATCH_SIZE + 1 },
    (_, index) => `item-${index}`,
  );
  await assert.rejects(
    () => client.inferBatch(tooMany, { privateKey: wallet }),
    /between 1 and 32 requests/,
  );
});

test('streamTask parses CRLF/LF SSE frames and heartbeats in order', async () => {
  const seenHeaders = [];
  const { server, baseUrl } = await listen((req, res) => {
    const url = new URL(req.url, 'http://localhost');
    seenHeaders.push(req.headers);
    if (url.pathname !== '/v1/tasks/task-1/events') {
      sendJson(res, { error: 'not found' }, 404);
      return;
    }
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
    });
    res.write(': heartbeat\n\n');
    res.write('event: status\r\n');
    res.write('data: {"taskId":"task-1","status":"pending"}\r\n\r\n');
    res.write(': keep-alive\r\n\r\n');
    res.write('event: result\r\n');
    res.write('data: {"taskId":"task-1",\r\n');
    res.write('data: "task":{"taskId":"task-1","status":"completed",\r\n');
    res.write('data: "resultOutput":"hello"},\r\n');
    res.write('data: "receipt":{"taskId":"task-1","outputHash":"aa"}}\r\n\r\n');
    res.write('event: done\r\n');
    res.write('data: {"taskId":"task-1"}\r\n\r\n');
    res.end();
  });

  try {
    const client = new Client({ baseUrl, apiKey: 'sse-key' });
    const order = [];
    const seenEvents = [];
    const result = await client.streamTask('task-1', {
      onEvent(event, data) {
        seenEvents.push({ event, data });
      },
      onStatus(data) {
        order.push(`status:${data.status}`);
      },
      onResult(data) {
        order.push(`result:${data.task.resultOutput}`);
      },
      onDone(data) {
        order.push(`done:${data.taskId}`);
      },
    });

    assert.deepEqual(order, ['status:pending', 'result:hello', 'done:task-1']);
    assert.deepEqual(
      seenEvents.map((entry) => entry.event),
      ['status', 'result', 'done'],
    );
    assert.deepEqual(seenEvents[1].data, {
      taskId: 'task-1',
      task: {
        taskId: 'task-1',
        status: 'completed',
        resultOutput: 'hello',
      },
      receipt: { taskId: 'task-1', outputHash: 'aa' },
    });
    assert.equal(seenEvents[1].data.resultOutput, undefined);
    assert.equal(result.taskId, 'task-1');
    assert.equal(result.done, true);
    assert.equal(result.timedOut, false);
    assert.deepEqual(
      result.events.map((entry) => entry.event),
      ['status', 'result', 'done'],
    );
    assert.equal(result.events[2].data.taskId, 'task-1');
    assert.equal(seenHeaders[0].authorization, 'Bearer sse-key');
    assert.equal(seenHeaders[0].accept, 'text/event-stream');
  } finally {
    await closeServer(server);
  }
});

test('streamTask aborts an open SSE stream via AbortSignal', async () => {
  let heartbeat;
  const { server, baseUrl } = await listen((req, res) => {
    const url = new URL(req.url, 'http://localhost');
    if (url.pathname !== '/v1/tasks/task-abort/events') {
      sendJson(res, { error: 'not found' }, 404);
      return;
    }
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      connection: 'keep-alive',
    });
    res.write('event: status\r\n');
    res.write('data: {"taskId":"task-abort","status":"pending"}\r\n\r\n');
    heartbeat = setInterval(() => {
      try {
        res.write(': ping\r\n\r\n');
      } catch {
        // The client closed the stream.
      }
    }, 25);
    res.on('close', () => clearInterval(heartbeat));
  });

  try {
    const client = new Client({ baseUrl });
    const controller = new AbortController();
    let statuses = 0;
    const promise = client.streamTask('task-abort', {
      signal: controller.signal,
      onStatus() {
        statuses += 1;
        controller.abort();
      },
    });

    await assert.rejects(promise, (error) => {
      assert.equal(error.name, 'AbortError');
      return true;
    });
    assert.equal(statuses, 1);
  } finally {
    if (heartbeat) clearInterval(heartbeat);
    await closeServer(server);
  }
});

test('streamTask enforces an overall timeout on an idle SSE stream', async () => {
  let heartbeat;
  const { server, baseUrl } = await listen((req, res) => {
    const url = new URL(req.url, 'http://localhost');
    if (url.pathname !== '/v1/tasks/task-timeout/events') {
      sendJson(res, { error: 'not found' }, 404);
      return;
    }
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      connection: 'keep-alive',
    });
    res.write('event: status\r\n');
    res.write('data: {"taskId":"task-timeout","status":"pending"}\r\n\r\n');
    heartbeat = setInterval(() => {
      try {
        res.write(': ping\r\n\r\n');
      } catch {
        // The client closed the stream.
      }
    }, 25);
    res.on('close', () => clearInterval(heartbeat));
  });

  try {
    assert.ok(DEFAULT_STREAM_TIMEOUT_MS > 60_000);
    const client = new Client({ baseUrl });
    const started = Date.now();
    await assert.rejects(
      () => client.streamTask('task-timeout', {}, { timeoutMs: 75 }),
      (error) => {
        assert.ok(error instanceof VerinooError);
        assert.equal(error.timedOut, true);
        assert.match(error.message, /timed out after 75ms/);
        return true;
      },
    );
    assert.ok(Date.now() - started < 2_000);
  } finally {
    if (heartbeat) clearInterval(heartbeat);
    await closeServer(server);
  }
});
