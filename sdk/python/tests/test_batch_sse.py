import base64
import contextlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from verinoo import MAX_BATCH_SIZE, Client, load_wallet

SEED_B64 = base64.b64encode(bytes(range(32))).decode("ascii")


def _send_json(handler, status, body):
    payload = json.dumps(body).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


@contextlib.contextmanager
def _serve(handler_class):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_class)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_infer_batch_signs_transactions_and_returns_results():
    calls = []
    batch = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            calls.append(
                {
                    "method": "GET",
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                }
            )
            if self.path.startswith("/account/"):
                _send_json(self, 200, {"balance": 1_000_000, "nonce": 4})
            elif self.path == "/v1/input-key":
                _send_json(self, 404, {"error": "not found"})
            else:
                _send_json(self, 500, {"error": self.path})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            calls.append(
                {
                    "method": "POST",
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                    "body": body,
                }
            )
            if self.path != "/v1/infer/batch":
                _send_json(self, 500, {"error": self.path})
                return
            batch["body"] = body
            first, second, _third = body["transactions"]
            _send_json(
                self,
                200,
                {
                    "accepted": 2,
                    "rejected": 1,
                    "results": [
                        {
                            "index": 0,
                            "accepted": True,
                            "txId": first["txId"],
                            "taskId": "task-0",
                        },
                        {
                            "index": 1,
                            "accepted": True,
                            "txId": second["txId"],
                            "taskId": "task-1",
                        },
                        {
                            "index": 2,
                            "accepted": False,
                            "error": "model not allowed",
                            "txId": "bad",
                        },
                    ],
                },
            )

    with _serve(Handler) as base_url:
        client = Client(base_url=base_url, api_key="tenant-key")
        wallet = load_wallet({"privateKey": SEED_B64})
        response = client.infer_batch(
            [
                {"input": "first"},
                {"input": "second", "model": "tiny-gpt-v1", "max_fee": 7},
                "third",
            ],
            wallet,
            max_fee=11,
        )

    assert response["accepted"] == 2
    assert response["rejected"] == 1
    assert [item["index"] for item in response["results"]] == [0, 1, 2]
    assert response["results"][0]["taskId"] == "task-0"
    assert response["results"][2]["accepted"] is False
    assert response["results"][2]["error"] == "model not allowed"

    post = next(call for call in calls if call["path"] == "/v1/infer/batch")
    assert post["method"] == "POST"
    assert post["authorization"] == "Bearer tenant-key"
    transactions = batch["body"]["transactions"]
    assert len(transactions) == 3
    for tx in transactions:
        assert len(tx["txId"]) == 64
        assert len(tx["signature"]) == 128
        assert len(tx["publicKey"]) == 64
        assert tx["data"]["type"] == "submitInference"
        assert tx["to"] == ""
        assert tx["value"] == 0
    assert [tx["nonce"] for tx in transactions] == [5, 6, 7]
    assert transactions[0]["data"]["data"]["input"] == "first"
    assert transactions[0]["data"]["data"]["maxFee"] == 11
    assert transactions[1]["data"]["data"]["targetModel"] == "tiny-gpt-v1"
    assert transactions[1]["data"]["data"]["maxFee"] == 7
    assert transactions[2]["data"]["data"]["input"] == "third"
    assert len({tx["txId"] for tx in transactions}) == 3


def test_infer_batch_continues_implicit_nonces_from_explicit_items():
    batch = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path.startswith("/account/"):
                _send_json(self, 200, {"balance": 1_000_000, "nonce": 4})
            elif self.path == "/v1/input-key":
                _send_json(self, 404, {"error": "not found"})
            else:
                _send_json(self, 500, {"error": self.path})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            batch["body"] = body
            if self.path != "/v1/infer/batch":
                _send_json(self, 500, {"error": self.path})
                return
            _send_json(
                self,
                200,
                {
                    "accepted": len(body["transactions"]),
                    "rejected": 0,
                    "results": [
                        {
                            "index": index,
                            "accepted": True,
                            "txId": tx["txId"],
                            "taskId": f"task-{index}",
                        }
                        for index, tx in enumerate(body["transactions"])
                    ],
                },
            )

    with _serve(Handler) as base_url:
        client = Client(base_url=base_url)
        wallet = load_wallet({"privateKey": SEED_B64})
        client.infer_batch(
            [
                "implicit-first",
                {"input": "explicit-mid", "nonce": 20},
                "implicit-after",
                {"input": "explicit-later", "nonce": 30},
                "implicit-last",
            ],
            wallet,
        )

    assert [tx["nonce"] for tx in batch["body"]["transactions"]] == [
        5,
        20,
        21,
        30,
        31,
    ]


def test_infer_batch_returns_per_item_errors_when_all_items_are_rejected():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path.startswith("/account/"):
                _send_json(self, 200, {"balance": 1_000_000, "nonce": 4})
            elif self.path == "/v1/input-key":
                _send_json(self, 404, {"error": "not found"})
            else:
                _send_json(self, 500, {"error": self.path})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path != "/v1/infer/batch":
                _send_json(self, 500, {"error": self.path})
                return
            _send_json(
                self,
                200,
                {
                    "accepted": 0,
                    "rejected": len(body["transactions"]),
                    "results": [
                        {
                            "index": index,
                            "accepted": False,
                            "error": "policy rejected",
                            "txId": tx["txId"],
                        }
                        for index, tx in enumerate(body["transactions"])
                    ],
                },
            )

    with _serve(Handler) as base_url:
        client = Client(base_url=base_url)
        wallet = load_wallet({"privateKey": SEED_B64})
        response = client.infer_batch(["first", "second"], wallet)

    assert response["accepted"] == 0
    assert response["rejected"] == 2
    assert [item["accepted"] for item in response["results"]] == [
        False,
        False,
    ]
    assert [item["error"] for item in response["results"]] == [
        "policy rejected",
        "policy rejected",
    ]


def test_infer_batch_validates_size_before_any_request():
    client = Client(base_url="http://127.0.0.1:1")
    wallet = load_wallet({"privateKey": SEED_B64})

    with pytest.raises(ValueError, match="between 1 and 32 requests"):
        client.infer_batch([], wallet)
    with pytest.raises(ValueError, match="between 1 and 32 requests"):
        client.infer_batch(
            [f"item-{index}" for index in range(MAX_BATCH_SIZE + 1)],
            wallet,
        )
    with pytest.raises(TypeError, match="must be a list"):
        client.infer_batch("not-a-list", wallet)


def test_stream_task_parses_crlf_and_lf_frames_in_order():
    seen_headers = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            seen_headers.append(dict(self.headers))
            if self.path != "/v1/tasks/task-1/events":
                _send_json(self, 404, {"error": "not found"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(b": heartbeat\r\n\r\n")
            self.wfile.write(b"event: status\r\n")
            self.wfile.write(
                b'data: {"taskId":"task-1","status":"pending"}\r\n\r\n'
            )
            self.wfile.write(b": keep-alive\n\n")
            self.wfile.write(b"event: result\r\n")
            self.wfile.write(b'data: {"taskId":"task-1",\r\n')
            self.wfile.write(
                b'data: "task":{"taskId":"task-1","status":"completed",\r\n'
            )
            self.wfile.write(b'data: "resultOutput":"hello"},\r\n')
            self.wfile.write(
                b'data: "receipt":{"taskId":"task-1","outputHash":"aa"}}\r\n'
            )
            self.wfile.write(b"\r\n")
            self.wfile.write(b"event: done\r\n")
            self.wfile.write(b'data: {"taskId":"task-1"}\r\n')
            self.wfile.flush()

    with _serve(Handler) as base_url:
        client = Client(base_url=base_url, api_key="sse-key")
        callback_events = []
        events = list(
            client.stream_task(
                "task-1",
                callback=lambda event, data: callback_events.append(
                    (event, data)
                ),
            )
        )

    assert [event for event, _data in events] == [
        "status",
        "result",
        "done",
    ]
    assert events[0][1] == {"taskId": "task-1", "status": "pending"}
    assert events[1][1] == {
        "taskId": "task-1",
        "task": {
            "taskId": "task-1",
            "status": "completed",
            "resultOutput": "hello",
        },
        "receipt": {"taskId": "task-1", "outputHash": "aa"},
    }
    assert events[1][1]["task"]["resultOutput"] == "hello"
    assert "resultOutput" not in events[1][1]
    assert events[2][1] == {"taskId": "task-1"}
    assert callback_events == events
    assert seen_headers[0].get("Authorization") == "Bearer sse-key"
    assert seen_headers[0].get("Accept") == "text/event-stream"
