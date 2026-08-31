import json, os, hashlib, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

UPSTREAM = "https://openrouter.ai/api/v1/chat/completions"
PROVIDER = {"only": ["parasail/bf16"], "allow_fallbacks": False}
KEY = os.environ["OPENROUTER_API_KEY"]
LOG = Path(os.environ.get("TRACE_DIR", "data/traces/dev")) / "calls.jsonl"
LOG.parent.mkdir(parents=True, exist_ok=True)
lock = threading.Lock()


def traj_id(messages):
    head = json.dumps(messages[:1], sort_keys=True)
    return hashlib.sha1(head.encode()).hexdigest()[:16]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        body["provider"] = PROVIDER
        body["stream"] = False
        body["usage"] = {"include": True}

        req = urllib.request.Request(
            UPSTREAM, json.dumps(body).encode(),
            {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        )
        t0 = time.time()
        try:
            resp, code = json.load(urllib.request.urlopen(req, timeout=900)), 200
        except urllib.error.HTTPError as e:
            resp, code = {"error": e.read().decode()[:2000]}, e.code

        # capture: the exact body we sent (messages + tools + sampling params) and the exact
        # body we got back (content, reasoning, tool_calls, usage), appended as one JSONL line
        with lock, open(LOG, "a") as f:
            f.write(json.dumps({
                "traj": traj_id(body["messages"]),
                "ts": t0,
                "latency": round(time.time() - t0, 3),
                "request": body,
                "response": resp,
            }) + "\n")

        out = json.dumps(resp).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"proxy on :{port} -> {LOG}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
