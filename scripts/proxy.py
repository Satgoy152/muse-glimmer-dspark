import json, os, hashlib, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

UPSTREAM = os.environ.get("UPSTREAM_URL", "https://openrouter.ai/api/v1/chat/completions")
KEY = os.environ.get("UPSTREAM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")

# The Nebius tunnel gateway authenticates in front of vLLM and its scheme is not
# necessarily Bearer, so both are overridable. An unset key sends no header at
# all, which is what vLLM started without --api-key wants.
AUTH_HEADER = os.environ.get("UPSTREAM_AUTH_HEADER", "Authorization")
AUTH_SCHEME = os.environ.get("UPSTREAM_AUTH_SCHEME", "Bearer")

HEADERS = {"Content-Type": "application/json"}
if KEY:
    HEADERS[AUTH_HEADER] = f"{AUTH_SCHEME} {KEY}".strip()
LOG = Path(os.environ.get("TRACE_DIR", "data/traces/dev")) / "calls.jsonl"
LOG.parent.mkdir(parents=True, exist_ok=True)
lock = threading.Lock()
log_file = open(LOG, "a", buffering=1)


def traj_id(messages):
    first_user = next((m for m in messages if m.get("role") == "user"), messages[0])
    return hashlib.sha1(json.dumps(first_user, sort_keys=True).encode()).hexdigest()[:16]


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 512


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        body["stream"] = False

        req = urllib.request.Request(UPSTREAM, json.dumps(body).encode(), dict(HEADERS))
        t0 = time.time()
        try:
            resp, code = json.load(urllib.request.urlopen(req, timeout=900)), 200
        except urllib.error.HTTPError as e:
            resp, code = {"error": e.read().decode()[:2000]}, e.code
        except Exception as e:
            resp, code = {"error": f"{type(e).__name__}: {e}"[:2000]}, 502

        # capture: body we sent (messages + tools + sampling params) and 
        # body we get back (content, reasoning, tool_calls, usage)
        with lock:
            log_file.write(json.dumps({
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
    Server(("0.0.0.0", port), Handler).serve_forever()
