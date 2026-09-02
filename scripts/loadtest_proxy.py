import json, os, statistics, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROXY = os.environ.get("PROXY_URL", "http://127.0.0.1:8080/v1/chat/completions")
CONCURRENCY = int(sys.argv[1]) if len(sys.argv) > 1 else 40
REQUESTS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
PREFIX_TOKENS = 30000

REPLY = {
    "id": "mock", "provider": "Mock", "model": "mock",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                 "finish_reason": "stop"}],
    "usage": {"prompt_tokens": PREFIX_TOKENS, "completion_tokens": 8,
              "total_tokens": PREFIX_TOKENS + 8, "cost": 0.0,
              "prompt_tokens_details": {"cached_tokens": PREFIX_TOKENS - 100},
              "completion_tokens_details": {"reasoning_tokens": 0}},
}


class Mock(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps(REPLY).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def call(i):
    payload = {
        "model": "mock",
        "messages": [{"role": "user", "content": "x" * 4 * PREFIX_TOKENS}],
        "max_tokens": 8,
    }
    t0 = time.time()
    req = urllib.request.Request(PROXY, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=120).read()
        return time.time() - t0, None
    except Exception as e:
        return time.time() - t0, type(e).__name__


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 9099), Mock)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(1)

    t0 = time.time()
    with ThreadPoolExecutor(CONCURRENCY) as ex:
        results = list(ex.map(call, range(REQUESTS)))
    elapsed = time.time() - t0

    lat = sorted(r[0] for r in results)
    errs = [r[1] for r in results if r[1]]
    print(f"concurrency={CONCURRENCY} requests={REQUESTS} elapsed={elapsed:.1f}s "
          f"rps={REQUESTS/elapsed:.1f}")
    print(f"latency p50={lat[len(lat)//2]*1000:.0f}ms p95={lat[int(len(lat)*.95)]*1000:.0f}ms "
          f"max={lat[-1]*1000:.0f}ms")
    print(f"errors={len(errs)} {dict((e, errs.count(e)) for e in set(errs))}")
    server.shutdown()
