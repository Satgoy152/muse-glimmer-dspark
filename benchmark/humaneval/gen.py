#!/usr/bin/env python3
"""Generate from HumanEval/MBPP prompts against a speculator endpoint.

Why a new driver instead of replay.py: replay.py replays a frozen trace of
recorded Terminal-Bench bodies, so it can only measure prompts we already have
responses for. Acceptance on a coding benchmark needs the model to actually
generate from the benchmark's own prompts. Everything else is deliberately
replay.py's: same output schema (request/response/latency/ttft/tpot), same
streaming metrics extraction, same resume-on-preemption behaviour, so the
metrics side reads both without a special case.

The decoding regime is NOT Terminal-Bench's. The published reference table this
is compared against is greedy at concurrency 1; our replay eval is temperature
1.0 / top_k 64 at concurrency 10. Acceptance is not contention-independent
(we measured +0.04-0.07 going from concurrency 10 to 64), so concurrency is a
required, recorded field rather than a convenience default.

Prompts are sent as a single user message with the raw stub as content, which
is what the upstream speculators HumanEval example does via guidellm's
generative column mapper. No chat_template_kwargs: reasoning_strength is a
Muse-Glimmer-specific knob with no counterpart in the published table, so we
leave the model at its served default and record what came back.
"""

import argparse, json, os, queue, sys, threading, time, urllib.error, urllib.request
from pathlib import Path


def load_rows(src):
    if src.endswith(".parquet"):
        import pyarrow.parquet as pq
        return pq.read_table(src).to_pylist()
    return [json.loads(l) for l in open(src) if l.strip()]


def post(url, body, key, timeout, stream):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = dict(body)
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    else:
        body["stream"] = False
    req = urllib.request.Request(url, json.dumps(body).encode(), headers)

    t0 = time.time()
    ttft = None
    if not stream:
        resp = json.load(urllib.request.urlopen(req, timeout=timeout))
        return resp, round(time.time() - t0, 3), None

    # streaming: first content chunk marks TTFT; the last usage chunk carries
    # the per-request speculative_decoding block
    content, reasoning, usage, metrics = [], [], None, None
    fid = model = fr = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            fid = fid or d.get("id")
            model = model or d.get("model")
            if d.get("usage"):
                usage = d["usage"]
            if d.get("metrics"):
                metrics = d["metrics"]
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                if ch.get("finish_reason"):
                    fr = ch["finish_reason"]
                if ttft is None and (delta.get("content") or delta.get("tool_calls")
                                     or delta.get("reasoning_content")):
                    ttft = round(time.time() - t0, 3)
                if delta.get("content"):
                    content.append(delta["content"])
                # kept separate: the reasoning parser routes thinking here, and
                # feeding it to the code extractor would poison pass@1
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
    lat = round(time.time() - t0, 3)
    msg = {"role": "assistant", "content": "".join(content)}
    if reasoning:
        msg["reasoning_content"] = "".join(reasoning)
    resp = {"id": fid, "model": model, "usage": usage,
            "choices": [{"index": 0, "finish_reason": fr, "message": msg}]}
    if metrics:
        resp["metrics"] = metrics
    return resp, lat, ttft


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src", help="HumanEval/MBPP jsonl with task_id + prompt")
    p.add_argument("--url", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--concurrency", type=int, default=1,
                   help="pinned and recorded: acceptance depends on it")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=2048,
                   help="a runaway generation moves a pooled number by ~0.08, "
                        "so the cap is explicit and truncations are reported")
    p.add_argument("--api-key", default=os.environ.get("UPSTREAM_API_KEY", ""))
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--stream", action="store_true")
    p.add_argument("--model", default="", help="served model id; queried if empty")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    model = a.model
    if not model:
        base = a.url.rsplit("/chat/completions", 1)[0]
        with urllib.request.urlopen(base + "/models", timeout=30) as r:
            model = json.load(r)["data"][0]["id"]
        print(f"served model: {model}", flush=True)

    rows = load_rows(a.src)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in open(out):
            try:
                done.add(json.loads(line)["task_id"])
            except Exception:
                pass
    todo = [r for r in rows if r["task_id"] not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(rows)} prompts, {len(done)} already done, {len(todo)} to run "
          f"(temp={a.temperature} concurrency={a.concurrency} "
          f"max_tokens={a.max_tokens})", flush=True)

    q = queue.Queue()
    for r in todo:
        q.put(r)

    lock = threading.Lock()
    fh = open(out, "a", buffering=1)
    state = {"n": 0, "err": 0, "tok": 0, "t0": time.time()}
    total = len(todo)

    def worker():
        while True:
            try:
                r = q.get_nowait()
            except queue.Empty:
                return
            body = {"model": model, "messages":
                    [{"role": "user", "content": r["prompt"]}],
                    "temperature": a.temperature, "max_tokens": a.max_tokens}
            try:
                resp, lat, ttft = post(a.url, body, a.api_key, a.timeout, a.stream)
            except urllib.error.HTTPError as e:
                resp, lat, ttft = {"error": e.read().decode()[:2000]}, 0, None
            except Exception as e:
                resp, lat, ttft = {"error": f"{type(e).__name__}: {e}"[:2000]}, 0, None

            ct = ((resp.get("usage") or {}) or {}).get("completion_tokens") or 0
            tpot = round((lat - (ttft or 0)) / max(ct - 1, 1), 5) if ct > 1 else None
            rec = {"task_id": r["task_id"], "ts": time.time(), "latency": lat,
                   "ttft": ttft, "tpot": tpot, "concurrency": a.concurrency,
                   "temperature": a.temperature,
                   "entry_point": r.get("entry_point"), "test": r.get("test"),
                   "prompt": r["prompt"], "request": body, "response": resp}
            with lock:
                fh.write(json.dumps(rec) + "\n")
                state["n"] += 1
                state["tok"] += ct
                if "error" in resp:
                    state["err"] += 1
                n = state["n"]
                if n % 20 == 0 or n == total:
                    el = time.time() - state["t0"]
                    print(f"  {n}/{total}  {state['err']} err  {state['tok']:,} tok  "
                          f"{state['tok']/el:.0f} tok/s  {el/60:.1f} min", flush=True)

    ths = [threading.Thread(target=worker, daemon=True) for _ in range(a.concurrency)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    el = time.time() - state["t0"]
    print(f"DONE {state['n']} calls, {state['err']} errors, {state['tok']:,} tok "
          f"in {el/60:.1f} min -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
