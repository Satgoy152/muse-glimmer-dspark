#!/usr/bin/env python3
"""Replay recorded Terminal-Bench calls against a speculator endpoint.

Why replay rather than re-run the harness: acceptance length is a property of
the drafter, so every drafter must actually generate -- but it does not need the
agent loop or the per-task containers to do it. Feeding all drafters the same
frozen prompt set also removes trajectory drift as a confound, which an
independent agentic run per drafter would reintroduce.

What it does not give: `is_resolved`. Task completion needs the real harness.

Output schema is proxy.py's, so filter_canary.py and metrics.py read it
unchanged; ttft/tpot are extra keys those two ignore.

Calls within a trajectory are replayed in recorded order on one worker, so
prefix caching sees the same growing conversation it saw originally. Workers
run over different trajectories.
"""

import argparse, json, os, queue, sys, threading, time, urllib.error, urllib.request
from collections import defaultdict
from pathlib import Path

SEGMENTS = ["low", "medium", "high", "xhigh"]


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

    # streaming: first content chunk marks TTFT; last usage chunk carries the
    # per-request speculative_decoding block
    chunks, usage, metrics = [], None, None
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
                chunks.append(delta)
    lat = round(time.time() - t0, 3)
    content = "".join(c.get("content") or "" for c in chunks)
    resp = {"id": fid, "model": model, "usage": usage,
            "choices": [{"index": 0, "finish_reason": fr,
                         "message": {"role": "assistant", "content": content}}]}
    if metrics:
        resp["metrics"] = metrics
    return resp, lat, ttft


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src", help="raw parquet or jsonl of recorded calls")
    p.add_argument("--url", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--api-key", default=os.environ.get("UPSTREAM_API_KEY", ""))
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--stream", action="store_true",
                   help="measure TTFT; requires the endpoint to emit the "
                        "per-request metrics block on the final chunk")
    p.add_argument("--model", default="", help="override the model field")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    rows = load_rows(a.src)
    good = [r for r in rows
            if not r.get("is_error") and r.get("reasoning_strength") in SEGMENTS]

    # resume: a preempted node must not redo finished calls
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in open(out):
            try:
                done.add(json.loads(line)["replay_of"])
            except Exception:
                pass
    todo = [r for r in good if r["id"] + ":" + str(r["ts"]) not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(good)} replayable, {len(done)} already done, {len(todo)} to run",
          flush=True)

    by_traj = defaultdict(list)
    for r in todo:
        by_traj[r["id"]].append(r)
    for v in by_traj.values():
        v.sort(key=lambda r: float(r["ts"]))

    q = queue.Queue()
    for t in sorted(by_traj, key=lambda k: -len(by_traj[k])):   # longest first
        q.put(by_traj[t])

    lock = threading.Lock()
    fh = open(out, "a", buffering=1)
    state = {"n": 0, "err": 0, "tok": 0, "t0": time.time()}
    total = len(todo)

    def worker():
        while True:
            try:
                traj = q.get_nowait()
            except queue.Empty:
                return
            for r in traj:
                body = json.loads(r["request"]) if isinstance(r["request"], str) else r["request"]
                if a.model:
                    body = dict(body, model=a.model)
                try:
                    resp, lat, ttft = post(a.url, body, a.api_key, a.timeout, a.stream)
                except urllib.error.HTTPError as e:
                    resp, lat, ttft = {"error": e.read().decode()[:2000]}, 0, None
                except Exception as e:
                    resp, lat, ttft = {"error": f"{type(e).__name__}: {e}"[:2000]}, 0, None

                ct = ((resp.get("usage") or {}) or {}).get("completion_tokens") or 0
                tpot = round((lat - (ttft or 0)) / max(ct - 1, 1), 5) if ct > 1 else None
                rec = {"traj": r["id"], "ts": time.time(), "latency": lat,
                       "ttft": ttft, "tpot": tpot,
                       "replay_of": r["id"] + ":" + str(r["ts"]),
                       "request": body, "response": resp}
                with lock:
                    fh.write(json.dumps(rec) + "\n")
                    state["n"] += 1
                    state["tok"] += ct
                    if "error" in resp:
                        state["err"] += 1
                    n = state["n"]
                    if n % 25 == 0 or n == total:
                        el = time.time() - state["t0"]
                        print(f"  {n}/{total} calls  {state['err']} err  "
                              f"{state['tok']:,} tok  {state['tok']/el:.0f} tok/s  "
                              f"{el/60:.1f} min elapsed", flush=True)

    ths = [threading.Thread(target=worker, daemon=True) for _ in range(a.concurrency)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    el = time.time() - state["t0"]
    print(f"DONE {state['n']} calls, {state['err']} errors, {state['tok']:,} tok "
          f"in {el/60:.1f} min ({state['tok']/el:.0f} tok/s aggregate) -> {a.out}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
