#!/usr/bin/env python3
"""Terminal-Bench frozen-40 eval run, one segment per reasoning strength.

Mirrors run_streaming.py's watchdog (see a1d514a): the harness talks to the
recording proxy, not to the endpoint, so a healthy endpoint says nothing about
whether calls can be made. When the proxy died during the first TB run the
endpoint kept answering 200 and nothing noticed. The endpoint matters too --
these GPU boxes get preempted mid-run.

Terminal-Bench writes one results.json per task, so an interrupted run resumes
by simply not passing the task IDs that already have results. Nothing is
read-modify-written, unlike mini-swe-agent's preds.json.

The effort split is deterministic: tasks sorted by (difficulty, task_id) and
dealt round-robin into four buckets, so every segment gets the same difficulty
mix rather than one segment absorbing all the hard tasks.

Segments run sequentially by default. --segment-concurrency runs several at
once, which is much faster: agentic tasks are mostly not talking to the model
(measured 0.42 concurrent-request-equivalents across 1117 calls, against
max-num-seqs 128), so one segment leaves both the GPU and the generation box
idle. Raising it changes contention, so tokens/s is only comparable between runs
that used the same value; acceptance length is unaffected.
"""

import argparse
import glob
import json
import os
import signal
import socket
import sys
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.request
from pathlib import Path

# tb_agent and mini-swe-setup.sh.j2 live beside this file
HERE = Path(__file__).resolve().parent

SEGMENTS = ["low", "medium", "high", "xhigh"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def split_tasks(manifest):
    tasks = json.load(open(manifest))["tasks"]
    order = {"easy": 0, "medium": 1, "hard": 2}
    tasks.sort(key=lambda t: (order[t["difficulty"]], t["task_id"]))
    buckets = {s: [] for s in SEGMENTS}
    for i, t in enumerate(tasks):
        buckets[SEGMENTS[i % len(SEGMENTS)]].append(t["task_id"])
    return buckets


def completed(runs_root, seg):
    """Task IDs that already have a result, so a resumed run can skip them."""
    done = set()
    for rj in glob.glob(os.path.join(runs_root, f"tb-{seg}", "*", "results.json")):
        try:
            d = json.load(open(rj))
        except Exception:
            continue
        for r in d.get("results", []):
            if r.get("task_id"):
                done.add(r["task_id"])
    return done


def endpoint_ok(url, token, timeout=10):
    if not url:
        return True
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}"} if token else {}
    )
    try:
        return urllib.request.urlopen(req, timeout=timeout).status == 200
    except Exception:
        return False


def proxy_ok(hostport):
    """TCP connect only -- a request through the proxy would land in the trace
    file it is recording into."""
    if not hostport:
        return True
    host, _, port = hostport.rpartition(":")
    try:
        with socket.create_connection((host or "127.0.0.1", int(port)), timeout=5):
            return True
    except Exception:
        return False


class Watchdog(threading.Thread):
    """Stop every harness process when the proxy or endpoint goes away.

    Left alone, `tb run` keeps handing out tasks against a dead upstream and
    records each as unresolved -- indistinguishable, later, from a task the
    model genuinely failed. Stopping instead leaves them simply unrun, and the
    next pass picks them up.

    One watchdog covers all segments rather than one per segment: they would
    otherwise poll the same endpoint N times over, and a dead upstream means
    every segment has to stop anyway.

    Several consecutive failures are required so one transient 504 does not
    trip it.
    """

    def __init__(self, url, token, proxy, fails=3, interval=30):
        super().__init__(daemon=True)
        self.url, self.token, self.proxy = url, token, proxy
        self.fails, self.interval = fails, interval
        self.stop, self.tripped = threading.Event(), False
        self._procs = []
        self._lock = threading.Lock()

    def register(self, proc):
        with self._lock:
            self._procs.append(proc)

    def _kill_all(self):
        with self._lock:
            procs = list(self._procs)
        for proc in procs:
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception as e:
                    log(f"  watchdog: killpg failed: {e}")

    def run(self):
        consecutive = 0
        while not self.stop.is_set():
            ok_ep, ok_px = endpoint_ok(self.url, self.token), proxy_ok(self.proxy)
            if ok_ep and ok_px:
                consecutive = 0
            else:
                consecutive += 1
                what = "endpoint" if not ok_ep else "proxy"
                log(f"  watchdog: {what} unreachable ({consecutive}/{self.fails})")
                if consecutive >= self.fails:
                    self.tripped = True
                    log("  watchdog: stopping all harnesses to avoid burning eval tasks")
                    self._kill_all()
                    return
            self.stop.wait(self.interval)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(HERE / "tasks_frozen40.json"))
    p.add_argument("--dataset-path", default="/data/terminal-bench-1/original-tasks")
    p.add_argument("--runs", default="/data/runs")
    p.add_argument("--model", default="openai/meta-models/Muse-Glimmer-30B")
    p.add_argument("--agent", default="tb_agent:MuseGlimmerMiniAgent")
    p.add_argument("--concurrent", type=int, default=8)
    # `tb` lives beside the interpreter when this runs out of the project venv,
    # and that venv is not necessarily on PATH.
    _tb = Path(sys.executable).parent / "tb"
    p.add_argument("--tb-bin", default=str(_tb) if _tb.exists() else "tb")
    p.add_argument("--segments", default=",".join(SEGMENTS))
    p.add_argument("--segment-concurrency", type=int, default=1,
                   help="segments to run at once; >1 is much faster but changes "
                        "contention, so tokens/s is only comparable across runs "
                        "that used the same value")
    p.add_argument("--health", default=os.environ.get("UPSTREAM_HEALTH_URL", ""),
                   help="endpoint /v1/models URL; watchdog stops the run when it fails")
    p.add_argument("--token", default=os.environ.get("UPSTREAM_API_KEY", ""))
    p.add_argument("--proxy", default=os.environ.get("PROXY_HOSTPORT", ""),
                   help="host:port of the recording proxy")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    buckets = split_tasks(a.manifest)
    segments = a.segments.split(",")

    pending = []
    for seg in segments:
        ids = buckets[seg]
        done = completed(a.runs, seg)
        todo = [i for i in ids if i not in done]
        if not todo:
            log(f"[{seg}] all {len(ids)} tasks already have results, skipping")
            continue
        log(f"[{seg}] {len(todo)}/{len(ids)} outstanding ({len(done)} already done)")
        if a.dry_run:
            log(f"  would run: {' '.join(todo)}")
            continue
        pending.append((seg, ids, todo))

    if a.dry_run or not pending:
        return 0

    wd = Watchdog(a.health, a.token, a.proxy) if (a.health or a.proxy) else None
    if wd:
        wd.start()

    def run_segment(item):
        seg, ids, todo = item
        cmd = [a.tb_bin, "run", "--dataset-path", a.dataset_path,
               "--agent-import-path", a.agent, "-m", a.model,
               "--output-path", os.path.join(a.runs, f"tb-{seg}"),
               "--n-concurrent", str(a.concurrent), "--no-livestream"]
        for i in todo:
            cmd += ["-t", i]

        # Per-segment copy: TB_REASONING_STRENGTH must not be shared mutable
        # state when segments run concurrently.
        env = dict(os.environ)
        env["PYTHONPATH"] = str(HERE)
        env["TB_REASONING_STRENGTH"] = seg

        t0 = time.time()
        # Own process group, so the watchdog can signal the whole tree.
        proc = subprocess.Popen(cmd, env=env, start_new_session=True)
        if wd:
            wd.register(proc)
        rc = proc.wait()
        left = [i for i in ids if i not in completed(a.runs, seg)]
        log(f"[{seg}] exit={rc} in {time.time()-t0:.0f}s, {len(left)} still outstanding")
        return rc

    workers = max(1, min(a.segment_concurrency, len(pending)))
    if workers > 1:
        log(f"running {len(pending)} segments, {workers} at a time")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(run_segment, pending))
    else:
        for item in pending:
            run_segment(item)
            if wd and wd.tripped:
                break

    if wd:
        wd.stop.set()
        wd.join(timeout=5)
        if wd.tripped:
            log("aborting: upstream went away, rerun once it is back")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
