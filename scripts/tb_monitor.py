#!/usr/bin/env python3
"""Progress and serving metrics for the Terminal-Bench eval run.

scripts/monitor.py is bound to the SWE-Gym run's shape -- fixed 2000-instance
segments, a --subset manifest, one `<id>.traj.json` per instance -- none of
which Terminal-Bench produces. This reads what the TB run actually writes.

Acceptance length comes from the trace rather than from the endpoint's
`vllm:spec_decode_*` counters. Those counters are server-global, so they blend
in any other workload sharing the endpoint; `--per-request-spec-decode-metrics
summary` puts the same numbers in each response body, which the proxy records.
Totals are pooled (sum accepted / sum steps), not an average of per-call means,
so long calls are not down-weighted against short ones.
"""

import argparse
import glob
import json
import os
import re
import time
import urllib.request
from pathlib import Path

SEGMENTS = ["low", "medium", "high", "xhigh"]
METRIC_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9.eE+-]+)$")


def scrape(url, token=""):
    """Sum each metric family across labels. {} if the endpoint is unreachable."""
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}"} if token else {}
    )
    try:
        body = urllib.request.urlopen(req, timeout=10).read().decode()
    except Exception:
        return {}
    out = {}
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        m = METRIC_RE.match(line.strip())
        if m:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
    return out


def read_results(runs_root):
    """Per-segment resolved/total, from each `tb run` output directory."""
    segs = {}
    for seg in SEGMENTS:
        resolved = total = 0
        tasks = {}
        for rj in glob.glob(os.path.join(runs_root, f"tb-{seg}", "*", "results.json")):
            try:
                d = json.load(open(rj))
            except Exception:
                continue
            for r in d.get("results", []):
                tid = r.get("task_id")
                ok = bool(r.get("is_resolved"))
                tasks[tid] = ok
        for ok in tasks.values():
            total += 1
            resolved += ok
        segs[seg] = {"resolved": resolved, "total": total, "tasks": tasks}
    return segs


class TraceReader:
    """Incremental reader -- calls.jsonl is append-only, so track a byte offset."""

    def __init__(self, path):
        self.path = Path(path)
        self.off = 0
        self.calls = self.errors = self.ok = 0
        self.tok_in = self.tok_out = 0
        self.lat = 0.0
        self.steps = self.accepted = self.drafted = 0
        self.by_strength = {}

    def poll(self):
        if not self.path.exists():
            return
        with open(self.path) as f:
            f.seek(self.off)
            for line in f:
                if not line.endswith("\n"):
                    break  # partial trailing write
                self.off += len(line.encode())
                self.calls += 1
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                rsp = rec.get("response", {})
                if "error" in rsp:
                    # A failed call's latency is its timeout, not the model's
                    # response time -- averaging it in makes an outage look like
                    # a slow endpoint. Count it and move on.
                    self.errors += 1
                    continue
                self.lat += rec.get("latency", 0) or 0
                self.ok += 1
                u = rsp.get("usage") or {}
                self.tok_in += u.get("prompt_tokens", 0) or 0
                self.tok_out += u.get("completion_tokens", 0) or 0
                sd = (rsp.get("metrics") or {}).get("speculative_decoding") or {}
                steps = sd.get("num_spec_steps") or 0
                acc = sd.get("num_accepted_draft_tokens") or 0
                dft = sd.get("num_draft_tokens") or 0
                self.steps += steps
                self.accepted += acc
                self.drafted += dft
                # Attribute to the segment via the request's own reasoning_strength,
                # so a resumed or re-ordered run still buckets correctly.
                ctk = (rec.get("request") or {}).get("chat_template_kwargs") or {}
                s = ctk.get("reasoning_strength", "?")
                b = self.by_strength.setdefault(
                    s, {"calls": 0, "steps": 0, "accepted": 0, "drafted": 0, "out": 0}
                )
                b["calls"] += 1
                b["steps"] += steps
                b["accepted"] += acc
                b["drafted"] += dft
                b["out"] += u.get("completion_tokens", 0) or 0

    @staticmethod
    def _accept_len(accepted, steps):
        return 1 + accepted / steps if steps else None

    def summary(self):
        return {
            "calls": self.calls,
            "errors": self.errors,
            "prompt_tokens": self.tok_in,
            "completion_tokens": self.tok_out,
            "avg_latency_s": round(self.lat / self.ok, 2) if self.ok else None,
            "acceptance_length": _r(self._accept_len(self.accepted, self.steps)),
            "draft_acceptance_rate": _r(
                self.accepted / self.drafted if self.drafted else None
            ),
            "by_strength": {
                s: {
                    "calls": b["calls"],
                    "completion_tokens": b["out"],
                    "acceptance_length": _r(self._accept_len(b["accepted"], b["steps"])),
                    "draft_acceptance_rate": _r(
                        b["accepted"] / b["drafted"] if b["drafted"] else None
                    ),
                }
                for s, b in sorted(self.by_strength.items())
            },
        }


def _r(v, nd=3):
    return round(v, nd) if isinstance(v, (int, float)) else None


def render(snap):
    t, segs, v = snap["traces"], snap["segments"], snap.get("vllm", {})
    done = sum(s["resolved"] for s in segs.values())
    total = sum(s["total"] for s in segs.values())
    L = [f"=== {snap['time']}  Terminal-Bench frozen-40 ==="]
    for seg in SEGMENTS:
        s = segs[seg]
        bar = "".join("#" if v else "." for v in s["tasks"].values())
        L.append(
            f"  {seg:6s} resolved={s['resolved']}/{s['total']:<3d} [{bar:<10s}]"
        )
    L.append(f"  TOTAL  resolved={done}/{total} of 40")
    L.append(
        f"  TRACES calls={t['calls']} errors={t['errors']} "
        f"out_tok={t['completion_tokens']} in_tok={t['prompt_tokens']} "
        f"avg_lat={t['avg_latency_s']}s"
    )
    L.append(
        f"  SPEC   accept_len={t['acceptance_length']} "
        f"draft_accept_rate={t['draft_acceptance_rate']}   (from trace, per-request)"
    )
    for s, b in t["by_strength"].items():
        L.append(
            f"    {s:8s} calls={b['calls']:<5d} out_tok={b['completion_tokens']:<8d} "
            f"accept_len={b['acceptance_length']} rate={b['draft_acceptance_rate']}"
        )
    if v.get("up"):
        L.append(
            f"  VLLM   running={v.get('running')} waiting={v.get('waiting')} "
            f"kv={v.get('kv_cache_pct')}% gen={v.get('gen_tok_s')}tok/s"
        )
    elif v:
        L.append("  VLLM   DOWN / unreachable")
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traces", default="/data/traces/tb/calls.jsonl")
    p.add_argument("--runs", default="/data/runs", help="parent of the tb-<seg> dirs")
    p.add_argument("--state-dir", default="/data/monitor-tb")
    p.add_argument("--vllm", default=os.environ.get("VLLM_METRICS_URL", ""))
    p.add_argument("--token", default=os.environ.get("UPSTREAM_API_KEY", ""))
    p.add_argument("--interval", type=int, default=60)
    p.add_argument("--once", action="store_true")
    a = p.parse_args()

    state = Path(a.state_dir)
    state.mkdir(parents=True, exist_ok=True)
    reader = TraceReader(a.traces)
    prev_m, prev_t = None, None

    while True:
        now = time.time()
        reader.poll()
        snap = {
            "ts": now,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "segments": read_results(a.runs),
            "traces": reader.summary(),
        }
        if a.vllm:
            m = scrape(a.vllm, a.token)
            if m:
                dt = now - prev_t if prev_t else 0
                gen = None
                if prev_m and dt > 0:
                    d = m.get("vllm:generation_tokens_total", 0) - prev_m.get(
                        "vllm:generation_tokens_total", 0
                    )
                    gen = round(d / dt, 1) if d >= 0 else None
                snap["vllm"] = {
                    "up": True,
                    "running": m.get("vllm:num_requests_running"),
                    "waiting": m.get("vllm:num_requests_waiting"),
                    "kv_cache_pct": round(100 * m.get("vllm:kv_cache_usage_perc", 0), 1),
                    "gen_tok_s": gen,
                }
                prev_m, prev_t = m, now
            else:
                snap["vllm"] = {"up": False}

        (state / "status.json").write_text(json.dumps(snap, indent=2))
        text = render(snap)
        (state / "status.txt").write_text(text)
        print(text, flush=True)

        if a.once:
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
