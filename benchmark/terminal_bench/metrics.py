#!/usr/bin/env python3
"""Baseline numbers for the Terminal-Bench eval run.

Acceptance length is pooled from the per-request `metrics.speculative_decoding`
block in the trace -- not from the endpoint's `vllm:spec_decode_*` counters,
which are server-global and blend in anything else sharing the endpoint.

Excluded from the speculative-decoding table:
  * errored calls (the connectivity outage window), which have no generation
  * calls with no recorded `reasoning_strength` (hand-sent connectivity probes)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

SEGMENTS = ["low", "medium", "high", "xhigh"]


def spec_table(src):
    agg = defaultdict(lambda: defaultdict(float))
    errored = unknown = 0
    for line in open(src):
        if not line.strip():
            continue
        r = json.loads(line)
        resp = r["response"]
        if "error" in resp:
            errored += 1
            continue
        strength = (r["request"].get("chat_template_kwargs") or {}).get("reasoning_strength")
        if strength not in SEGMENTS:
            unknown += 1
            continue
        a = agg[strength]
        a["calls"] += 1
        a["out_tok"] += (resp.get("usage") or {}).get("completion_tokens", 0) or 0
        a["latency"] += r.get("latency", 0) or 0
        sd = ((resp.get("metrics") or {}).get("speculative_decoding")) or {}
        a["steps"] += sd.get("num_spec_steps", 0) or 0
        a["accepted"] += sd.get("num_accepted_draft_tokens", 0) or 0
        a["drafted"] += sd.get("num_draft_tokens", 0) or 0
    return agg, errored, unknown


def fmt(agg):
    rows = []
    pooled = defaultdict(float)
    for seg in SEGMENTS:
        a = agg.get(seg)
        if not a:
            continue
        for k, v in a.items():
            pooled[k] += v
        rows.append((seg, a))
    rows.append(("pooled", pooled))

    print(f"{'strength':<9} {'calls':>6} {'out_tok':>9} {'accept_len':>11} "
          f"{'draft_rate':>11} {'tok/s':>8}")
    for seg, a in rows:
        accept = 1 + a["accepted"] / a["steps"] if a["steps"] else float("nan")
        rate = a["accepted"] / a["drafted"] if a["drafted"] else float("nan")
        tps = a["out_tok"] / a["latency"] if a["latency"] else float("nan")
        print(f"{seg:<9} {int(a['calls']):>6} {int(a['out_tok']):>9,} "
              f"{accept:>11.3f} {rate:>11.3f} {tps:>8.1f}")


def completion(runs_root):
    """Latest result per task_id, across re-run directories."""
    by_task = {}
    for seg in SEGMENTS:
        segdir = Path(runs_root) / f"tb-{seg}"
        if not segdir.is_dir():
            continue
        for rundir in sorted(segdir.iterdir()):        # sorted == chronological
            f = rundir / "results.json"
            if not f.is_file():
                continue
            for res in json.load(open(f))["results"]:
                by_task[res["task_id"]] = (seg, res.get("is_resolved"),
                                           res.get("failure_mode"))

    per_seg = defaultdict(lambda: [0, 0])
    modes = defaultdict(int)
    graded = resolved = null = 0
    for task, (seg, ok, mode) in sorted(by_task.items()):
        if not ok:                                     # failure modes of the non-resolved only
            modes[mode] += 1
        if ok is None:
            null += 1
            continue
        graded += 1
        per_seg[seg][1] += 1
        if ok:
            resolved += 1
            per_seg[seg][0] += 1

    print(f"\n{len(by_task)} tasks with results")
    for seg in SEGMENTS:
        n, d = per_seg[seg]
        if d:
            print(f"  {seg:<7} {n}/{d}")
    pct = 100 * resolved / graded if graded else float("nan")
    print(f"  total   {resolved}/{graded} ({pct:.1f}%)"
          + (f", {null} ungraded (resolved: null)" if null else ""))
    print("  failure modes (unresolved): " + ", ".join(f"{k}={v}" for k, v in
                                          sorted(modes.items(), key=lambda kv: -kv[1])))
    return by_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--runs", default="/data/runs")
    ap.add_argument("--per-task", action="store_true", help="list every task")
    args = ap.parse_args()

    agg, errored, unknown = spec_table(args.trace)
    print(f"{args.trace}: excluded {errored} errored calls, "
          f"{unknown} calls with no reasoning_strength\n")
    fmt(agg)
    by_task = completion(args.runs)
    if args.per_task:
        print()
        for task, (seg, ok, mode) in sorted(by_task.items()):
            flag = "resolved" if ok else ("UNGRADED" if ok is None else "failed")
            print(f"  {task:<42} {seg:<7} {flag:<9} {mode}")


if __name__ == "__main__":
    main()
