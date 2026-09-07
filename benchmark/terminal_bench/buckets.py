#!/usr/bin/env python3
"""Pool acceptance by context-length bucket x reasoning strength, per drafter.

Same pooling as metrics.py -- 1 + sum(accepted)/sum(steps) -- applied within
each cell instead of within each segment. Context length is the request's
prompt_tokens, so the bucket is what the drafter actually had to condition on.
"""
import glob, json, os, sys
from collections import defaultdict

EDGES = [0, 2000, 8000, 16000, 32000, 64000, 10**9]
LABELS = ["<2K", "2-8K", "8-16K", "16-32K", "32-64K", "64K+"]
SEGMENTS = ["low", "medium", "high", "xhigh"]


def bucket(n):
    for i in range(len(EDGES) - 1):
        if EDGES[i] <= n < EDGES[i + 1]:
            return LABELS[i]
    return LABELS[-1]


out = {}
for path in sorted(glob.glob("/mnt/data/traces/tb-*/calls.clean.jsonl")):
    name = os.path.basename(os.path.dirname(path))[3:]
    cell = defaultdict(lambda: defaultdict(float))
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        resp = r["response"]
        if "error" in resp:
            continue
        s = (r["request"].get("chat_template_kwargs") or {}).get("reasoning_strength")
        if s not in SEGMENTS:
            continue
        u = resp.get("usage") or {}
        pt = u.get("prompt_tokens") or 0
        sd = ((resp.get("metrics") or {}).get("speculative_decoding")) or {}
        if not sd.get("num_spec_steps"):
            continue
        c = cell[f"{bucket(pt)}|{s}"]
        c["calls"] += 1
        c["steps"] += sd.get("num_spec_steps", 0)
        c["accepted"] += sd.get("num_accepted_draft_tokens", 0)
        c["drafted"] += sd.get("num_draft_tokens", 0)
        c["out_tok"] += u.get("completion_tokens", 0) or 0
    out[name] = {k: dict(v) for k, v in cell.items()}

json.dump({"labels": LABELS, "segments": SEGMENTS, "data": out},
          open("/mnt/data/eval/buckets.json", "w"), indent=1)
print("drafters:", list(out))
for n, cells in out.items():
    print(f"\n=== {n} ===")
    print(f"{'ctx':<8}" + "".join(f"{s:>10}" for s in SEGMENTS))
    for lb in LABELS:
        row = f"{lb:<8}"
        for s in SEGMENTS:
            c = cells.get(f"{lb}|{s}")
            row += f"{(1 + c['accepted']/c['steps']):>10.3f}" if c and c["steps"] else f"{'-':>10}"
        print(row)
