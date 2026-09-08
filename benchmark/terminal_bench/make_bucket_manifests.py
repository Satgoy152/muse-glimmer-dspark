#!/usr/bin/env python3
"""Freeze the four output-length bucket manifests for the bucket x concurrency sweep.

Two decisions are baked in here on purpose, both of which the sweep is worthless
without:

1. **Bucketing is on the ORIGINAL recording's `completion_tokens`**, never on
   what a drafter emits at replay time. Bucketing on the replay selects on the
   outcome, and `docs/RESULTS-paired.md` shows the control then swings as hard
   as the effect.

2. **The requests are rewritten to greedy** (`temperature: 0`, `top_p`/`top_k`
   dropped). The recorded set is temperature 1.0 / top_k 64, so every drafter
   would otherwise decode a different token sequence and the comparison would
   carry sampling noise. Greedy rows are NOT comparable to the temperature-1.0
   rows already in `docs/RESULTS-paired.md`.

Sampling, when a bucket is subsampled, is round-robin across trajectories in
recorded order. That maximises the number of distinct trajectories in the cell,
which matters because `replay.py` gives one worker per trajectory -- a cell with
12 trajectories cannot actually run at concurrency 32 no matter what is asked
for. The per-bucket sizes are a constant in this file, fixed before any result
was looked at, and the manifest metadata records them alongside a sha256 of each
file so a later run can prove it replayed the same bytes.
"""
import argparse, hashlib, json, os, random
from collections import defaultdict

SEGMENTS = ["low", "medium", "high", "xhigh"]
# Chosen 2026-09-08, before any cell was run, to fit ~100K output tokens per
# (drafter, concurrency) pass -- which is what makes 8 drafters x 4 concurrencies
# fit an overnight budget on one H200. Weighted toward the cheap buckets so the
# expensive >=1K bucket does not eat the whole run.
SIZES = {"64-128": 160, "128-256": 120, "256-1K": 60, ">=1K": 20}
ORDER = ["64-128", "128-256", "256-1K", ">=1K"]


def bucket(c):
    if c < 64:
        return None          # deliberately outside every bucket
    if c < 128:
        return "64-128"
    if c < 256:
        return "128-256"
    if c < 1024:
        return "256-1K"
    return ">=1K"


def greedy(body):
    """Same request, decoded greedily."""
    b = dict(body)
    b["temperature"] = 0.0
    b.pop("top_p", None)
    b.pop("top_k", None)
    b.pop("stream", None)     # replay.py sets this itself
    return b


def load(src):
    if src.endswith(".parquet"):
        import pyarrow.parquet as pq
        return pq.read_table(src).to_pylist()
    return [json.loads(l) for l in open(src) if l.strip()]


def roundrobin(rows, n):
    """Take n rows, spread as evenly as possible over trajectories."""
    if n >= len(rows):
        return list(rows)
    by = defaultdict(list)
    for r in rows:
        by[r["id"]].append(r)
    for v in by.values():
        v.sort(key=lambda r: float(r["ts"]))
    trajs = sorted(by)
    out, i = [], 0
    while len(out) < n:
        took = False
        for t in trajs:
            if i < len(by[t]):
                out.append(by[t][i])
                took = True
                if len(out) == n:
                    break
        if not took:
            break
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/data/eval/raw.parquet")
    ap.add_argument("--out-dir", default="/mnt/data/eval/manifests")
    ap.add_argument("--greedy", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    rows = load(a.src)
    good = [r for r in rows
            if not r.get("is_error") and r.get("reasoning_strength") in SEGMENTS]
    by_bucket = defaultdict(list)
    for r in good:
        b = bucket(r["completion_tokens"])
        if b:
            by_bucket[b].append(r)
    for v in by_bucket.values():
        v.sort(key=lambda r: (r["id"], float(r["ts"])))

    meta = {"src": a.src, "greedy": bool(a.greedy), "sizes_requested": SIZES,
            "replayable_total": len(good), "buckets": {}}
    for b in ORDER:
        pool = by_bucket[b]
        sel = roundrobin(pool, SIZES[b])
        path = os.path.join(a.out_dir, f"b{b.replace('>=', 'ge').replace('-', '_')}.jsonl")
        with open(path, "w") as fh:
            for r in sel:
                body = json.loads(r["request"]) if isinstance(r["request"], str) else r["request"]
                if a.greedy:
                    body = greedy(body)
                fh.write(json.dumps({
                    "id": r["id"], "ts": r["ts"], "is_error": False,
                    "reasoning_strength": r["reasoning_strength"],
                    "rec_prompt_tokens": r["prompt_tokens"],
                    "rec_completion_tokens": r["completion_tokens"],
                    "bucket": b, "request": body}) + "\n")
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
        meta["buckets"][b] = {
            "path": path, "sha256_16": h, "pool": len(pool), "calls": len(sel),
            "trajs": len(set(r["id"] for r in sel)),
            "rec_out_tok": sum(r["completion_tokens"] for r in sel),
            "rec_out_tok_mean": round(
                sum(r["completion_tokens"] for r in sel) / max(len(sel), 1), 1),
            "rec_out_tok_max": max((r["completion_tokens"] for r in sel), default=0),
        }
        print(f"{b:>8}  pool={len(pool):4d}  took={len(sel):4d}  "
              f"trajs={meta['buckets'][b]['trajs']:3d}  "
              f"rec_out_tok={meta['buckets'][b]['rec_out_tok']:8,d}  sha={h}")

    meta["total_calls"] = sum(v["calls"] for v in meta["buckets"].values())
    meta["total_rec_out_tok"] = sum(v["rec_out_tok"] for v in meta["buckets"].values())
    json.dump(meta, open(os.path.join(a.out_dir, "manifest.json"), "w"), indent=1)
    print(f"\ntotal {meta['total_calls']} calls, "
          f"{meta['total_rec_out_tok']:,} recorded output tokens per pass")
    print(f"wrote {a.out_dir}/manifest.json")


if __name__ == "__main__":
    main()
