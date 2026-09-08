#!/usr/bin/env python3
"""Measure, and optionally reshape, the assistant-turn size mix of a prepared set.

WHY. Terminal-Bench acceptance is a step-weighted average, and 78.7% of its
steps sit in turns of >=256 tokens. `Satgoy152/Muse-Glimmer-SWE-Gym-2k` is the
opposite shape -- 83.2% of its 154,831 assistant turns are under 64 tokens, and
those short turns carry 28.7% of the supervision against 2.0% of TB's decode
tokens. mini-swe-agent emits terse bash; half its turns are under 25 tokens. So
the drafter is trained mostly on a regime the benchmark barely measures.

WHAT THIS DOES. Turn size is not a row-level property -- a rendered row packs
many assistant turns of every size, so you cannot fix the mix by selecting rows.
It is a *loss-mask* property: each maximal run of 1s in `loss_mask` is one
supervised assistant turn. This script segments those runs, reports the mix, and
(with --target) zeroes whole runs until the surviving supervision matches a
target token distribution. Rows left with fewer than --min-supervised surviving
positions are dropped.

Nothing is re-rendered and no GPU or extraction server is needed: input_ids are
untouched, only loss_mask changes.

THE CEILING. Down-weighting an over-represented bucket is free -- you delete
supervision. Up-weighting is not: you cannot sample tokens that do not exist.
The corpus has ~0% of its tokens in turns >4K, so a target asking for 10.6%
there is unreachable at any dataset size. --renormalize drops unreachable
buckets and rescales the rest, and reports exactly what it could not deliver.
Read that report before trusting the output: the residual is the part of the
distribution gap that needs *different trajectories*, not resampling.

    # measure only -- no decisions needed, this is the missing number
    python3 scripts/reshape_turn_mix.py --data /mnt/data/runs/data-32k-mix

    # reshape toward the Terminal-Bench replay mix
    python3 scripts/reshape_turn_mix.py \\
        --data /mnt/data/runs/data-32k-mix \\
        --out /mnt/data/runs/data-32k-turnmix \\
        --target tb --renormalize
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Upper bound (exclusive) of each bucket, and the label used in reports.
BUCKETS = [(64, "<64"), (256, "64-256"), (1024, "256-1K"), (4096, "1K-4K"),
           (np.inf, ">4K")]

# Terminal-Bench replay, share of DECODE TOKENS per bucket. Steps would be the
# more faithful weight, but the trainer's unit is a supervised token, and the
# two orderings agree; see docs/train.md.
TB_TOKEN_MIX = {"<64": 0.020, "64-256": 0.247, "256-1K": 0.411,
                "1K-4K": 0.216, ">4K": 0.106}


def bucket_of(n: int) -> str:
    for hi, label in BUCKETS:
        if n < hi:
            return label
    return BUCKETS[-1][1]


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of 1s as (start, length). Each is one supervised turn."""
    m = np.asarray(mask, dtype=np.int8)
    if m.size == 0 or not m.any():
        return []
    d = np.diff(np.concatenate(([0], m, [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), (ends - starts).tolist()))


def summarize(per_row_runs: list[list[tuple[int, int]]]) -> dict:
    calls = {label: 0 for _, label in BUCKETS}
    toks = {label: 0 for _, label in BUCKETS}
    lengths = []
    for rs in per_row_runs:
        for _, n in rs:
            b = bucket_of(n)
            calls[b] += 1
            toks[b] += n
            lengths.append(n)
    total_calls = sum(calls.values()) or 1
    total_toks = sum(toks.values()) or 1
    return {
        "calls": calls, "tokens": toks,
        "total_calls": total_calls, "total_tokens": total_toks,
        "median": int(np.median(lengths)) if lengths else 0,
        "p90": int(np.percentile(lengths, 90)) if lengths else 0,
        "p99": int(np.percentile(lengths, 99)) if lengths else 0,
    }


def report(tag: str, s: dict, target: dict | None = None) -> None:
    print(f"\n{tag}: {s['total_calls']:,} turns, {s['total_tokens']:,} supervised tokens")
    print(f"  median {s['median']}  p90 {s['p90']}  p99 {s['p99']}")
    head = f"  {'bucket':<8}{'calls':>10}{'% calls':>9}{'tokens':>12}{'% tokens':>10}"
    if target:
        head += f"{'target':>9}{'gap':>8}"
    print(head)
    for _, label in BUCKETS:
        pc = 100 * s["calls"][label] / s["total_calls"]
        pt = 100 * s["tokens"][label] / s["total_tokens"]
        line = (f"  {label:<8}{s['calls'][label]:>10,}{pc:>8.1f}%"
                f"{s['tokens'][label]:>12,}{pt:>9.1f}%")
        if target:
            tg = 100 * target.get(label, 0.0)
            line += f"{tg:>8.1f}%{pt - tg:>+8.1f}"
        print(line)


def plan(s: dict, target: dict, renormalize: bool) -> tuple[dict, list[str]]:
    """Token budget per bucket that holds `target`, given available supply."""
    notes = []
    avail = s["tokens"]
    unreachable = [b for b, w in target.items() if w > 0 and avail.get(b, 0) == 0]
    tgt = dict(target)
    if unreachable:
        if not renormalize:
            raise SystemExit(
                f"buckets {unreachable} are asked for but have ZERO tokens in this "
                "dataset -- no subset can reach the target. Re-run with "
                "--renormalize to match the reachable buckets and see the residual, "
                "or source trajectories with longer assistant turns."
            )
        lost = sum(tgt.pop(b) for b in unreachable)
        notes.append(
            f"UNREACHABLE {unreachable}: {100*lost:.1f}% of the target token mass has "
            "no supply at any dataset size. Renormalized over the rest; this residual "
            "is a DATA problem, not a sampling one."
        )
        total = sum(tgt.values())
        tgt = {b: w / total for b, w in tgt.items()}
    # Largest total that every requested bucket can still cover at its share.
    budget = min(avail[b] / w for b, w in tgt.items() if w > 0)
    binder = min((b for b in tgt if tgt[b] > 0), key=lambda b: avail[b] / tgt[b])
    notes.append(f"binding bucket: {binder} ({avail[binder]:,} tokens available)")
    return {b: int(budget * w) for b, w in tgt.items()}, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="prepared dataset dir")
    ap.add_argument("--out", default=None, help="write reshaped dataset here")
    ap.add_argument("--target", default=None,
                    help="'tb' for the Terminal-Bench mix, or a JSON object of "
                         "bucket->token fraction")
    ap.add_argument("--renormalize", action="store_true",
                    help="drop unreachable buckets and rescale the rest")
    ap.add_argument("--min-supervised", type=int, default=16,
                    help="drop rows left with fewer surviving positions "
                         "(mirrors prepare-data --minimum-valid-tokens)")
    ap.add_argument("--seed", type=int, default=20260830)
    args = ap.parse_args()

    from datasets import load_from_disk

    ds = load_from_disk(args.data)
    masks = ds.with_format(None)["loss_mask"]
    per_row = [runs(m) for m in masks]
    before = summarize(per_row)
    report("BEFORE", before, TB_TOKEN_MIX if args.target else None)

    if not args.target:
        print("\n(measure only; pass --target tb --out ... to reshape)")
        return 0

    target = TB_TOKEN_MIX if args.target == "tb" else json.loads(args.target)
    budget, notes = plan(before, target, args.renormalize)
    for n in notes:
        print(f"\n  ! {n}")

    rng = np.random.default_rng(args.seed)
    # Shuffle turns within each bucket, then keep until that bucket's budget is met.
    pool: dict[str, list[tuple[int, int, int]]] = {label: [] for _, label in BUCKETS}
    for ri, rs in enumerate(per_row):
        for start, n in rs:
            pool[bucket_of(n)].append((ri, start, n))
    keep: set[tuple[int, int]] = set()
    for label, items in pool.items():
        want = budget.get(label, 0)
        if want <= 0:
            continue
        idx = rng.permutation(len(items))
        got = 0
        for i in idx:
            ri, start, n = items[i]
            if got >= want:
                break
            keep.add((ri, start))
            got += n

    new_masks, kept_rows = [], []
    for ri, (rs, m) in enumerate(zip(per_row, masks)):
        nm = np.zeros(len(m), dtype=np.int8)
        for start, n in rs:
            if (ri, start) in keep:
                nm[start:start + n] = 1
        if int(nm.sum()) >= args.min_supervised:
            kept_rows.append(ri)
            new_masks.append(nm.tolist())

    after = summarize([[(s, n) for s, n in per_row[ri]
                        if (ri, s) in keep] for ri in kept_rows])
    report("AFTER", after, target)
    print(f"\nrows {len(ds):,} -> {len(kept_rows):,}"
          f"   supervised tokens {before['total_tokens']:,} -> {after['total_tokens']:,}"
          f"  ({100*after['total_tokens']/before['total_tokens']:.1f}% retained)")

    if not args.out:
        print("\n(dry run; pass --out to write)")
        return 0

    out = ds.select(kept_rows).remove_columns("loss_mask").add_column("loss_mask", new_masks)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    out.save_to_disk(args.out)
    # token_freq.pt is per-dataset; the trainer derives d2t/t2d from it.
    from speculators.train.vocab_mapping import save_token_frequency_distribution
    save_token_frequency_distribution(dataset=out,
                                      output_path=Path(args.out) / "token_freq.pt")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
