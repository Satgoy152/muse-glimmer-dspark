#!/usr/bin/env python3
"""Do the drafters actually emit the same tokens under greedy decoding?

Speculative decoding is meant to be output-preserving, so at temperature 0 every
drafter should produce the identical completion for the identical prompt, and
the acceptance comparison is then exact rather than paired-with-noise. That is a
claim to test, not to assume: vLLM's continuous batching reduces over whatever
sequences happen to be resident, so greedy is only bitwise reproducible when the
batch shape is, which it is not at concurrency > 1.

The check is per-call `completion_tokens` equality against the no-spec run at
the same bucket and concurrency (or against `dflash2` where no-spec has no cell).
"""
import json, sys
from collections import defaultdict

P = json.load(open("/mnt/data/eval/sweep/percall.json"))
cells = defaultdict(dict)          # (bucket, conc, repeat) -> label -> percall
for cell, pc in P.items():
    parts = cell.split("__")
    if len(parts) < 4:
        continue
    label, bucket, conc, rep = parts[0], parts[1], parts[2], parts[-1]
    cells[(bucket, conc, rep)][label] = pc

print(f"{'bucket':<10}{'conc':>6}{'ref':>10}  {'drafter':<24}{'n':>6}"
      f"{'same ctok':>11}{'same sha':>10}")
for key in sorted(cells):
    bucket, conc, rep = key
    by = cells[key]
    ref = "nospec" if "nospec" in by else ("dflash2" if "dflash2" in by else None)
    if not ref:
        continue
    R = by[ref]
    for label in sorted(by):
        if label == ref:
            continue
        A = by[label]
        ids = sorted(set(R) & set(A))
        if not ids:
            continue
        same_c = sum(1 for i in ids if R[i]["ctok"] == A[i]["ctok"])
        same_s = sum(1 for i in ids if R[i]["sha"] == A[i]["sha"])
        print(f"{bucket:<10}{conc:>6}{ref:>10}  {label:<24}{len(ids):>6}"
              f"{same_c/len(ids):>10.1%}{same_s/len(ids):>10.1%}")
