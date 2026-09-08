#!/usr/bin/env python3
"""Resolved rate with a Wilson score interval.

At n=32 the normal approximation is not usable: it goes below zero for a low
rate and its coverage is wrong at the ends. Wilson stays inside [0,1] and holds
its nominal coverage at small n, which is the only reason a resolved rate on a
sample this size is quotable at all.
"""
import argparse, glob, json, math, os


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+",
                    help="swebench run_evaluation report json files, or dirs holding them")
    ap.add_argument("--n-expected", type=int, default=0,
                    help="denominator to use; defaults to each report's own instance count. "
                         "Pass the full task count so instances that crashed count as unresolved.")
    a = ap.parse_args()

    paths = []
    for r in a.reports:
        paths.extend(sorted(glob.glob(os.path.join(r, "*.json"))) if os.path.isdir(r) else [r])

    print(f"{'run':<34}{'resolved':>9}{'n':>5}{'rate':>8}{'95% Wilson':>20}")
    for p in paths:
        d = json.load(open(p))
        res = d.get("resolved_ids") or d.get("resolved") or []
        k = len(res) if isinstance(res, list) else int(res)
        n = a.n_expected or d.get("total_instances") or d.get("submitted_instances") or 0
        pr, lo, hi = wilson(k, n)
        print(f"{os.path.basename(p)[:34]:<34}{k:>9}{n:>5}{pr:>8.1%}"
              f"{f'[{lo:.1%}, {hi:.1%}]':>20}")


if __name__ == "__main__":
    main()
