#!/usr/bin/env python3
"""Paired comparisons inside each bucket x concurrency cell of the task-1 sweep.

Pairing is on `replay_of` -- the same recorded call, byte-identical prompt -- so
the only thing that differs between the two sides is the drafter. Under greedy
decoding the two sides should also have emitted the same tokens, which
`sweep_divergence.py` checks separately; where they did not, the pairing is
still valid but the difference includes a decoding difference.

Every comparison is printed next to the repeat control (`dflash2` r1 vs r2 in
the same cell). A delta without its control is not a result: the cells here hold
120-360 calls, far fewer than the 1,753 that gave the +-0.11 interval quoted in
docs/RESULTS-paired.md, so the control is what says whether a delta means
anything at this sample size.
"""
import argparse, json, math, random
from collections import defaultdict

P = json.load(open("/mnt/data/eval/sweep/percall.json"))
BL = {"b64_128": "64-128", "b128_256": "128-256", "b256_1K": "256-1K",
      "bge1K": ">=1K", "full": "full set"}


def split(cell):
    p = cell.split("__")
    return (p[0], p[1], p[2], p[-1]) if len(p) >= 4 else None


def sw(rows):
    a = sum(r["acc"] for r in rows); s = sum(r["steps"] for r in rows)
    return 1 + a / s if s else float("nan")


def boot(ka, kb, n=2000, seed=0):
    p = lambda k: 1 + sum(x[0] for x in k) / max(sum(x[1] for x in k), 1)
    d0 = p(ka) - p(kb)
    rnd = random.Random(seed); N = len(ka); ds = []
    for _ in range(n):
        idx = [rnd.randrange(N) for _ in range(N)]
        ds.append(p([ka[i] for i in idx]) - p([kb[i] for i in idx]))
    ds.sort()
    return d0, ds[int(.025 * n)], ds[int(.975 * n) - 1]


def compare(A, B, minsteps=5):
    ids = sorted(set(A) & set(B))
    if not ids:
        return None
    ka = [(A[i]["acc"], A[i]["steps"]) for i in ids]
    kb = [(B[i]["acc"], B[i]["steps"]) for i in ids]
    d, lo, hi = boot(ka, kb)
    m = [i for i in ids if A[i]["steps"] >= minsteps and B[i]["steps"] >= minsteps]
    xa = [1 + A[i]["acc"] / A[i]["steps"] for i in m]
    xb = [1 + B[i]["acc"] / B[i]["steps"] for i in m]
    dm = [u - v for u, v in zip(xa, xb)]
    if len(dm) > 1:
        mu = sum(dm) / len(dm)
        sd = math.sqrt(sum((x - mu) ** 2 for x in dm) / (len(dm) - 1))
        ci = 1.96 * sd / math.sqrt(len(dm))
        w = sum(1 for x in dm if x > 0)
    else:
        mu = ci = float("nan"); w = 0
    return dict(n=len(ids), sw_a=1 + sum(x[0] for x in ka) / max(sum(x[1] for x in ka), 1),
                sw_b=1 + sum(x[0] for x in kb) / max(sum(x[1] for x in kb), 1),
                d=d, lo=lo, hi=hi, n_pr=len(dm), pr_d=mu, pr_ci=ci,
                win=(w / len(dm) if dm else float("nan")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="dflash2", help="baseline drafter to compare against")
    a = ap.parse_args()

    cells = defaultdict(dict)
    for cell, pc in P.items():
        s = split(cell)
        if not s:
            continue
        label, bucket, conc, rep = s
        cells[(bucket, conc)][(label, rep)] = pc

    for key in sorted(cells, key=lambda k: (int(k[1][1:]), k[0])):
        bucket, conc = key
        by = cells[key]
        ref = (a.ref, "r1")
        if ref not in by:
            continue
        print(f"\n== output bucket {BL.get(bucket, bucket)}, concurrency {conc[1:]} "
              f"-- A vs {a.ref} (greedy) ==")
        print(f"{'A':<24}{'n':>6}{'sw A':>9}{'sw B':>9}{'sw delta':>10}"
              f"{'95% CI':>22}{'per-req delta':>15}{'win':>7}")
        rows = [(k, v) for k, v in sorted(by.items()) if k != ref]
        # put the repeat control last so it reads as the yardstick
        rows.sort(key=lambda kv: (kv[0][0] == a.ref, kv[0]))
        for (label, rep), pc in rows:
            r = compare(pc, by[ref])
            if not r:
                continue
            tag = f"{label}" + ("" if rep == "r1" else f" [{rep}]")
            if label == a.ref:
                tag = f"*control* {a.ref} {rep}"
            ci = f"[{r['lo']:+.3f}, {r['hi']:+.3f}]"
            pr = f"{r['pr_d']:+.3f}+-{r['pr_ci']:.3f}"
            print(f"{tag:<24}{r['n']:>6}{r['sw_a']:>9.3f}{r['sw_b']:>9.3f}"
                  f"{r['d']:>+10.3f}{ci:>22}{pr:>15}{r['win']:>7.0%}")


if __name__ == "__main__":
    main()
