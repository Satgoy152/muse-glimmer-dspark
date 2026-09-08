#!/usr/bin/env python3
"""Task 3: does the midpoint-beats-final ordering survive repeated replays?

The final DFlash2 run-D checkpoint's regression rested on a single replay that
happened to draw one 58,019-token completion. A step-weighted pool is a
token-weighted average, so one call that long carries as much weight as roughly
four hundred median calls -- it can move the pooled number on its own. This
re-pools every available full-set replay of the three checkpoints, at the
settings those runs were made at (temperature 1.0, top_k 64, concurrency 10,
NUM_SPEC_TOKENS=15), and reports the pooled acceptance twice: as measured, and
with each run's single longest call removed.

Removing the longest call is a diagnostic, not a correction. It says how much of
the ordering rests on one draw; it does not say the draw was invalid.
"""
import argparse, glob, json, math, os, random, re
from collections import defaultdict

# checkpoint -> the traces that are full-set concurrency-10 temperature-1.0
# replays of it. Anything at another concurrency (the _c64-* runs) is excluded:
# t_step and acceptance are both functions of the batch.
GROUPS = {
    "dflash2": ["/mnt/data/traces/tb-dflash2",
                "/mnt/data/traces/tb-dflash2-repeat2",
                "/mnt/data/traces/sweep/dflash2__full__c10__t1p0__r2",
                "/mnt/data/traces/sweep/dflash2__full__c10__t1p0__r3"],
    "dflash2-run-d-mid": ["/mnt/data/traces/tb-dflash2-run-d-32k-mix-step1976",
                          "/mnt/data/traces/sweep/dflash2-run-d-mid__full__c10__t1p0__r2",
                          "/mnt/data/traces/sweep/dflash2-run-d-mid__full__c10__t1p0__r3"],
    "dflash2-run-d-final": ["/mnt/data/traces/tb-dflash2-run-d-32k-mix",
                            "/mnt/data/traces/sweep/dflash2-run-d-final__full__c10__t1p0__r2",
                            "/mnt/data/traces/sweep/dflash2-run-d-final__full__c10__t1p0__r3"],
}


def load(trace_dir):
    p = os.path.join(trace_dir, "calls.clean.jsonl")
    if not os.path.exists(p):
        p = os.path.join(trace_dir, "calls.jsonl")
    if not os.path.exists(p):
        return None
    out = {}
    for line in open(p):
        if not line.strip():
            continue
        r = json.loads(line)
        resp = r["response"]
        if "error" in resp:
            continue
        sd = ((resp.get("metrics") or {}).get("speculative_decoding")) or {}
        u = resp.get("usage") or {}
        out[r["replay_of"]] = {"acc": sd.get("num_accepted_draft_tokens", 0) or 0,
                               "steps": sd.get("num_spec_steps", 0) or 0,
                               "ctok": u.get("completion_tokens", 0) or 0}
    return out


def sw(rs, drop=()):
    a = s = 0
    for k, v in rs.items():
        if k in drop:
            continue
        a += v["acc"]; s += v["steps"]
    return 1 + a / s if s else float("nan")


def pr(rs, drop=(), minsteps=5):
    v = [1 + r["acc"] / r["steps"] for k, r in rs.items()
         if k not in drop and r["steps"] >= minsteps]
    return sum(v) / len(v) if v else float("nan")


def boot(A, B, n=2000, seed=0):
    """Paired bootstrap over calls on the step-weighted pool."""
    ids = sorted(set(A) & set(B))
    ka = [(A[i]["acc"], A[i]["steps"]) for i in ids]
    kb = [(B[i]["acc"], B[i]["steps"]) for i in ids]
    p = lambda k: 1 + sum(x[0] for x in k) / max(sum(x[1] for x in k), 1)
    d0 = p(ka) - p(kb)
    rnd = random.Random(seed)
    ds = []
    N = len(ids)
    for _ in range(n):
        idx = [rnd.randrange(N) for _ in range(N)]
        ds.append(p([ka[i] for i in idx]) - p([kb[i] for i in idx]))
    ds.sort()
    return d0, ds[int(0.025 * n)], ds[int(0.975 * n) - 1], N


def main():
    runs = {}
    for ck, dirs in GROUPS.items():
        for d in dirs:
            rs = load(d)
            if rs:
                runs[(ck, os.path.basename(d))] = rs

    print(f"{'checkpoint':<22}{'run':<46}{'calls':>7}{'out_tok':>11}"
          f"{'max_ctok':>10}{'accept_sw':>11}{'accept_pr':>11}{'sw_no_max':>11}")
    pooled = defaultdict(dict)
    for (ck, name), rs in sorted(runs.items()):
        mx = max(rs, key=lambda k: rs[k]["ctok"])
        print(f"{ck:<22}{name:<46}{len(rs):>7}{sum(v['ctok'] for v in rs.values()):>11,}"
              f"{rs[mx]['ctok']:>10,}{sw(rs):>11.4f}{pr(rs):>11.4f}{sw(rs, {mx}):>11.4f}")
        for k, v in rs.items():
            pooled[ck].setdefault(name, {})[k] = v

    print("\npooled over runs, per checkpoint")
    print(f"{'checkpoint':<22}{'runs':>6}{'calls':>8}{'accept_sw':>11}{'accept_pr':>11}"
          f"{'sw_drop_max_per_run':>21}")
    flat, flat_nodrop = {}, {}
    for ck, byrun in sorted(pooled.items()):
        allc, drop = {}, set()
        for name, rs in byrun.items():
            mx = max(rs, key=lambda k: rs[k]["ctok"])
            for k, v in rs.items():
                allc[f"{name}|{k}"] = v
            drop.add(f"{name}|{mx}")
        flat[ck] = allc
        flat_nodrop[ck] = {k: v for k, v in allc.items() if k not in drop}
        print(f"{ck:<22}{len(byrun):>6}{len(allc):>8}{sw(allc):>11.4f}{pr(allc):>11.4f}"
              f"{sw(allc, drop):>21.4f}")

    # Balance the run counts before pairing. Summing accepted and steps across
    # runs is scale-invariant per call, but a call seen in three runs on one side
    # and two on the other is not the same call twice -- it is more of that
    # call's weight on one side. Take the same number of runs from each.
    kmin = min(len(pooled.get("dflash2-run-d-mid", {})),
               len(pooled.get("dflash2-run-d-final", {})))
    keep = set()
    for ck in ("dflash2-run-d-mid", "dflash2-run-d-final"):
        for name in sorted(pooled.get(ck, {}))[:kmin]:
            keep.add((ck, name))
    print(f"\nmidpoint vs final, {kmin} run(s) per checkpoint "
          f"({', '.join(sorted(n for _, n in keep))})")
    for label, D in (("as measured", flat), ("longest call per run removed", flat_nodrop)):
        A = {k: v for k, v in D.get("dflash2-run-d-mid", {}).items()
             if ("dflash2-run-d-mid", k.split("|", 1)[0]) in keep}
        B = {k: v for k, v in D.get("dflash2-run-d-final", {}).items()
             if ("dflash2-run-d-final", k.split("|", 1)[0]) in keep}
        # pair on the call id only, pooling each checkpoint's runs first
        pa, pb = defaultdict(lambda: {"acc": 0, "steps": 0}), defaultdict(lambda: {"acc": 0, "steps": 0})
        for k, v in A.items():
            i = k.split("|", 1)[1]; pa[i]["acc"] += v["acc"]; pa[i]["steps"] += v["steps"]
        for k, v in B.items():
            i = k.split("|", 1)[1]; pb[i]["acc"] += v["acc"]; pb[i]["steps"] += v["steps"]
        if not pa or not pb:
            print(f"  {label}: not enough runs yet"); continue
        d, lo, hi, n = boot(dict(pa), dict(pb))
        verdict = ("mid > final" if lo > 0 else
                   "final > mid" if hi < 0 else "not separable from zero")
        print(f"  {label:<32} n={n}  delta={d:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  -> {verdict}")


if __name__ == "__main__":
    main()
