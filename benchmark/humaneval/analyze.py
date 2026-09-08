#!/usr/bin/env python3
"""Cross-drafter analysis for a set of HumanEval/MBPP traces.

metrics.py reports one drafter. This reports the comparisons between them, and
it exists because every one of these numbers is a claim in
docs/coding_benchmarks.md that would otherwise be unreproducible from the repo.

Four things, in the order they should be believed:

  identity   Under greedy decoding, lossless speculation means every drafter
             should emit the same tokens. If it does, acceptance is the only
             thing that differs and pass@1 cannot differentiate drafters -- so
             this is the check that says whether the comparison is sound at
             all, not a curiosity.

  sensitivity  Pooled acceptance over all rows, and over cleanly-terminated
             rows only. Truncated generations carry 22% of HumanEval tokens and
             43% of MBPP's, so the absolute values move; the point is whether
             the ranking and the fine-tune gaps survive.

  throughput  Output tokens over summed e2e latency at concurrency 1. The
             payoff of acceptance, and the only speed column that survives the
             latency check below.

  latency    Whether client-side TTFT is usable. With --enable-auto-tool-choice
             the tool-call parser buffers deltas, so the "first" delta can
             arrive near the end and TTFT collapses onto e2e. Server-side
             histograms (/mnt/data/prom_latency.py) are unaffected; this only
             says whether the client numbers may be quoted.

Usage:
    analyze.py /mnt/data/traces/he-<name>/gen.jsonl ...
    analyze.py --glob '/mnt/data/traces/he-mbpp-*/gen.jsonl'
"""
import argparse, glob as globmod, json, os
from collections import Counter


def name_of(path):
    return os.path.basename(os.path.dirname(path))


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def pooled(rows):
    a = s = d = 0
    for r in rows:
        sd = ((r["response"].get("metrics") or {}).get("speculative_decoding")) or {}
        a += sd.get("num_accepted_draft_tokens", 0) or 0
        s += sd.get("num_spec_steps", 0) or 0
        d += sd.get("num_draft_tokens", 0) or 0
    return (1 + a / s if s else float("nan"), a / d if d else float("nan"))


def ok(rows):
    return [r for r in rows if "error" not in r["response"]]


def fr(r):
    return (r["response"].get("choices") or [{}])[0].get("finish_reason")


def content(r):
    return ((r["response"].get("choices") or [{}])[0].get("message") or {}).get("content") or ""


def ctok(r):
    return (r["response"].get("usage") or {}).get("completion_tokens", 0) or 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="*")
    ap.add_argument("--glob", default="")
    a = ap.parse_args()
    paths = sorted(a.traces or globmod.glob(a.glob))
    if not paths:
        raise SystemExit("no traces; pass paths or --glob")
    data = {name_of(p): ok(load(p)) for p in paths}

    print("== acceptance, all rows vs cleanly-terminated only ==")
    print(f"{'drafter':38s} {'n':>4} {'all':>8} {'stop-only':>10} {'shift':>7} "
          f"{'rate':>7} {'tok/s':>7} {'trunc':>6}")
    rank = {}
    for n, rows in data.items():
        acc, rate = pooled(rows)
        stop = [r for r in rows if fr(r) == "stop"]
        sacc, _ = pooled(stop)
        lat = sum(r.get("latency", 0) or 0 for r in rows)
        tps = sum(ctok(r) for r in rows) / lat if lat else float("nan")
        rank[n] = (acc, sacc)
        print(f"{n:38s} {len(rows):>4} {acc:8.4f} {sacc:10.4f} {sacc-acc:+7.4f} "
              f"{rate:7.4f} {tps:7.1f} {len(rows)-len(stop):>6}")
    print("\n  ranking all      :", " > ".join(sorted(rank, key=lambda k: -rank[k][0])))
    print("  ranking stop-only:", " > ".join(sorted(rank, key=lambda k: -rank[k][1])))

    print("\n== output identity (greedy speculation should be lossless) ==")
    ref = list(data)[0]
    by = {n: {r["task_id"]: content(r) for r in rows} for n, rows in data.items()}
    for n in list(data)[1:]:
        shared = set(by[ref]) & set(by[n])
        same = sum(1 for t in shared if by[ref][t] == by[n][t])
        print(f"  {n:38s} identical to {ref}: {same}/{len(shared)}")

    print("\n== truncation vs delivered content ==")
    for n, rows in data.items():
        c = Counter((fr(r), "content" if content(r) else "EMPTY") for r in rows)
        tt = sum(ctok(r) for r in rows if fr(r) == "length")
        at = sum(ctok(r) for r in rows) or 1
        print(f"  {n:38s} {dict(c)}  truncated tokens {tt/at:.1%}  "
              f"max_ctok {max((ctok(r) for r in rows), default=0)}")

    print("\n== client-side latency validity ==")
    print("  (TTFT near e2e means the tool-call parser buffered; use server-side)")
    for n, rows in data.items():
        pairs = [(r["ttft"], r["latency"]) for r in rows
                 if r.get("ttft") is not None and (r.get("latency") or 0) > 0]
        if not pairs:
            print(f"  {n:38s} no ttft recorded")
            continue
        ratios = sorted(t / e for t, e in pairs)
        med = ratios[len(ratios) // 2]
        near = sum(1 for x in ratios if x > 0.95) / len(ratios)
        print(f"  {n:38s} median ttft/e2e {med:.3f}  within 5% of e2e: {near:.1%}"
              f"   -> client TTFT {'UNUSABLE' if med > 0.5 else 'ok'}")


if __name__ == "__main__":
    main()
