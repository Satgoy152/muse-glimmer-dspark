#!/usr/bin/env python3
"""Per-strength table for a replay trace, plus the latency columns.

Acceptance length uses metrics.py's pooling exactly -- 1 + sum(accepted)/sum(steps)
over the per-request speculative_decoding blocks -- so the accept_len/draft_rate
columns are directly comparable to the published baseline. The latency columns
are new: TTFT is only populated when the replay ran with --stream.
"""
import argparse, json, statistics as st
from collections import defaultdict

SEGMENTS = ["low", "medium", "high", "xhigh"]


def pct(v, q):
    if not v:
        return float("nan")
    v = sorted(v)
    k = min(int(round(q * (len(v) - 1))), len(v) - 1)
    return v[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--name", default="")
    a = ap.parse_args()

    agg = defaultdict(lambda: defaultdict(float))
    lat = defaultdict(lambda: defaultdict(list))
    errored = unknown = 0
    for line in open(a.trace):
        if not line.strip():
            continue
        r = json.loads(line)
        resp = r["response"]
        if "error" in resp:
            errored += 1
            continue
        s = (r["request"].get("chat_template_kwargs") or {}).get("reasoning_strength")
        if s not in SEGMENTS:
            unknown += 1
            continue
        d = agg[s]
        d["calls"] += 1
        ct = (resp.get("usage") or {}).get("completion_tokens", 0) or 0
        d["out_tok"] += ct
        d["latency"] += r.get("latency", 0) or 0
        sd = ((resp.get("metrics") or {}).get("speculative_decoding")) or {}
        d["steps"] += sd.get("num_spec_steps", 0) or 0
        d["accepted"] += sd.get("num_accepted_draft_tokens", 0) or 0
        d["drafted"] += sd.get("num_draft_tokens", 0) or 0
        if r.get("ttft") is not None:
            lat[s]["ttft"].append(r["ttft"])
        if r.get("tpot") is not None:
            lat[s]["tpot"].append(r["tpot"])
        lat[s]["e2e"].append(r.get("latency", 0) or 0)

    print(f"{a.trace}: excluded {errored} errored calls, {unknown} without reasoning_strength\n")
    hdr = (f"{'strength':<9} {'calls':>6} {'out_tok':>9} {'accept_len':>11} "
           f"{'draft_rate':>11} {'tok/s':>8} {'ttft_p50':>9} {'ttft_p95':>9} "
           f"{'tpot_p50':>9} {'e2e_p50':>8}")
    print(hdr)
    rows, pooled, plat = [], defaultdict(float), defaultdict(list)
    for s in SEGMENTS:
        if s in agg:
            rows.append((s, agg[s], lat[s]))
            for k, v in agg[s].items():
                pooled[k] += v
            for k, v in lat[s].items():
                plat[k].extend(v)
    rows.append(("pooled", pooled, plat))

    out = {}
    for s, d, L in rows:
        acc = 1 + d["accepted"] / d["steps"] if d["steps"] else float("nan")
        rate = d["accepted"] / d["drafted"] if d["drafted"] else float("nan")
        tps = d["out_tok"] / d["latency"] if d["latency"] else float("nan")
        t50, t95 = pct(L.get("ttft", []), .5), pct(L.get("ttft", []), .95)
        p50 = pct(L.get("tpot", []), .5)
        e50 = pct(L.get("e2e", []), .5)
        print(f"{s:<9} {int(d['calls']):>6} {int(d['out_tok']):>9,} {acc:>11.3f} "
              f"{rate:>11.3f} {tps:>8.1f} {t50:>9.3f} {t95:>9.3f} {p50:>9.4f} {e50:>8.2f}")
        out[s] = {"calls": int(d["calls"]), "out_tok": int(d["out_tok"]),
                  "accept_len": round(acc, 4), "draft_rate": round(rate, 4),
                  "tok_s": round(tps, 2), "ttft_p50": t50, "ttft_p95": t95,
                  "tpot_p50": p50, "e2e_p50": e50}
    out["_meta"] = {"name": a.name, "trace": a.trace,
                    "errored": errored, "no_strength": unknown}
    if a.json_out:
        json.dump(out, open(a.json_out, "w"), indent=1)
        print(f"\nwrote {a.json_out}")


if __name__ == "__main__":
    main()
