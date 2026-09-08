#!/usr/bin/env python3
"""Condense the sweep's 20 per-cell tables into the four that carry the result.

A pooled-over-buckets acceptance number is deliberately *not* the raw pool of
the four cells: the per-bucket sample sizes were chosen to fit a time budget
(160/120/60/20), so pooling them raw would report the budget, not the workload.
The `reweighted` column instead weights each bucket's acceptance by that
bucket's share of output tokens in the full 1,753-call recording, which is the
mix a deployment actually sees.

t_step is pooled across buckets without reweighting, and that is safe for the
opposite reason: docs/RESULTS-paired.md shows it is a function of the drafter and
the batch and not of the workload. The per-bucket spread is printed so the
reader can check that here too.
"""
import json, math, sys
from collections import defaultdict

R = json.load(open("/mnt/data/eval/sweep/report.json"))
BUCKETS = ["b64_128", "b128_256", "b256_1K", "bge1K"]
BL = {"b64_128": "64-128", "b128_256": "128-256", "b256_1K": "256-1K", "bge1K": ">=1K"}
# output tokens per bucket in the full replayable recording (1,753 calls)
WEIGHT = {"b64_128": 60925, "b128_256": 64821, "b256_1K": 204806, "bge1K": 179612}
ORDER = ["nospec", "dflash-official", "dflash2", "dflash2-run-d-mid",
         "dflash2-run-d-final", "dspark-community", "dspark-run-a-32k",
         "dspark-run-b-49k"]

cells = {}
for k, v in R.items():
    m = v["meta"]
    if m.get("bucket") in BUCKETS and m.get("repeat") == "r1":
        cells[(m["label"], m["bucket"], m["concurrency"])] = v

labels = [l for l in ORDER if any(k[0] == l for k in cells)]
concs = sorted({k[2] for k in cells})


def g(lab, b, c, field, where="server"):
    v = cells.get((lab, b, c))
    if not v:
        return None
    d = v.get(where) or {}
    return d.get(field) if where == "server" else v.get(field)


def reweighted(lab, c, field, where="server"):
    num = den = 0.0
    for b in BUCKETS:
        x = g(lab, b, c, field, where)
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        num += WEIGHT[b] * x; den += WEIGHT[b]
    return num / den if den else None


def cell(x, p=3):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{p}f}"


out = []
for pool_field, pool_where, title, p in (
        ("accept_len_sw", "server", "Acceptance length, step-weighted", 3),
        ("accept_len_pr", "cell", "Acceptance length, per-request", 3)):
    for c in concs:
        out.append(f"\n**{title} — greedy, concurrency {c}**\n")
        out.append("| drafter | " + " | ".join(BL[b] for b in BUCKETS)
                   + " | reweighted to the full mix |")
        out.append("|---" * (len(BUCKETS) + 2) + "|")
        for lab in labels:
            if lab == "nospec":
                continue
            row = [cell(g(lab, b, c, pool_field, pool_where), p) for b in BUCKETS]
            out.append(f"| `{lab}` | " + " | ".join(row) + " | "
                       + cell(reweighted(lab, c, pool_field, pool_where), p) + " |")

out.append("\n\n**Decode throughput over the no-spec control (x), greedy**\n")
out.append("| drafter | bucket | " + " | ".join(f"c{c}" for c in concs) + " |")
out.append("|---|---" + "|---" * len(concs) + "|")
for lab in labels:
    if lab == "nospec":
        continue
    for b in BUCKETS:
        row = []
        for c in concs:
            a = g(lab, b, c, "decode_tok_s"); d = g("nospec", b, c, "decode_tok_s")
            row.append(f"{a/d:.2f}x" if a and d else "-")
        out.append(f"| `{lab}` | {BL[b]} | " + " | ".join(row) + " |")

out.append("\n\n**`t_step` (ms), pooled over the four buckets, with the "
           "per-bucket spread**\n")
out.append("| drafter | " + " | ".join(f"c{c}" for c in concs) + " |")
out.append("|---" * (len(concs) + 1) + "|")
for lab in labels:
    row = []
    for c in concs:
        num = den = 0.0; vals = []
        for b in BUCKETS:
            v = cells.get((lab, b, c))
            if not v or not v["server"]:
                continue
            s = v["server"]
            num += s["t_step_ms"] * s["steps"]; den += s["steps"]
            vals.append(s["t_step_ms"])
        if not den:
            row.append("-"); continue
        m = num / den
        spread = (max(vals) - min(vals)) / m if len(vals) > 1 else 0.0
        row.append(f"{m:.2f} (±{spread:.1%})")
    out.append(f"| `{lab}` | " + " | ".join(row) + " |")

md = "\n".join(out)
open("/mnt/data/eval/sweep/tables.md", "w").write(md)
print(md)
