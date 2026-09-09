#!/usr/bin/env python3
"""Acceptance heatmaps: context length x reasoning strength, 64-256 output slice.

Our final DFlash2 checkpoint against the native DFlash2 baseline, with the
same-drafter repeat control on the same colour scale as a noise floor.

Every number is HARD-CODED below, so this file runs anywhere with no access to
the GPU node, the repo's results/ tree, or any other script.

PROVENANCE -- READ THIS BEFORE QUOTING
--------------------------------------
Rebuilt from results/node-artifacts/analysis/{an_class,an_percall}.json, the
per-call records of the FULL 1,753-call frozen Terminal-Bench replay.

That campaign is **temperature 1.0, top_k 64, concurrency 10** -- NOT the greedy
concurrency-1 sweep every other figure in benchmark/plots/ uses.  The two are
not comparable and must not share a table.  No greedy version of this breakdown
exists: that sweep was built around OUTPUT-LENGTH buckets with 160/120/60/20-call
manifests, and reasoning strength was never a design variable in it.

SLICE.  Only calls whose completion length in the ORIGINAL frozen recording is
64 <= ctok < 256 -- 1,058 of the 1,808 classified calls.  Bucket membership is
therefore fixed across drafters and does not select on the replayed outcome.
Context bucket and strength also come from the original recording.

TWO VALUES, AND THEY DISAGREE
-----------------------------
  per_token  (default) 1 + sum(accepted) / sum(steps) within the cell.
             Token-weighted, so long calls dominate.  This is the acceptance
             that maps to throughput.
  per_turn   mean over the cell's calls of that call's own acceptance, every
             call weighted equally.  Runs ~1.3-2.0 higher here.

CELL COUNTS ARE NOT PRINTED ON THE MAP -- they crowd it -- but they matter and
are in the --table-only output.  They range from 8 to 167 calls.  The `64K+`
row exists only at xhigh.

WHY THE CONTROL PANEL IS NOT OPTIONAL
-------------------------------------
`dflash2-repeat2` is the native DFlash2 drafter replayed a SECOND time on the
same calls, so its cell deltas against the first pass are pure noise.  In this
slice they reach 1.04 (`32-64K|medium`, n=19) and 0.85 (`32-64K|high`, n=17).
Several fine-tune deltas are smaller than the control's delta in the same cell.
Show the two bottom panels together or not at all.

USAGE
-----
  uv run --with matplotlib python benchmark/plots/slide_heatmap.py
  uv run --with matplotlib python benchmark/plots/slide_heatmap.py --value per_turn
  uv run python benchmark/plots/slide_heatmap.py --table-only
"""
from __future__ import annotations

import argparse
import os
import sys

CTX = ["<2K", "2-8K", "8-16K", "16-32K", "32-64K", "64K+"]
SEG = ["low", "medium", "high", "xhigh"]
N_CALLS = 1058          # calls in the 64-256 slice, of 1,808 classified
DELTA_CLIP = 0.85       # ~p90 of |delta| across both bottom panels

# (per_token acceptance, per_turn acceptance, n calls); None where empty.
CELLS = {
    "dflash2": {
        "<2K": [(6.3296, 7.0911, 16), (6.438, 7.0164, 13), (3.251, 6.3001, 8), (4.062, 5.0529, 15)],
        "2-8K": [(5.1392, 6.1082, 37), (4.1545, 5.6335, 44), (4.4537, 6.3504, 30), (4.0095, 5.708, 53)],
        "8-16K": [(5.139, 6.0894, 39), (4.782, 6.0206, 84), (3.6834, 5.4911, 44), (3.7415, 5.0113, 52)],
        "16-32K": [(4.7134, 5.8076, 74), (4.234, 5.9797, 49), (3.8996, 5.2515, 38), (4.2417, 5.8204, 134)],
        "32-64K": [(4.3256, 5.6879, 83), (4.8416, 6.3488, 19), (5.1562, 5.9496, 17), (4.232, 6.0401, 167)],
        "64K+": [None, None, None, (4.2459, 5.1343, 32)],
    },
    "dflash2-run-d-32k-mix": {
        "<2K": [(6.5902, 7.643, 14), (3.4026, 7.3447, 13), (5.4977, 7.2857, 8), (4.8133, 6.4001, 15)],
        "2-8K": [(5.4861, 6.4694, 35), (4.1249, 6.5193, 44), (4.8941, 7.4984, 29), (4.049, 6.5717, 52)],
        "8-16K": [(5.4623, 6.8351, 40), (4.7085, 6.8665, 85), (3.931, 6.4414, 43), (3.8236, 5.8335, 52)],
        "16-32K": [(4.4505, 7.093, 73), (4.4377, 6.6605, 48), (3.6199, 6.1069, 36), (3.8167, 6.5579, 134)],
        "32-64K": [(4.4464, 6.6286, 83), (5.78, 7.6392, 18), (4.7081, 6.1685, 17), (4.3461, 6.8452, 164)],
        "64K+": [None, None, None, (4.7133, 5.928, 30)],
    },
    "dflash2-repeat2": {
        "<2K": [(6.5366, 8.034, 16), (6.4, 6.5882, 13), (3.6062, 5.1752, 8), (4.7363, 5.8429, 14)],
        "2-8K": [(5.6, 6.1509, 37), (4.0874, 5.829, 44), (3.9833, 5.8793, 30), (3.8045, 5.4534, 53)],
        "8-16K": [(4.7291, 5.572, 41), (4.6859, 5.7478, 85), (3.7011, 5.325, 46), (3.4907, 5.0886, 53)],
        "16-32K": [(4.3606, 5.8557, 75), (4.0193, 5.6823, 49), (3.4612, 5.0698, 37), (4.3839, 5.9837, 133)],
        "32-64K": [(3.9892, 5.4021, 82), (5.8794, 6.8443, 19), (4.3037, 5.3778, 17), (4.2121, 5.947, 168)],
        "64K+": [None, None, None, (4.5806, 5.3692, 32)],
    },
}
TITLES = {
    "dflash2": "DFlash2 baseline (native)",
    "dflash2-run-d-32k-mix": "Ours — DFlash2 fine-tune, step 3,956",
    "dflash2-repeat2": "DFlash2 replayed again (control)",
}
IDX = {"per_token": 0, "per_turn": 1}


def grid(key, value):
    i = IDX[value]
    return [[None if c is None else c[i] for c in CELLS[key][ctx]] for ctx in CTX]


def counts():
    return [[None if c is None else c[2] for c in CELLS["dflash2"][ctx]] for ctx in CTX]


def delta(a, b, value):
    ga, gb = grid(a, value), grid(b, value)
    return [[None if (x is None or y is None) else x - y
             for x, y in zip(ra, rb)] for ra, rb in zip(ga, gb)]


def make_chart(value, out_dir, dpi, ext):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    base, fine, ctrl = "dflash2", "dflash2-run-d-32k-mix", "dflash2-repeat2"
    d_fine, d_ctrl = delta(fine, base, value), delta(ctrl, base, value)
    absv = [v for g in (grid(base, value), grid(fine, value))
            for r in g for v in r if v is not None]
    vmin, vmax = min(absv), max(absv)
    n_off = sum(1 for g in (d_fine, d_ctrl) for r in g
                for v in r if v is not None and abs(v) > DELTA_CLIP)

    panels = [
        (grid(base, value), TITLES[base], "Blues", vmin, vmax, "{:.2f}"),
        (grid(fine, value), TITLES[fine], "Blues", vmin, vmax, "{:.2f}"),
        (d_fine, "Fine-tune − baseline", "RdBu_r", -DELTA_CLIP, DELTA_CLIP, "{:+.2f}"),
        (d_ctrl, "CONTROL: same drafter, 2nd replay − baseline\n(the noise floor)",
         "RdBu_r", -DELTA_CLIP, DELTA_CLIP, "{:+.2f}"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.0))
    for ax, (g, title, cmap, lo, hi, fmt) in zip(axes.ravel(), panels):
        arr = np.array([[np.nan if v is None else v for v in r] for r in g], float)
        im = ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
        for i in range(len(CTX)):
            for j in range(len(SEG)):
                if g[i][j] is None:
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=11, color="#999999")
                    continue
                rel = (min(max(g[i][j], lo), hi) - lo) / (hi - lo)
                col = "white" if (rel > 0.74 or rel < 0.26) else "#111111"
                ax.text(j, i, fmt.format(g[i][j]), ha="center", va="center",
                        fontsize=11.5, fontweight="bold", color=col)
        ax.set_xticks(range(len(SEG)))
        ax.set_xticklabels(SEG, fontsize=10)
        ax.set_yticks(range(len(CTX)))
        ax.set_yticklabels(CTX, fontsize=10)
        ax.set_xlabel("reasoning strength", fontsize=10)
        ax.set_ylabel("prompt context length", fontsize=10)
        ax.set_title(title, fontsize=11, pad=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    vname = {"per_token": "per-token (step-weighted)", "per_turn": "per-turn"}[value]
    fig.suptitle(f"Acceptance by context × reasoning strength — {vname}, "
                 f"64–256 output slice ({N_CALLS} calls)", fontsize=13, y=0.985)
    fig.text(0.5, 0.010,
             "Slice fixed on the ORIGINAL recording's completion length, so it "
             "does not select on the replayed outcome.\n"
             "Temperature 1.0, top_k 64, concurrency 10 — NOT the greedy "
             "concurrency-1 regime used elsewhere; do not put them in one table.\n"
             f"Bottom panels share a scale clipped at ±{DELTA_CLIP:.2f} "
             f"({n_off} cells off-scale; values still printed). Cell n ranges "
             "8–167 — see --table-only.\n"
             "Several fine-tune deltas are smaller than the control's delta in "
             "the same cell.",
             ha="center", va="bottom", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.105, 1, 0.965))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"slide_heatmap_dflash2_final_{value}.{ext}")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def table(value):
    out = []
    for key in ("dflash2", "dflash2-run-d-32k-mix", "dflash2-repeat2"):
        g = grid(key, value)
        out.append(f"**{TITLES[key]}** ({value})\n")
        out.append("| ctx | " + " | ".join(SEG) + " |")
        out.append("|---" * (len(SEG) + 1) + "|")
        for i, c in enumerate(CTX):
            out.append(f"| {c} | " + " | ".join(
                "—" if g[i][j] is None else f"{g[i][j]:.3f}"
                for j in range(len(SEG))) + " |")
        out.append("")
    for name, d in (("Fine-tune − baseline",
                     delta("dflash2-run-d-32k-mix", "dflash2", value)),
                    ("CONTROL: 2nd replay − baseline",
                     delta("dflash2-repeat2", "dflash2", value))):
        out.append(f"**{name}** ({value})\n")
        out.append("| ctx | " + " | ".join(SEG) + " |")
        out.append("|---" * (len(SEG) + 1) + "|")
        for i, c in enumerate(CTX):
            out.append(f"| {c} | " + " | ".join(
                "—" if d[i][j] is None else f"{d[i][j]:+.3f}"
                for j in range(len(SEG))) + " |")
        out.append("")
    cn = counts()
    out.append("**Cell counts (calls)** — not printed on the map\n")
    out.append("| ctx | " + " | ".join(SEG) + " |")
    out.append("|---" * (len(SEG) + 1) + "|")
    for i, c in enumerate(CTX):
        out.append(f"| {c} | " + " | ".join(
            "—" if cn[i][j] is None else str(cn[i][j]) for j in range(len(SEG))) + " |")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--value", choices=["per_token", "per_turn"], default="per_token",
                   help="per_token (default) = step-weighted within the cell; "
                        "per_turn = equal weight per call")
    p.add_argument("--out-dir", default="benchmark/plots/out")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--ext", default="png", choices=["png", "pdf", "svg"])
    p.add_argument("--table-only", action="store_true")
    a = p.parse_args(argv)

    print(table(a.value))
    print()
    if a.table_only:
        return 0
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib is not installed. Either:\n"
              "  uv run --with matplotlib python benchmark/plots/slide_heatmap.py\n"
              "Or re-run with --table-only.", file=sys.stderr)
        return 1
    print("wrote", make_chart(a.value, a.out_dir, a.dpi, a.ext))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
