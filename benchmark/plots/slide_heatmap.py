#!/usr/bin/env python3
"""Acceptance heatmaps: context length x reasoning strength.

Our final DFlash2 checkpoint against the native DFlash2 baseline, with the
same-drafter repeat control shown on the same colour scale as a noise floor.

Every number is HARD-CODED below, so this file runs anywhere with no access to
the GPU node, the repo's results/ tree, or any other script.

PROVENANCE -- READ THIS BEFORE QUOTING
--------------------------------------
Source: results/terminal_bench/buckets.json, the FULL 1,753-call frozen
Terminal-Bench replay.  That campaign is **temperature 1.0, top_k 64,
concurrency 10** -- NOT the greedy concurrency-1 sweep that every other figure
in benchmark/plots/ uses.  The two are not comparable and must not share a
table.

There is no updated (greedy, concurrency-1) version of this breakdown: that
sweep was designed around OUTPUT-LENGTH buckets, with per-bucket manifests of
160/120/60/20 calls, and reasoning strength was never a design variable in it.
Producing a greedy context x strength grid would need a new run.

Cells are step-weighted acceptance, 1 + sum(accepted) / sum(steps), pooled over
the calls in that cell.  `dflash2-run-d-32k-mix` is our FINAL checkpoint,
step 3,956.

WHY THE CONTROL PANEL IS NOT OPTIONAL
-------------------------------------
`dflash2-repeat2` is the native DFlash2 drafter replayed a SECOND time on the
same 1,753 calls.  Its cell-level deltas against the first pass are the noise
floor, and they are large: `<2K|medium` moves +3.372 between two runs of the
identical drafter, on n=19 calls.  Several fine-tune deltas are smaller than the
control's delta in the same cell.  Show the two bottom panels together or not at
all -- the top-right panel alone invites reading structure that the control says
is not there.

Cell counts range from 19 to 248 calls.  The `64K+` row exists only at xhigh.

USAGE
-----
  uv run --with matplotlib python benchmark/plots/slide_heatmap.py
  uv run python benchmark/plots/slide_heatmap.py --table-only
"""
from __future__ import annotations

import argparse
import os
import sys

CTX = ["<2K", "2-8K", "8-16K", "16-32K", "32-64K", "64K+"]
SEG = ["low", "medium", "high", "xhigh"]

# (step-weighted acceptance, n calls); None where the cell is empty.
CELLS = {
    "dflash2": {
        "<2K":    [(5.502, 26), (3.248, 19), (3.261, 26), (5.470, 26)],
        "2-8K":   [(4.945, 56), (4.102, 63), (4.345, 74), (4.051, 86)],
        "8-16K":  [(5.162, 44), (3.784, 120), (3.618, 87), (3.659, 76)],
        "16-32K": [(4.518, 100), (4.111, 108), (3.881, 102), (3.726, 204)],
        "32-64K": [(4.202, 119), (5.310, 56), (5.469, 41), (3.927, 248)],
        "64K+":   [None, None, None, (4.042, 56)],
    },
    "dflash2-run-d-32k-mix": {
        "<2K":    [(5.918, 24), (3.330, 19), (3.568, 26), (6.050, 26)],
        "2-8K":   [(4.956, 55), (3.538, 63), (4.292, 73), (3.690, 82)],
        "8-16K":  [(5.164, 45), (3.433, 120), (3.758, 84), (3.516, 76)],
        "16-32K": [(4.422, 98), (4.045, 109), (3.680, 98), (3.742, 204)],
        "32-64K": [(4.320, 118), (5.471, 49), (5.071, 41), (3.997, 245)],
        "64K+":   [None, None, None, (3.917, 54)],
    },
    "dflash2-repeat2": {
        "<2K":    [(5.634, 26), (6.620, 19), (3.303, 26), (6.143, 25)],
        "2-8K":   [(5.169, 57), (3.916, 63), (4.034, 74), (3.762, 86)],
        "8-16K":  [(5.038, 46), (3.818, 121), (3.584, 89), (3.568, 77)],
        "16-32K": [(4.287, 101), (4.019, 108), (3.799, 101), (3.751, 202)],
        "32-64K": [(3.919, 118), (5.457, 55), (4.791, 41), (4.078, 247)],
        "64K+":   [None, None, None, (4.172, 56)],
    },
}

TITLES = {
    "dflash2": "DFlash2 baseline (native)",
    "dflash2-run-d-32k-mix": "Ours — DFlash2 fine-tune, step 3,956",
    "dflash2-repeat2": "DFlash2 replayed again (control)",
}


def grid(key, field="acc"):
    out = []
    for c in CTX:
        row = []
        for i in range(len(SEG)):
            v = CELLS[key][c][i]
            row.append(None if v is None else (v[0] if field == "acc" else v[1]))
        out.append(row)
    return out


def delta(a, b):
    ga, gb = grid(a), grid(b)
    return [[None if (x is None or y is None) else x - y
             for x, y in zip(ra, rb)] for ra, rb in zip(ga, gb)]


def make_chart(out_dir, dpi, ext):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    base, fine, ctrl = "dflash2", "dflash2-run-d-32k-mix", "dflash2-repeat2"
    d_fine, d_ctrl = delta(fine, base), delta(ctrl, base)

    absvals = [v for g in (grid(base), grid(fine)) for r in g for v in r if v is not None]
    vmin, vmax = min(absvals), max(absvals)
    # Clip the diverging scale.  The control's <2K|medium cell is +3.372 and
    # would otherwise saturate the map and wash out every other cell; the max
    # of all remaining cells is 0.678.  The outlier still prints its own value,
    # so clipping hides nothing -- it just stops one cell owning the colourbar.
    all_d = sorted(abs(v) for g in (d_fine, d_ctrl) for r in g
                   for v in r if v is not None)
    DELTA_CLIP = 0.70
    dmax = DELTA_CLIP
    n_off = sum(1 for v in all_d if v > DELTA_CLIP)

    panels = [
        (grid(base), TITLES[base], "Blues", vmin, vmax, "{:.2f}", True),
        (grid(fine), TITLES[fine], "Blues", vmin, vmax, "{:.2f}", True),
        (d_fine, "Fine-tune − baseline", "RdBu_r", -dmax, dmax, "{:+.2f}", False),
        (d_ctrl, "CONTROL: same drafter, 2nd replay − baseline\n"
                 "(this is the noise floor)", "RdBu_r", -dmax, dmax, "{:+.2f}", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.2))
    counts = grid(base, "n")
    for ax, (g, title, cmap, lo, hi, fmt, show_n) in zip(axes.ravel(), panels):
        arr = np.array([[np.nan if v is None else v for v in r] for r in g], dtype=float)
        im = ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
        for i in range(len(CTX)):
            for j in range(len(SEG)):
                if g[i][j] is None:
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=11, color="#999999")
                    continue
                rel = (g[i][j] - lo) / (hi - lo)
                col = "white" if (rel > 0.72 or rel < 0.28) else "#111111"
                txt = fmt.format(g[i][j])
                if show_n:
                    txt += f"\nn={counts[i][j]}"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=9.5, fontweight="bold", color=col)
        ax.set_xticks(range(len(SEG)))
        ax.set_xticklabels(SEG, fontsize=10)
        ax.set_yticks(range(len(CTX)))
        ax.set_yticklabels(CTX, fontsize=10)
        ax.set_xlabel("reasoning strength", fontsize=10)
        ax.set_ylabel("prompt context length", fontsize=10)
        ax.set_title(title, fontsize=11, pad=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle("Acceptance length by context × reasoning strength — "
                 "Terminal-Bench full replay (1,753 calls)", fontsize=13, y=0.985)
    fig.text(0.5, 0.010,
             "Step-weighted acceptance per cell. Temperature 1.0, top_k 64, "
             "concurrency 10 — NOT the greedy concurrency-1 regime\n"
             "used elsewhere in these figures; do not put them in one table.\n"
             f"Bottom panels share a colour scale, clipped at ±{DELTA_CLIP:.2f} "
             f"({n_off} cell off-scale; its value is still printed).\n"
             "Several fine-tune deltas are smaller than the control's delta in "
             "the same cell — <2K|medium moves +3.37 between two runs of the "
             "identical drafter (n=19).",
             ha="center", va="bottom", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.095, 1, 0.965))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"slide_heatmap_dflash2_final.{ext}")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def table():
    out = []
    counts = grid("dflash2", "n")
    for key in ("dflash2", "dflash2-run-d-32k-mix", "dflash2-repeat2"):
        out.append(f"**{TITLES[key]}**\n")
        out.append("| ctx | " + " | ".join(SEG) + " |")
        out.append("|---" * (len(SEG) + 1) + "|")
        g = grid(key)
        for i, c in enumerate(CTX):
            out.append(f"| {c} | " + " | ".join(
                "—" if g[i][j] is None else f"{g[i][j]:.3f}"
                for j in range(len(SEG))) + " |")
        out.append("")
    for name, d in (("Fine-tune − baseline", delta("dflash2-run-d-32k-mix", "dflash2")),
                    ("CONTROL: 2nd replay − baseline", delta("dflash2-repeat2", "dflash2"))):
        out.append(f"**{name}**\n")
        out.append("| ctx | " + " | ".join(SEG) + " |")
        out.append("|---" * (len(SEG) + 1) + "|")
        for i, c in enumerate(CTX):
            out.append(f"| {c} | " + " | ".join(
                "—" if d[i][j] is None else f"{d[i][j]:+.3f}"
                for j in range(len(SEG))) + " |")
        out.append("")
    out.append("**Cell counts (calls)**\n")
    out.append("| ctx | " + " | ".join(SEG) + " |")
    out.append("|---" * (len(SEG) + 1) + "|")
    for i, c in enumerate(CTX):
        out.append(f"| {c} | " + " | ".join(
            "—" if counts[i][j] is None else str(counts[i][j])
            for j in range(len(SEG))) + " |")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="benchmark/plots/out")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--ext", default="png", choices=["png", "pdf", "svg"])
    p.add_argument("--table-only", action="store_true")
    a = p.parse_args(argv)

    print(table())
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
    print("wrote", make_chart(a.out_dir, a.dpi, a.ext))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
