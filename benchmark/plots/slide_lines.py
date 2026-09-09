#!/usr/bin/env python3
"""Line charts for the deck: how each drafter's speed holds up as concurrency rises.

Companion to slide_bars.py.  Every number is HARD-CODED below, so this file runs
anywhere with no access to the GPU node, the repo's results/ tree, or any other
script.

PROVENANCE
----------
Source: the Terminal-Bench bucket x concurrency sweep, /mnt/data/eval/sweep on
the Nebius node (report.json + percall.json), summarised in
docs/RESULTS-eval-sweep.md.

  * greedy decoding, NUM_SPEC_TOKENS = 15, one vLLM server per drafter
  * concurrencies 1 / 2 / 8 / 32, each a separately timed replay
  * output bucket `64-256` = the sweep's b64_128 (160 calls) + b128_256 (120),
    280 calls per drafter per concurrency
  * bucket membership comes from the ORIGINAL frozen recording's
    completion_tokens, so nothing selects on the replayed outcome

WHAT IS PLOTTED
---------------
  tps        pooled decode throughput, sum(gen) / sum(steps * t_step).
             This is the same pooled quantity as slide_bars.py --metric tokens,
             i.e. the token-weighted harmonic mean of the per-call rates.  See
             that file's docstring for why the arithmetic mean over turns is
             the wrong average for a rate.
  accept_sw  step-weighted acceptance, 1 + sum(accepted) / sum(steps).
  vs_dflash  pooled tok/s divided by dflash-official's, at the same concurrency.
  vs_nospec  pooled tok/s divided by the no-speculation control's.

ERROR BARS
----------
95% percentile bootstrap CIs, 4,000 resamples, seed 20260830, resampling calls.

For `tps`, `accept_sw` and `vs_dflash` the bootstrap is PAIRED: every drafter
replayed the byte-identical 280 calls (verified -- the per-call keys intersect
280/280), so one resample of call indices is applied to every drafter and to
the ratio's numerator and denominator together.  That is why the vs_dflash
intervals are much tighter than dividing two independent intervals would give.

`vs_nospec` has NO interval.  The no-speculation control ran as its own job with
its own call keys and reports steps == 0 per request, so it cannot be paired or
resampled; its throughput comes from the server counters and is treated as a
constant.  Read vs_nospec as a point estimate only.

CAVEATS TO KEEP WITH ANY SLIDE BUILT FROM THIS
----------------------------------------------
* The `64-256` bucket is 60.35% of Terminal-Bench calls but only 24.19% of its
  decoded tokens.  These are step-weighted/pooled numbers within that bucket,
  not over the full workload.
* The c32 column is honest for this bucket but the sweep's `>=1K` bucket could
  not reach concurrency 32 (20 calls over 20 trajectories, one worker per
  trajectory).  That caveat does not apply to the two buckets used here, which
  hold 160 and 120 calls.
* Acceptance is nearly flat in concurrency by design -- it is a property of the
  drafter and the prompt.  What moves is t_step.  A line chart of accept_sw is
  therefore expected to be four roughly horizontal lines; that is a result, not
  a broken plot.

USAGE
-----
  uv run --with matplotlib python benchmark/plots/slide_lines.py --family dspark
  uv run --with matplotlib python benchmark/plots/slide_lines.py --family dspark \
      --metric vs_dflash
  uv run python benchmark/plots/slide_lines.py --family all --table-only
"""
from __future__ import annotations

import argparse
import os
import sys

CONCURRENCIES = [1, 2, 8, 32]
N_CALLS = 280
BOOTSTRAP_RESAMPLES = 4000   # seed 20260830, percentile 95% CI, paired over calls

LABELS = {
    "dflash-official":     "DFlash (official)",
    "dflash2":             "DFlash2",
    "dspark-community":    "DSpark (community)",
    "dspark-run-a-32k":    "Ours DSpark 32K",
    "dspark-run-b-49k":    "Ours DSpark 49K",
    "dflash2-run-d-mid":   "Ours DFlash2 mid",
    "dflash2-run-d-final": "Ours DFlash2 final",
}

COLORS = {
    "dflash-official":     "#9AA3AE",
    "dflash2":             "#6B7580",
    "dspark-community":    "#BBD1EC",
    "dspark-run-a-32k":    "#4F86C6",
    "dspark-run-b-49k":    "#2C5F9E",
    "dflash2-run-d-mid":   "#E8A87C",
    "dflash2-run-d-final": "#D2691E",
}

MARKERS = {
    "dflash-official":     "o",
    "dflash2":             "s",
    "dspark-community":    "^",
    "dspark-run-a-32k":    "D",
    "dspark-run-b-49k":    "v",
    "dflash2-run-d-mid":   "D",
    "dflash2-run-d-final": "v",
}

FAMILIES = {
    "dspark": ["dflash-official", "dflash2",
               "dspark-community", "dspark-run-a-32k", "dspark-run-b-49k"],
    "dflash2": ["dflash-official", "dspark-community",
                "dflash2", "dflash2-run-d-mid", "dflash2-run-d-final"],
    "all": ["dflash-official", "dflash2",
            "dflash2-run-d-mid", "dflash2-run-d-final",
            "dspark-community", "dspark-run-a-32k", "dspark-run-b-49k"],
}

DEFAULT_REF = {"dspark": "dspark-community", "dflash2": "dflash2", "all": "dflash2"}

CONC_DATA = {
    "dflash-official": {
        1: dict(accept_sw=4.8242, accept_sw_ci=(4.5987, 5.1047), tps=251.862, tps_ci=(240.2769, 266.4223), vs_dflash=1.0, vs_dflash_ci=(1.0, 1.0), vs_nospec=4.0394, nospec_tps=62.352),
        2: dict(accept_sw=4.7254, accept_sw_ci=(4.4817, 5.0311), tps=228.346, tps_ci=(216.8097, 242.776), vs_dflash=1.0, vs_dflash_ci=(1.0, 1.0), vs_nospec=3.6886, nospec_tps=61.905),
        8: dict(accept_sw=4.8518, accept_sw_ci=(4.6134, 5.1471), tps=199.713, tps_ci=(190.1342, 211.5506), vs_dflash=1.0, vs_dflash_ci=(1.0, 1.0), vs_nospec=3.3379, nospec_tps=59.832),
        32: dict(accept_sw=4.8497, accept_sw_ci=(4.6056, 5.1427), tps=96.678, tps_ci=(91.7957, 102.3178), vs_dflash=1.0, vs_dflash_ci=(1.0, 1.0), vs_nospec=1.7756, nospec_tps=54.447),
    },
    "dflash2": {
        1: dict(accept_sw=4.839, accept_sw_ci=(4.6158, 5.1077), tps=249.648, tps_ci=(238.0951, 263.5608), vs_dflash=0.9912, vs_dflash_ci=(0.9533, 1.029), vs_nospec=4.0039, nospec_tps=62.352),
        2: dict(accept_sw=4.8703, accept_sw_ci=(4.6419, 5.1421), tps=243.486, tps_ci=(232.1975, 257.0706), vs_dflash=1.0663, vs_dflash_ci=(1.0337, 1.0991), vs_nospec=3.9332, nospec_tps=61.905),
        8: dict(accept_sw=4.9419, accept_sw_ci=(4.7162, 5.2162), tps=202.617, tps_ci=(193.5587, 213.738), vs_dflash=1.0145, vs_dflash_ci=(0.9884, 1.0379), vs_nospec=3.3864, nospec_tps=59.832),
        32: dict(accept_sw=4.9058, accept_sw_ci=(4.6785, 5.1727), tps=97.618, tps_ci=(93.2197, 102.7684), vs_dflash=1.0097, vs_dflash_ci=(0.9795, 1.0367), vs_nospec=1.7929, nospec_tps=54.447),
    },
    "dflash2-run-d-mid": {
        1: dict(accept_sw=5.0664, accept_sw_ci=(4.7901, 5.4174), tps=259.707, tps_ci=(245.6374, 277.5883), vs_dflash=1.0311, vs_dflash_ci=(0.9996, 1.0668), vs_nospec=4.1652, nospec_tps=62.352),
        2: dict(accept_sw=5.0282, accept_sw_ci=(4.7063, 5.4115), tps=249.884, tps_ci=(234.0727, 268.7261), vs_dflash=1.0943, vs_dflash_ci=(1.0554, 1.1382), vs_nospec=4.0366, nospec_tps=61.905),
        8: dict(accept_sw=5.0657, accept_sw_ci=(4.7828, 5.435), tps=205.054, tps_ci=(193.9381, 219.6134), vs_dflash=1.0267, vs_dflash_ci=(0.9917, 1.0619), vs_nospec=3.4272, nospec_tps=59.832),
        32: dict(accept_sw=5.1211, accept_sw_ci=(4.8346, 5.4736), tps=100.934, tps_ci=(95.4615, 107.5263), vs_dflash=1.044, vs_dflash_ci=(1.0146, 1.0718), vs_nospec=1.8538, nospec_tps=54.447),
    },
    "dflash2-run-d-final": {
        1: dict(accept_sw=5.1896, accept_sw_ci=(4.8859, 5.5636), tps=265.85, tps_ci=(250.4393, 284.7404), vs_dflash=1.0555, vs_dflash_ci=(1.0106, 1.101), vs_nospec=4.2637, nospec_tps=62.352),
        2: dict(accept_sw=5.0055, accept_sw_ci=(4.7133, 5.3836), tps=248.921, tps_ci=(234.5324, 267.4567), vs_dflash=1.0901, vs_dflash_ci=(1.0535, 1.1326), vs_nospec=4.021, nospec_tps=61.905),
        8: dict(accept_sw=5.313, accept_sw_ci=(5.0318, 5.6701), tps=213.881, tps_ci=(202.7342, 227.918), vs_dflash=1.0709, vs_dflash_ci=(1.0396, 1.1048), vs_nospec=3.5747, nospec_tps=59.832),
        32: dict(accept_sw=4.9976, accept_sw_ci=(4.7252, 5.344), tps=102.923, tps_ci=(97.4702, 109.6915), vs_dflash=1.0646, vs_dflash_ci=(1.0256, 1.1016), vs_nospec=1.8903, nospec_tps=54.447),
    },
    "dspark-community": {
        1: dict(accept_sw=3.6199, accept_sw_ci=(3.5093, 3.7408), tps=183.929, tps_ci=(178.4158, 190.0227), vs_dflash=0.7303, vs_dflash_ci=(0.701, 0.7562), vs_nospec=2.9499, nospec_tps=62.352),
        2: dict(accept_sw=3.6478, accept_sw_ci=(3.5327, 3.7759), tps=174.821, tps_ci=(169.2915, 180.875), vs_dflash=0.7656, vs_dflash_ci=(0.7328, 0.7946), vs_nospec=2.824, nospec_tps=61.905),
        8: dict(accept_sw=3.6409, accept_sw_ci=(3.5221, 3.7762), tps=148.084, tps_ci=(143.3091, 153.4756), vs_dflash=0.7415, vs_dflash_ci=(0.713, 0.7689), vs_nospec=2.475, nospec_tps=59.832),
        32: dict(accept_sw=3.6074, accept_sw_ci=(3.4916, 3.7444), tps=71.551, tps_ci=(69.0234, 74.2306), vs_dflash=0.7401, vs_dflash_ci=(0.7111, 0.7673), vs_nospec=1.3141, nospec_tps=54.447),
    },
    "dspark-run-a-32k": {
        1: dict(accept_sw=4.8065, accept_sw_ci=(4.5557, 5.1278), tps=239.731, tps_ci=(227.5692, 255.0965), vs_dflash=0.9518, vs_dflash_ci=(0.9254, 0.9817), vs_nospec=3.8448, nospec_tps=62.352),
        2: dict(accept_sw=4.8324, accept_sw_ci=(4.5371, 5.1803), tps=224.126, tps_ci=(210.9219, 239.7517), vs_dflash=0.9815, vs_dflash_ci=(0.9488, 1.0142), vs_nospec=3.6205, nospec_tps=61.905),
        8: dict(accept_sw=4.8147, accept_sw_ci=(4.5288, 5.1567), tps=192.127, tps_ci=(181.1447, 205.279), vs_dflash=0.962, vs_dflash_ci=(0.9351, 0.9918), vs_nospec=3.2111, nospec_tps=59.832),
        32: dict(accept_sw=4.8237, accept_sw_ci=(4.5534, 5.1541), tps=96.416, tps_ci=(91.389, 102.3929), vs_dflash=0.9973, vs_dflash_ci=(0.9716, 1.0246), vs_nospec=1.7708, nospec_tps=54.447),
    },
    "dspark-run-b-49k": {
        1: dict(accept_sw=4.7796, accept_sw_ci=(4.5197, 5.1), tps=238.6, tps_ci=(226.0904, 254.1184), vs_dflash=0.9473, vs_dflash_ci=(0.9201, 0.9772), vs_nospec=3.8267, nospec_tps=62.352),
        2: dict(accept_sw=4.7371, accept_sw_ci=(4.4546, 5.0944), tps=219.567, tps_ci=(207.1487, 235.3167), vs_dflash=0.9616, vs_dflash_ci=(0.9324, 0.9968), vs_nospec=3.5468, nospec_tps=61.905),
        8: dict(accept_sw=4.8119, accept_sw_ci=(4.5234, 5.158), tps=192.182, tps_ci=(181.2289, 205.588), vs_dflash=0.9623, vs_dflash_ci=(0.9265, 0.9991), vs_nospec=3.212, nospec_tps=59.832),
        32: dict(accept_sw=4.707, accept_sw_ci=(4.4195, 5.0554), tps=97.434, tps_ci=(91.8515, 104.1565), vs_dflash=1.0078, vs_dflash_ci=(0.9696, 1.0476), vs_nospec=1.7895, nospec_tps=54.447),
    },
}

METRICS = {
    "tps": dict(
        field="tps", ci="tps_ci",
        ylabel="Decode throughput (output tokens / s), pooled",
        title="Decode throughput vs concurrency",
        fmt="{:.0f}", ratio=False,
        note="pooled Σgen / Σ(steps × t_step); error bars = 95% paired bootstrap CI"),
    "accept_sw": dict(
        field="accept_sw", ci="accept_sw_ci",
        ylabel="Acceptance length, step-weighted",
        title="Acceptance length vs concurrency",
        fmt="{:.2f}", ratio=False,
        note="1 + Σaccepted/Σsteps; error bars = 95% paired bootstrap CI"),
    "vs_dflash": dict(
        field="vs_dflash", ci="vs_dflash_ci",
        ylabel="Decode throughput relative to DFlash (official)",
        title="Speed-up over the DFlash baseline vs concurrency",
        fmt="{:.2f}x", ratio=True,
        note="pooled tok/s ÷ DFlash official's at the same concurrency; "
             "error bars = 95% paired bootstrap CI"),
    "vs_nospec": dict(
        field="vs_nospec", ci=None,
        ylabel="Decode throughput relative to no speculation",
        title="Speed-up over autoregressive decoding vs concurrency",
        fmt="{:.2f}x", ratio=True,
        note="pooled tok/s ÷ the no-spec control's; POINT ESTIMATE ONLY -- the "
             "control cannot be paired or resampled (see module docstring)"),
}


def series(key, metric):
    spec = METRICS[metric]
    vals = [CONC_DATA[key][c][spec["field"]] for c in CONCURRENCIES]
    if not spec["ci"]:
        return vals, None
    lo, hi = [], []
    for c, v in zip(CONCURRENCIES, vals):
        a, b = CONC_DATA[key][c][spec["ci"]]
        lo.append(v - a)
        hi.append(b - v)
    return vals, [lo, hi]


def make_chart(family, metric, ref, out_dir, dpi, ext, annotate):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = METRICS[metric]
    keys = FAMILIES[family]
    if metric == "vs_dflash":
        keys = [k for k in keys if k != "dflash-official"]

    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    x = list(range(len(CONCURRENCIES)))
    # Series converge at high concurrency, so labels are staggered vertically
    # rather than all sitting at +8pt on top of each other.
    offsets = [(0, 12), (0, -26), (0, 36), (0, -50), (0, 58), (0, -72), (0, 80)]

    for si, k in enumerate(keys):
        vals, yerr = series(k, metric)
        ax.errorbar(x, vals, yerr=yerr, label=LABELS[k], color=COLORS[k],
                    marker=MARKERS[k], markersize=6.5, linewidth=2.0,
                    capsize=4, elinewidth=1.1, zorder=3)
        if annotate == "none":
            continue
        for i, v in enumerate(vals):
            if annotate == "endpoints" and i not in (0, len(vals) - 1):
                continue
            txt = spec["fmt"].format(v)
            if annotate == "value+pct" and k != ref and not spec["ratio"]:
                rv = CONC_DATA[ref][CONCURRENCIES[i]][spec["field"]]
                txt += f"\n{100.0 * (v - rv) / rv:+.0f}%"
            ax.annotate(txt, (i, v), textcoords="offset points",
                        xytext=offsets[si % len(offsets)],
                        ha="center", fontsize=8.5, fontweight="bold",
                        color=COLORS[k], zorder=5)

    if spec["ratio"]:
        ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1.0, zorder=1)
        ax.annotate("parity", (len(x) - 1, 1.0), textcoords="offset points",
                    xytext=(6, 2), fontsize=8.5, color="#444444")

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in CONCURRENCIES])
    ax.set_xlabel("Concurrent requests", fontsize=11)
    ax.set_ylabel(spec["ylabel"], fontsize=11)
    ax.set_title(spec["title"], fontsize=12.5, pad=14)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5,
              loc="best" if spec["ratio"] else "upper right")

    tail = ""
    if annotate == "value+pct" and not spec["ratio"]:
        tail = f"  %% is vs {LABELS[ref]}."
    fig.text(0.5, 0.035,
             f"{spec['note']} (n={N_CALLS} per point).{tail}\n"
             f"Greedy, NUM_SPEC_TOKENS=15, output bucket 64-256 from the original "
             f"recording, Terminal-Bench replay.",
             ha="center", va="bottom", fontsize=8.5, color="#555555")

    fig.tight_layout(rect=(0, 0.10, 1, 1))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"slide_{family}_line_{metric}.{ext}")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def table(family, metric):
    spec = METRICS[metric]
    keys = FAMILIES[family]
    head = "| drafter | " + " | ".join(f"c{c}" for c in CONCURRENCIES) + " |"
    sep = "|---" * (len(CONCURRENCIES) + 1) + "|"
    rows = []
    for k in keys:
        vals, _ = series(k, metric)
        cells = []
        for c, v in zip(CONCURRENCIES, vals):
            cell = spec["fmt"].format(v)
            if spec["ci"]:
                a, b = CONC_DATA[k][c][spec["ci"]]
                cell += f" [{a:.3g}, {b:.3g}]"
            cells.append(cell)
        rows.append(f"| {LABELS[k]} | " + " | ".join(cells) + " |")
    ns = "| *no speculation* | " + " | ".join(
        f"{CONC_DATA['dflash-official'][c]['nospec_tps']:.1f}" for c in CONCURRENCIES) + " |"
    out = [f"**{spec['title']}** -- {spec['note']}", "", head, sep] + rows
    if metric == "tps":
        out.append(ns)
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", choices=sorted(FAMILIES), default="dspark")
    p.add_argument("--metric", choices=sorted(METRICS) + ["all"], default="all")
    p.add_argument("--ref", default=None, help="drafter key used for the %% labels")
    p.add_argument("--annotate",
                   choices=["none", "endpoints", "value", "value+pct"],
                   default="endpoints",
                   help="endpoints (default) labels only c1 and c32, which is the "
                        "only mode that stays readable where the lines converge")
    p.add_argument("--out-dir", default="benchmark/plots/out")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--ext", default="png", choices=["png", "pdf", "svg"])
    p.add_argument("--table-only", action="store_true")
    a = p.parse_args(argv)

    ref = a.ref or DEFAULT_REF[a.family]
    if ref not in CONC_DATA:
        p.error(f"--ref must be one of {sorted(CONC_DATA)}")
    metrics = sorted(METRICS) if a.metric == "all" else [a.metric]

    for m in metrics:
        print(table(a.family, m))
        print()
    if a.table_only:
        return 0
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib is not installed. Either:\n"
              "  uv run --with matplotlib python benchmark/plots/slide_lines.py ...\n"
              "  pip install matplotlib\n"
              "Or re-run with --table-only.", file=sys.stderr)
        return 1
    for m in metrics:
        print("wrote", make_chart(a.family, m, ref, a.out_dir, a.dpi, a.ext, a.annotate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
