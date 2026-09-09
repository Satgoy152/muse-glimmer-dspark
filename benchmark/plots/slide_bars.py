#!/usr/bin/env python3
"""Bar charts for the deck: acceptance length and decode throughput per drafter.

All numbers are HARD-CODED below so this file runs anywhere, with no access to
the GPU node, the repo's results/ tree, or any other script.

PROVENANCE OF THE NUMBERS
-------------------------
Source: the Terminal-Bench bucket x concurrency sweep, /mnt/data/eval/sweep on
the Nebius node (report.json + percall.json), summarised in
docs/RESULTS-eval-sweep.md.

  * greedy decoding, NUM_SPEC_TOKENS = 15, one vLLM server per drafter
  * concurrency 1, timed serially -- the tok/s here are MEASURED, not the
    normalised `accept / t_step` estimates that appear in docs/RESULTS-turn-length.md
  * output bucket `64-256`, which is the union of the sweep's `b64_128` (160
    calls) and `b128_256` (120 calls) cells = 280 calls per drafter
  * bucket membership comes from the ORIGINAL frozen Terminal-Bench recording's
    completion_tokens, not from each drafter's own replayed output, so nothing
    selects on the outcome

Two aggregations are carried, because they answer different questions and
disagree by a large margin:

  per_request  mean over the 280 calls, every call weighted equally. This is
               the "accept len by request" the slide asks for. It has a
               per-call distribution, so its error bars are the SEM.
  pooled       for acceptance: 1 + sum(accepted) / sum(steps) (step-weighted).
               for throughput: sum(gen) / sum(steps * t_step).
               One ratio over the whole bucket, so long calls dominate. Its
               error bars are a 95% percentile bootstrap CI resampling calls
               (4,000 resamples, seed 20260830), computed on the node from
               percall.json and hard-coded here.

WHICH AVERAGE TO USE FOR THROUGHPUT
-----------------------------------
Both aggregations are equally *measured* -- they read the same per-call
counters. The choice is how calls are weighted, and for a rate it is not a
free choice:

  * `pooled` equals the TOKEN-WEIGHTED HARMONIC MEAN of the per-call rates,
    exactly, in every row. So the pooled number already IS a per-turn average
    of per-turn rates -- the harmonic one, which is the correct mean for a
    rate. (One mile at 20 mph and one at 60 mph averages 30 mph, not 40.)

  * `per_request` is the ARITHMETIC mean of per-call rates. Here
    corr(completion_tokens, per-call rate) is -0.35 to -0.42 for every
    DFlash/DSpark drafter: longer turns decode slower. The arithmetic mean
    weights a 64-token turn the same as a 256-token turn that occupies four
    times the wall-clock, so it over-weights the fast short turns. That is the
    whole 334-vs-252 gap.

  * The arithmetic mean is also the only summary under which our DSpark
    fine-tune beats DFlash official on speed:
        arithmetic mean   333.6 vs 341.7   ours +2.4%
        median            324.9 vs 316.6   ours -2.6%
        pooled/harmonic   251.9 vs 239.7   ours -4.8%
    A speed claim that survives only under the arithmetic mean is the weakest
    version of the claim.

  * Finally, a per-call rate is just accept_len_i / t_step, and t_step moves
    under 0.7% at concurrency 1. So the per-request throughput chart is the
    per-request acceptance chart with a rescaled y-axis; it carries no
    independent information.

If you want a genuinely per-turn speed statistic, use the MEDIAN with an IQR,
not the mean. Those are carried in DATA as tps_med / tps_p25 / tps_p75 (they
are not plotted; print them with --list-dist).

CAVEAT TO KEEP WITH ANY SLIDE BUILT FROM THIS
---------------------------------------------
The `64-256` bucket is 60.35% of Terminal-Bench calls but only 24.19% of its
decoded tokens. Per-request bars therefore flatter the fine-tunes relative to
what a deployment sees; the step-weighted column is much closer between
drafters. Show both, or say which one you are showing.

USAGE
-----
  uv run --with matplotlib python benchmark/plots/slide_bars.py --family dspark
  uv run --with matplotlib python benchmark/plots/slide_bars.py --family dflash2
  uv run --with matplotlib python benchmark/plots/slide_bars.py --family all \
      --value step_weighted --out-dir /tmp/figs
"""
from __future__ import annotations

import argparse
import os
import sys

# --------------------------------------------------------------------------
# DATA.  Terminal-Bench replay, greedy, concurrency 1, output bucket 64-256.
# n = 280 calls per drafter.  sem_* are standard errors of the mean over those
# 280 calls; they exist only for the per-request aggregation.
# --------------------------------------------------------------------------
N_CALLS = 280
BOOTSTRAP_RESAMPLES = 4000   # seed 20260830, percentile 95% CI, resampling calls

DATA = {
    "dflash-official": dict(
        label="DFlash\n(official)", t_step_ms=19.047,
        accept_pr=6.4360, accept_pr_sem=0.1382,
        accept_sw=4.8242, accept_sw_ci=(4.5987, 5.1047),
        tps_p25=228.5, tps_med=324.9, tps_p75=417.2,
        tps_pr=333.613, tps_pr_sem=7.018,
        tps_agg=251.862, tps_agg_ci=(240.277, 266.422)),
    "dflash2": dict(
        label="DFlash2", t_step_ms=19.386,
        accept_pr=6.3063, accept_pr_sem=0.1309,
        accept_sw=4.8390, accept_sw_ci=(4.6112, 5.1186),
        tps_p25=235.0, tps_med=298.6, tps_p75=412.1,
        tps_pr=325.485, tps_pr_sem=6.762,
        tps_agg=249.648, tps_agg_ci=(237.831, 264.074)),
    "dspark-community": dict(
        label="DSpark\n(community)", t_step_ms=19.700,
        accept_pr=4.0220, accept_pr_sem=0.0745,
        accept_sw=3.6199, accept_sw_ci=(3.5129, 3.7392),
        tps_p25=160.7, tps_med=189.0, tps_p75=232.5,
        tps_pr=204.475, tps_pr_sem=3.751,
        tps_agg=183.929, tps_agg_ci=(178.523, 189.974)),
    "dspark-run-a-32k": dict(
        label="Ours\nDSpark 32K", t_step_ms=19.724,
        accept_pr=6.9381, accept_pr_sem=0.1699,
        accept_sw=4.8065, accept_sw_ci=(4.5481, 5.1266),
        tps_p25=227.8, tps_med=316.6, tps_p75=432.0,
        tps_pr=341.708, tps_pr_sem=8.273,
        tps_agg=239.731, tps_agg_ci=(227.287, 255.264)),
    "dspark-run-b-49k": dict(
        label="Ours\nDSpark 49K", t_step_ms=19.724,
        accept_pr=6.9190, accept_pr_sem=0.1699,
        accept_sw=4.7796, accept_sw_ci=(4.5149, 5.1027),
        tps_p25=227.7, tps_med=314.1, tps_p75=428.7,
        tps_pr=341.074, tps_pr_sem=8.251,
        tps_agg=238.600, tps_agg_ci=(225.815, 254.195)),
    "dflash2-run-d-mid": dict(
        label="Ours\nDFlash2 mid", t_step_ms=19.370,
        accept_pr=7.2660, accept_pr_sem=0.1751,
        accept_sw=5.0664, accept_sw_ci=(4.7855, 5.4201),
        tps_p25=247.9, tps_med=338.0, tps_p75=474.0,
        tps_pr=370.425, tps_pr_sem=8.892,
        tps_agg=259.707, tps_agg_ci=(245.498, 277.602)),
    "dflash2-run-d-final": dict(
        label="Ours\nDFlash2 final", t_step_ms=19.397,
        accept_pr=7.3500, accept_pr_sem=0.1759,
        accept_sw=5.1896, accept_sw_ci=(4.8894, 5.5729),
        tps_p25=254.6, tps_med=345.9, tps_p75=476.9,
        tps_pr=374.567, tps_pr_sem=8.910,
        tps_agg=265.850, tps_agg_ci=(250.511, 285.300)),
}

# No-speculation control, same bucket / same concurrency, for the "x over
# autoregressive" annotation.  accept is 1.000 by construction.
NOSPEC = dict(accept_pr=1.000, accept_sw=1.000, tps_agg=62.4, t_step_ms=16.139)

# Bar order per slide.  DSpark family kept adjacent and in one hue so the
# fine-tune reads as an improvement on its own warm start.
FAMILIES = {
    "dspark": ["dflash-official", "dflash2",
               "dspark-community", "dspark-run-a-32k", "dspark-run-b-49k"],
    "dflash2": ["dflash-official", "dspark-community",
                "dflash2", "dflash2-run-d-mid", "dflash2-run-d-final"],
    "all": ["dflash-official", "dflash2",
            "dflash2-run-d-mid", "dflash2-run-d-final",
            "dspark-community", "dspark-run-a-32k", "dspark-run-b-49k"],
    # Three released baselines plus the best checkpoint from each of our two
    # families.  "Best" is by point estimate only -- DSpark 32K vs 49K and
    # DFlash2 final vs mid are inside each other's bootstrap intervals, so the
    # pick is a presentation choice, not a measured ranking.
    "best": ["dflash-official", "dflash2", "dspark-community",
             "dspark-run-a-32k", "dflash2-run-d-final"],
}

# Default reference for the "% improvement" annotation on each bar.
DEFAULT_REF = {"dspark": "dspark-community", "dflash2": "dflash2",
               "all": "dflash2", "best": "dflash-official"}

# Colour: each family gets a hue; the warm start is the pale shade, our
# fine-tunes are progressively saturated versions of the SAME hue.
COLORS = {
    "dflash-official":     "#9AA3AE",   # neutral grey  - baseline
    "dflash2":             "#6B7580",   # darker grey   - baseline
    "dspark-community":    "#BBD1EC",   # pale blue     - warm start
    "dspark-run-a-32k":    "#4F86C6",   # blue          - ours
    "dspark-run-b-49k":    "#2C5F9E",   # deep blue     - ours
    "dflash2-run-d-mid":   "#E8A87C",   # light orange  - ours
    "dflash2-run-d-final": "#D2691E",   # orange        - ours
}

METRICS = {
    "accept": dict(
        default_value="per_request",
        per_request=dict(field="accept_pr", sem="accept_pr_sem", ci=None,
                         note="per-request mean, error bars = SEM (n={n})"),
        pooled=dict(field="accept_sw", sem=None, ci="accept_sw_ci",
                    note="step-weighted, 1 + \u03a3accepted/\u03a3steps; error bars = 95% "
                         "bootstrap CI over calls"),
        ylabel="Acceptance length (tokens per decode step)",
        title="Acceptance length, Terminal-Bench 64-256 output bucket, concurrency 1",
        fmt="{:.2f}",
    ),
    "tokens": dict(
        default_value="pooled",
        per_request=dict(field="tps_pr", sem="tps_pr_sem", ci=None,
                         note="ARITHMETIC MEAN of per-call rates (n={n}), error bars = SEM "
                              "-- runs high; see --help"),
        pooled=dict(field="tps_agg", sem=None, ci="tps_agg_ci",
                    note="pooled \u03a3gen / \u03a3(steps \u00d7 t_step); "
                         "error bars = 95% bootstrap CI over calls"),
        ylabel="Decode throughput (output tokens / s)",
        title="Decode throughput, Terminal-Bench 64-256 output bucket, concurrency 1",
        fmt="{:.0f}",
    ),
}


def resolve_value(metric, value):
    return METRICS[metric]["default_value"] if value == "auto" else value


def series(family, metric, value):
    """Return (keys, values, yerr) where yerr is None, a 1-D SEM list, or a
    2xN [[lo],[hi]] array of asymmetric bootstrap-CI half-widths."""
    keys = FAMILIES[family]
    spec = METRICS[metric][resolve_value(metric, value)]
    vals = [DATA[k][spec["field"]] for k in keys]
    if spec["sem"]:
        return keys, vals, [DATA[k][spec["sem"]] for k in keys]
    if spec["ci"]:
        lo, hi = [], []
        for k, v in zip(keys, vals):
            a, b = DATA[k][spec["ci"]]
            lo.append(v - a)
            hi.append(b - v)
        return keys, vals, [lo, hi]
    return keys, vals, None


def make_chart(family, metric, value, ref, out_dir, dpi, fmt_ext):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    value = resolve_value(metric, value)
    keys, vals, yerr = series(family, metric, value)
    spec = METRICS[metric][value]
    refval = DATA[ref][spec["field"]]
    fmt = METRICS[metric]["fmt"]
    # top of each error bar, for label placement
    tops = [v + (yerr[1][i] if isinstance(yerr[0], list) else yerr[i]) if yerr else v
            for i, v in enumerate(vals)]

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    x = range(len(keys))
    bars = ax.bar(
        x, vals,
        color=[COLORS[k] for k in keys],
        width=0.62,
        yerr=yerr, capsize=5 if yerr else 0,
        error_kw=dict(ecolor="#333333", elinewidth=1.2),
        zorder=3,
    )

    headroom = max(vals) * 0.19
    for i, (k, v) in enumerate(zip(keys, vals)):
        ax.text(i, tops[i] + headroom * 0.10, fmt.format(v),
                ha="center", va="bottom", fontsize=12, fontweight="bold", zorder=4)
        if k != ref:
            pct = 100.0 * (v - refval) / refval
            ax.text(i, tops[i] + headroom * 0.40, f"{pct:+.1f}%",
                    ha="center", va="bottom", fontsize=10.5,
                    color="#1a7a3c" if pct >= 0 else "#a02020", zorder=4)

    ax.set_xticks(list(x))
    ax.set_xticklabels([DATA[k]["label"] for k in keys], fontsize=10.5)
    ax.set_ylabel(METRICS[metric]["ylabel"], fontsize=11)
    ax.set_ylim(0, max(tops) + headroom)
    ax.set_title(METRICS[metric]["title"], fontsize=12.5, pad=14)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    agg = spec["note"].format(n=N_CALLS)
    refname = DATA[ref]["label"].replace(chr(10), " ")
    fig.text(0.5, 0.035,
             f"{agg}.  % is vs {refname}.\n"
             f"Greedy, NUM_SPEC_TOKENS=15, output bucket 64-256 from the original "
             f"recording, concurrency 1.",
             ha="center", va="bottom", fontsize=8.5, color="#555555")

    fig.tight_layout(rect=(0, 0.10, 1, 1))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"slide_{family}_{metric}_{value}.{fmt_ext}")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def delta_table(family, ref):
    """Markdown table: every drafter vs `ref`, on all three aggregations."""
    keys = FAMILIES[family]
    rows = []
    for k in keys:
        d = DATA[k]
        r = DATA[ref]
        def pct(a, b):
            return "ref" if k == ref else f"{100.0 * (a - b) / b:+.1f}%"
        rows.append((
            d["label"].replace("\n", " "),
            f"{d['accept_pr']:.3f}", pct(d["accept_pr"], r["accept_pr"]),
            f"{d['accept_sw']:.3f}", pct(d["accept_sw"], r["accept_sw"]),
            f"{d['tps_agg']:.1f}", pct(d["tps_agg"], r["tps_agg"]),
            f"{d['tps_agg'] / NOSPEC['tps_agg']:.2f}x",
        ))
    head = ("| drafter | accept (per-req) | Δ | accept (step-w) | Δ | "
            "tok/s (pooled) | Δ | vs no-spec |")
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([
        f"Reference = {DATA[ref]['label'].replace(chr(10), ' ')}. "
        f"Terminal-Bench, greedy, concurrency 1, output bucket 64-256, n={N_CALLS}.",
        "", head, sep, body, "",
        "No-spec control in the same bucket: %.1f tok/s, t_step %.3f ms."
        % (NOSPEC["tps_agg"], NOSPEC["t_step_ms"]),
    ])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", choices=sorted(FAMILIES), default="dspark")
    p.add_argument("--metric", choices=["accept", "tokens", "both"], default="both")
    p.add_argument("--value", choices=["auto", "per_request", "pooled"], default="auto",
                   help="auto (default) = per_request for --metric accept, pooled for "
                        "--metric tokens. See the module docstring for why.")
    p.add_argument("--ref", default=None, help="drafter key used for the %% labels")
    p.add_argument("--out-dir", default="benchmark/plots/out")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--ext", default="png", choices=["png", "pdf", "svg"])
    p.add_argument("--table-only", action="store_true")
    p.add_argument("--list-dist", action="store_true",
                   help="print the per-call throughput distribution (mean / p25 / "
                        "median / p75) against the pooled value, then exit")
    a = p.parse_args(argv)

    if a.list_dist:
        print(f"{'drafter':22}{'mean':>8}{'p25':>8}{'median':>8}{'p75':>8}{'pooled':>8}")
        print("per-call decode rate, tok/s, TB 64-256 bucket, concurrency 1, n=%d\n"
              "pooled == the token-weighted harmonic mean of the same per-call rates."
              % N_CALLS)
        for k in FAMILIES["all"]:
            d = DATA[k]
            print(f"{k:22}{d['tps_pr']:8.1f}{d['tps_p25']:8.1f}"
                  f"{d['tps_med']:8.1f}{d['tps_p75']:8.1f}{d['tps_agg']:8.1f}")
        return 0

    ref = a.ref or DEFAULT_REF[a.family]
    if ref not in DATA:
        p.error(f"--ref must be one of {sorted(DATA)}")

    print(delta_table(a.family, ref))
    print()

    if a.table_only:
        return 0
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib is not installed. Either:\n"
              "  uv run --with matplotlib python benchmark/plots/slide_bars.py ...\n"
              "  pip install matplotlib\n"
              "Or re-run with --table-only.", file=sys.stderr)
        return 1

    metrics = ["accept", "tokens"] if a.metric == "both" else [a.metric]
    for m in metrics:
        path = make_chart(a.family, m, a.value, ref, a.out_dir, a.dpi, a.ext)
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
