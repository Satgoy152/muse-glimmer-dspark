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

Three different aggregations are carried, because they answer different
questions and disagree:

  per_request    mean over the 280 calls, every call weighted equally.
                 This is the "accept len by request" the slide asks for.
                 It has a per-call distribution, so it gets SEM error bars.
  step_weighted  1 + sum(accepted) / sum(steps). Long calls dominate. This is
                 the acceptance that actually drives aggregate throughput.
  aggregate      sum(gen_tokens) / sum(steps * t_step). Deployment throughput
                 over the whole bucket.

step_weighted and aggregate are single pooled ratios, NOT means over calls, so
they have no standard error to draw. The script refuses to put error bars on
them rather than inventing a spread.

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

DATA = {
    # key                     label                      accept_pr  sem   accept_sw  tps_pr  sem   tps_agg  t_step_ms
    "dflash-official":  dict(label="DFlash\n(official)",  accept_pr=6.4360, accept_pr_sem=0.1382, accept_sw=4.8242, tps_pr=333.613, tps_pr_sem=7.018, tps_agg=251.862, t_step_ms=19.047),
    "dflash2":          dict(label="DFlash2",             accept_pr=6.3063, accept_pr_sem=0.1309, accept_sw=4.8390, tps_pr=325.485, tps_pr_sem=6.762, tps_agg=249.648, t_step_ms=19.386),
    "dspark-community": dict(label="DSpark\n(community)", accept_pr=4.0220, accept_pr_sem=0.0745, accept_sw=3.6199, tps_pr=204.475, tps_pr_sem=3.751, tps_agg=183.929, t_step_ms=19.700),
    "dspark-run-a-32k": dict(label="Ours\nDSpark 32K",    accept_pr=6.9381, accept_pr_sem=0.1699, accept_sw=4.8065, tps_pr=341.708, tps_pr_sem=8.273, tps_agg=239.731, t_step_ms=19.724),
    "dspark-run-b-49k": dict(label="Ours\nDSpark 49K",    accept_pr=6.9190, accept_pr_sem=0.1699, accept_sw=4.7796, tps_pr=341.074, tps_pr_sem=8.251, tps_agg=238.600, t_step_ms=19.724),
    "dflash2-run-d-mid":   dict(label="Ours\nDFlash2 mid",   accept_pr=7.2660, accept_pr_sem=0.1751, accept_sw=5.0664, tps_pr=370.425, tps_pr_sem=8.892, tps_agg=259.707, t_step_ms=19.370),
    "dflash2-run-d-final": dict(label="Ours\nDFlash2 final", accept_pr=7.3500, accept_pr_sem=0.1759, accept_sw=5.1896, tps_pr=374.567, tps_pr_sem=8.910, tps_agg=265.850, t_step_ms=19.397),
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
}

# Default reference for the "% improvement" annotation on each bar.
DEFAULT_REF = {"dspark": "dspark-community", "dflash2": "dflash2", "all": "dflash2"}

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
        per_request=("accept_pr", "accept_pr_sem"),
        step_weighted=("accept_sw", None),
        aggregate=("accept_sw", None),
        ylabel="Acceptance length (tokens per decode step)",
        title="Acceptance length, Terminal-Bench 64-256 output bucket, concurrency 1",
        fmt="{:.2f}",
    ),
    "tokens": dict(
        per_request=("tps_pr", "tps_pr_sem"),
        step_weighted=("tps_agg", None),
        aggregate=("tps_agg", None),
        ylabel="Decode throughput (output tokens / s)",
        title="Decode throughput, Terminal-Bench 64-256 output bucket, concurrency 1",
        fmt="{:.0f}",
    ),
}


def series(family, metric, value):
    keys = FAMILIES[family]
    field, semfield = METRICS[metric][value]
    vals = [DATA[k][field] for k in keys]
    sems = [DATA[k][semfield] for k in keys] if semfield else None
    return keys, vals, sems


def make_chart(family, metric, value, ref, out_dir, dpi, fmt_ext):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys, vals, sems = series(family, metric, value)
    field = METRICS[metric][value][0]
    refval = DATA[ref][field]
    fmt = METRICS[metric]["fmt"]

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    x = range(len(keys))
    bars = ax.bar(
        x, vals,
        color=[COLORS[k] for k in keys],
        width=0.62,
        yerr=sems, capsize=5 if sems else 0,
        error_kw=dict(ecolor="#333333", elinewidth=1.2),
        zorder=3,
    )

    headroom = max(vals) * (0.19 if sems else 0.15)
    for i, (k, v) in enumerate(zip(keys, vals)):
        e = sems[i] if sems else 0.0
        ax.text(i, v + e + headroom * 0.10, fmt.format(v),
                ha="center", va="bottom", fontsize=12, fontweight="bold", zorder=4)
        if k != ref:
            pct = 100.0 * (v - refval) / refval
            ax.text(i, v + e + headroom * 0.40, f"{pct:+.1f}%",
                    ha="center", va="bottom", fontsize=10.5,
                    color="#1a7a3c" if pct >= 0 else "#a02020", zorder=4)

    ax.set_xticks(list(x))
    ax.set_xticklabels([DATA[k]["label"] for k in keys], fontsize=10.5)
    ax.set_ylabel(METRICS[metric]["ylabel"], fontsize=11)
    ax.set_ylim(0, max(v + (sems[i] if sems else 0) for i, v in enumerate(vals)) + headroom)
    ax.set_title(METRICS[metric]["title"], fontsize=12.5, pad=14)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    agg = {"per_request": "per-request mean, error bars = SEM (n=%d)" % N_CALLS,
           "step_weighted": "step-weighted pooled ratio - no per-call spread, so no error bars",
           "aggregate": "pooled over the bucket - no per-call spread, so no error bars"}[value]
    ax.text(0.0, -0.20, f"{agg}. % is vs {DATA[ref]['label'].replace(chr(10), ' ')}. "
                        f"Greedy, NUM_SPEC_TOKENS=15, bucketed on the original recording.",
            transform=ax.transAxes, fontsize=8.5, color="#555555")

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"slide_{family}_{metric}_{value}.{fmt_ext}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
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
    p.add_argument("--value", choices=["per_request", "step_weighted", "aggregate"],
                   default="per_request",
                   help="which aggregation to plot; only per_request has error bars")
    p.add_argument("--ref", default=None, help="drafter key used for the %% labels")
    p.add_argument("--out-dir", default="benchmark/plots/out")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--ext", default="png", choices=["png", "pdf", "svg"])
    p.add_argument("--table-only", action="store_true")
    a = p.parse_args(argv)

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
