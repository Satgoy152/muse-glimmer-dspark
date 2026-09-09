#!/usr/bin/env python3
"""Final-results figure: our best checkpoint against the three released baselines.

One grouped bar chart, three independent evaluations, four drafters.  Every
number is HARD-CODED below, so this file runs anywhere with no access to the
GPU node or the rest of the repo.

WHY STEP-WEIGHTED ACCEPTANCE ON ALL THREE PANELS
------------------------------------------------
The three evaluations ran at different sampling temperatures and different
concurrencies, so their tok/s columns are NOT comparable to each other -- at
concurrency 10 the same drafter replayed twice moves t_step by 5.8%, which is
larger than any effect here.  Step-weighted acceptance (1 + Sum accepted / Sum
steps) is contention-independent and is measured identically in all three, so
it is the only quantity that can share an axis.  Throughput lives on the other
slides, per benchmark.

WHY THE MIDPOINT AND NOT THE FINAL CHECKPOINT
---------------------------------------------
The final checkpoint (step 3,956) wins the concurrency-1 64-256 bucket, but the
midpoint (step 1,976) is ahead on the other two evaluations, and the final has a
reliability flag the midpoint does not: across three full Terminal-Bench replays
each, run-D final produced a completion over 16,000 tokens in EVERY run
(58,019 / 21,577 / 18,163) while none of the other seven runs -- four `dflash2`,
three midpoint -- produced one at all.  That tail is what drags its full-set
number.  The midpoint-vs-final ordering is not resolved either way
(docs/RESULTS-eval-sweep.md task 3), so this is a defensible pick, not a
measured ranking.

PROVENANCE
----------
Panel 1  Terminal-Bench full replay, 1,753 calls, temperature 1.0 / top_p 0.95 /
         top_k 64, concurrency 10.  results/terminal_bench/summary.md and
         docs/RESULTS-eval-sweep.md task 3.  Error bars are the BETWEEN-RUN sd
         over repeat replays, which is the honest spread here -- a within-run
         bootstrap says how tightly one run pins its own number down, not where
         the next run lands.  Only `dflash2` (4 runs) and our midpoint (3 runs)
         were repeated; the two single-run baselines get no bar.
Panel 2  Terminal-Bench 64-256 output bucket, 280 calls, GREEDY, concurrency 1.
         /mnt/data/eval/sweep.  Error bars are 95% paired bootstrap CIs over
         calls, 4,000 resamples, seed 20260830.
Panel 3  SWE-bench Multilingual, 2,742 calls, temperature 1.0, concurrency 10,
         32 instances over 32 repositories, ZERO overlap with the training
         corpus at both instance and repository level.
         docs/RESULTS-eval-sweep.md task 2.  No per-drafter interval was
         computed; the paired delta against `dflash2` is annotated instead.

NUM_SPEC_TOKENS = 15 everywhere.  Acceptance is only comparable at equal draft
length.

THE CAVEAT THIS FIGURE MUST NOT DROP
------------------------------------
On panel 1 -- the full Terminal-Bench workload -- our checkpoint does NOT beat
native `dflash2`.  It is a tie inside the between-run spread.  The wins are on
panel 2 (a 60%-of-calls / 24%-of-tokens slice) and panel 3 (a benchmark the
drafter was never trained or evaluated on).  Presenting panels 2 and 3 without
panel 1 is metric selection, and the deck should not do it.

USAGE
-----
  uv run --with matplotlib python benchmark/plots/slide_final.py
  uv run python benchmark/plots/slide_final.py --table-only
"""
from __future__ import annotations

import argparse
import os
import sys

PANELS = [
    dict(key="tb_full",
         title="Terminal-Bench, full workload",
         sub="1,753 calls · temp 1.0 · concurrency 10",
         err="between-run sd"),
    dict(key="tb_bucket",
         title="Terminal-Bench, 64-256 output bucket",
         sub="280 calls · greedy · concurrency 1",
         err="95% bootstrap CI"),
    dict(key="swebench_ml",
         title="SWE-bench Multilingual (held out)",
         sub="2,742 calls · temp 1.0 · concurrency 10",
         err=None),
]

ORDER = ["dspark-community", "dflash-official", "dflash2", "ours"]
LABELS = {
    "dspark-community": "DSpark (community)",
    "dflash-official":  "DFlash (official)",
    "dflash2":          "DFlash2",
    "ours":             "Ours — DFlash2 SWE-Gym\nfine-tune, step 1,976",
}
COLORS = {
    "dspark-community": "#BBD1EC",
    "dflash-official":  "#9AA3AE",
    "dflash2":          "#6B7580",
    "ours":             "#D2691E",
}

# Step-weighted acceptance length.  `sd` is a between-run sd (panel 1, where
# repeats exist); `ci` is a 95% bootstrap CI (panel 2).  None means no interval
# was computed -- the figure draws no bar rather than inventing one.
DATA = {
    "tb_full": {
        "dspark-community": dict(v=3.185, runs=1, sd=None),
        "dflash-official":  dict(v=3.854, runs=1, sd=None),
        "dflash2":          dict(v=4.0075, runs=4, sd=0.0151),
        "ours":             dict(v=3.9890, runs=3, sd=0.0716),
    },
    "tb_bucket": {
        "dspark-community": dict(v=3.6199, ci=(3.5129, 3.7392)),
        "dflash-official":  dict(v=4.8242, ci=(4.5987, 5.1047)),
        "dflash2":          dict(v=4.8390, ci=(4.6112, 5.1186)),
        "ours":             dict(v=5.0664, ci=(4.7855, 5.4201)),
    },
    "swebench_ml": {
        "dspark-community": dict(v=3.306),
        "dflash-official":  dict(v=4.708),
        "dflash2":          dict(v=4.796),
        "ours":             dict(v=5.081),
    },
}

# Paired delta of our checkpoint against `dflash2`, with its 95% interval, as
# reported for each evaluation.  None where no paired bootstrap was run.
PAIRED_VS_DFLASH2 = {
    "tb_full":     dict(text="tie", detail="-0.019, inside the 0.072 between-run sd"),
    "tb_bucket":   dict(text="+0.23", detail="sub-buckets: 64-128 +0.436 [+0.193, +0.746]; "
                                             "128-256 +0.113 [-0.056, +0.301]"),
    "swebench_ml": dict(text="+0.300", detail="95% CI [+0.189, +0.418]; "
                                              "same-drafter repeat control +0.021 [-0.084, +0.127]"),
}

NOSPEC = 1.000  # acceptance floor, by construction


def make_chart(out_dir, dpi, ext):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.9), sharey=True)
    width = 0.62

    for ax, panel in zip(axes, PANELS):
        d = DATA[panel["key"]]
        xs = list(range(len(ORDER)))
        vals = [d[k]["v"] for k in ORDER]
        yerr = None
        if panel["key"] == "tb_full":
            lo = [d[k].get("sd") or 0.0 for k in ORDER]
            yerr = [lo, lo]
        elif panel["key"] == "tb_bucket":
            lo = [d[k]["v"] - d[k]["ci"][0] for k in ORDER]
            hi = [d[k]["ci"][1] - d[k]["v"] for k in ORDER]
            yerr = [lo, hi]

        ax.bar(xs, vals, width=width, color=[COLORS[k] for k in ORDER],
               yerr=yerr, capsize=4,
               error_kw=dict(ecolor="#333333", elinewidth=1.1), zorder=3)
        top = [v + (yerr[1][i] if yerr else 0) for i, v in enumerate(vals)]
        for i, v in enumerate(vals):
            ax.text(i, top[i] + 0.10, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=11.5, fontweight="bold",
                    color="#8A4513" if ORDER[i] == "ours" else "#222222", zorder=4)

        p = PAIRED_VS_DFLASH2[panel["key"]]
        ax.set_title(f"{panel['title']}\n{panel['sub']}", fontsize=10.5, pad=10)
        ax.set_xticks(xs)
        ax.set_xticklabels(["" for _ in ORDER])
        ax.grid(axis="y", alpha=0.25, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.annotate(f"vs DFlash2:  {p['text']}", (0.5, 0.965),
                    xycoords="axes fraction", ha="center", va="top",
                    fontsize=10, fontweight="bold",
                    color="#8A4513" if p["text"].startswith("+") else "#555555")

    axes[0].set_ylabel("Acceptance length, step-weighted\n(tokens per decode step)",
                       fontsize=11)
    axes[0].set_ylim(0, 6.3)

    import matplotlib.patches as mpatches
    fig.legend(handles=[mpatches.Patch(color=COLORS[k],
                                       label=LABELS[k].replace("\n", " "))
                        for k in ORDER],
               frameon=False, fontsize=10, ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, 0.145))

    fig.suptitle("Our fine-tune against the three released drafters — "
                 "step-weighted acceptance, NUM_SPEC_TOKENS=15",
                 fontsize=13, y=0.985)
    fig.text(0.5, 0.010,
             "Step-weighted acceptance is contention-independent and measured "
             "identically in all three; the tok/s columns are not comparable "
             "across panels (different temperature and concurrency).\n"
             "On the FULL Terminal-Bench workload our checkpoint ties native "
             "DFlash2 — it does not beat it. The wins are on the 64-256 slice "
             "(60% of calls, 24% of tokens) and on the held-out benchmark.",
             ha="center", va="bottom", fontsize=8.5, color="#555555")

    fig.tight_layout(rect=(0, 0.20, 1, 0.965))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"slide_final_results.{ext}")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path



# --------------------------------------------------------------------------
# Two-metric variant: one evaluation, per-turn acceptance beside pooled tok/s.
#
# Terminal-Bench 64-256 output bucket, 280 calls, GREEDY, concurrency 1 --
# the only evaluation where BOTH metrics are measured with intervals and where
# tok/s is actually resolvable (at concurrency 10 the same drafter replayed
# twice moves t_step 5.8%, which is larger than any effect here).
#
#   accept_pr  per-turn mean acceptance over the 280 calls, error bars = SEM.
#   tps        pooled Sum gen / Sum (steps x t_step).  This is exactly the
#              TOKEN-WEIGHTED HARMONIC MEAN of the per-call rates -- verified
#              identical to the last digit -- which is the correct mean for a
#              rate.  Error bars are 95% paired bootstrap CIs over calls,
#              4,000 resamples, seed 20260830.
#
# THE CAVEAT THIS PANEL PAIR MUST NOT DROP: this is the 64-256 slice, which is
# 60.35% of Terminal-Bench calls but only 24.19% of its decoded tokens.  On the
# FULL Terminal-Bench workload the final checkpoint scores 3.923 step-weighted
# against native DFlash2's 4.008 -- it loses there.  Quote this chart as the
# training-length slice, not as the whole workload.
# --------------------------------------------------------------------------
TWO_METRIC = {
    "dspark-community":    dict(accept_pr=4.0220, accept_pr_sem=0.0745,
                                tps=183.929, tps_ci=(178.5230, 189.9745)),
    "dflash-official":     dict(accept_pr=6.4360, accept_pr_sem=0.1382,
                                tps=251.862, tps_ci=(240.2769, 266.4223)),
    "dflash2":             dict(accept_pr=6.3063, accept_pr_sem=0.1309,
                                tps=249.648, tps_ci=(237.8312, 264.0745)),
    "dflash2-run-d-mid":   dict(accept_pr=7.2660, accept_pr_sem=0.1751,
                                tps=259.707, tps_ci=(245.4982, 277.6018)),
    "dflash2-run-d-final": dict(accept_pr=7.3500, accept_pr_sem=0.1759,
                                tps=265.850, tps_ci=(250.5107, 285.3001)),
}
NOSPEC_TPS = 62.4          # no-speculation control, same bucket, concurrency 1
HIGHLIGHT_LABELS = {
    "final": "Ours — DFlash2 SWE-Gym\nfine-tune, step 3,956",
    "mid":   "Ours — DFlash2 SWE-Gym\nfine-tune, step 1,976",
}
HIGHLIGHT_KEY = {"final": "dflash2-run-d-final", "mid": "dflash2-run-d-mid"}
REF = "dflash2"


def make_chart_two_metric(highlight, out_dir, dpi, ext):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hk = HIGHLIGHT_KEY[highlight]
    keys = ["dspark-community", "dflash-official", "dflash2", hk]
    names = [LABELS["dspark-community"], LABELS["dflash-official"],
             LABELS["dflash2"], HIGHLIGHT_LABELS[highlight]]
    colors = [COLORS["dspark-community"], COLORS["dflash-official"],
              COLORS["dflash2"], COLORS["ours"]]

    panels = [
        dict(field="accept_pr", err="sem",
             title="Acceptance length, per turn",
             ylab="Tokens accepted per decode step\n(mean over 280 calls)",
             fmt="{:.2f}", note="error bars = SEM"),
        dict(field="tps", err="ci",
             title="Decode throughput, token-weighted harmonic mean",
             ylab="Output tokens / s\n(pooled Σgen / Σ decode time)",
             fmt="{:.0f}", note="error bars = 95% paired bootstrap CI"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.9))
    for ax, pan in zip(axes, panels):
        vals = [TWO_METRIC[k][pan["field"]] for k in keys]
        if pan["err"] == "sem":
            e = [TWO_METRIC[k]["accept_pr_sem"] for k in keys]
            yerr, tops = [e, e], [v + x for v, x in zip(vals, e)]
        else:
            lo = [v - TWO_METRIC[k]["tps_ci"][0] for k, v in zip(keys, vals)]
            hi = [TWO_METRIC[k]["tps_ci"][1] - v for k, v in zip(keys, vals)]
            yerr, tops = [lo, hi], [v + x for v, x in zip(vals, hi)]

        xs = list(range(len(keys)))
        ax.bar(xs, vals, width=0.62, color=colors, yerr=yerr, capsize=5,
               error_kw=dict(ecolor="#333333", elinewidth=1.2), zorder=3)

        refv = TWO_METRIC[REF][pan["field"]]
        headroom = max(tops) * 0.19
        for i, (k, v) in enumerate(zip(keys, vals)):
            ax.text(i, tops[i] + headroom * 0.09, pan["fmt"].format(v),
                    ha="center", va="bottom", fontsize=13, fontweight="bold",
                    color="#8A4513" if k == hk else "#222222", zorder=4)
            if k != REF:
                pct = 100.0 * (v - refv) / refv
                ax.text(i, tops[i] + headroom * 0.42, f"{pct:+.1f}%",
                        ha="center", va="bottom", fontsize=11,
                        fontweight="bold" if k == hk else "normal",
                        color="#1a7a3c" if pct >= 0 else "#a02020", zorder=4)

        if pan["field"] == "tps":
            ax.axhline(NOSPEC_TPS, color="#a02020", linestyle="--", linewidth=1.2,
                       zorder=2)
            ax.annotate(f"no speculation — {NOSPEC_TPS:.0f} tok/s",
                        (-0.42, NOSPEC_TPS), textcoords="offset points",
                        xytext=(0, 7), ha="left", fontsize=9, color="#a02020")
            for i, k in enumerate(keys):
                ax.text(i, TWO_METRIC[k]["tps"] * 0.5,
                        f"{TWO_METRIC[k]['tps'] / NOSPEC_TPS:.2f}x",
                        ha="center", va="center", fontsize=12 if k == hk else 10.5,
                        fontweight="bold", color="white", zorder=5)

        ax.set_ylim(0, max(tops) + headroom)
        ax.set_xticks(xs)
        ax.set_xticklabels(names, fontsize=9.5)
        ax.set_ylabel(pan["ylab"], fontsize=10.5)
        ax.set_title(f"{pan['title']}\n({pan['note']}, % vs DFlash2)",
                     fontsize=11.5, pad=12)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.suptitle("Terminal-Bench 64-256 output bucket · greedy · concurrency 1 · "
                 "NUM_SPEC_TOKENS=15", fontsize=12.5, y=0.985)
    fig.text(0.5, 0.012,
             "Pooled throughput equals the token-weighted harmonic mean of the "
             "per-call rates, exactly — the correct mean for a rate. The "
             "arithmetic mean over turns runs ~33% high.\n"
             "This is the 64-256 slice: 60.4% of Terminal-Bench calls but 24.2% "
             "of its decoded tokens. On the FULL workload this checkpoint scores "
             "3.92 step-weighted against native DFlash2's 4.01.",
             ha="center", va="bottom", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.10, 1, 0.965))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"slide_final_two_metric_{highlight}.{ext}")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def table():
    out = ["**Step-weighted acceptance length, NUM_SPEC_TOKENS=15**\n"]
    out.append("| drafter | " + " | ".join(p["title"] for p in PANELS) + " |")
    out.append("|---" * (len(PANELS) + 1) + "|")
    for k in ORDER:
        cells = []
        for p in PANELS:
            d = DATA[p["key"]][k]
            c = f"{d['v']:.3f}"
            if d.get("sd"):
                c += f" ±{d['sd']:.4f} (sd, {d['runs']} runs)"
            elif d.get("ci"):
                c += f" [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]"
            cells.append(c)
        out.append(f"| {LABELS[k].replace(chr(10), ' ')} | " + " | ".join(cells) + " |")
    out.append("\n**Our checkpoint vs `dflash2`, paired**\n")
    out.append("| evaluation | delta | detail |")
    out.append("|---|---|---|")
    for p in PANELS:
        q = PAIRED_VS_DFLASH2[p["key"]]
        out.append(f"| {p['title']} | {q['text']} | {q['detail']} |")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chart", choices=["three_panel", "two_metric", "all"],
                   default="all",
                   help="three_panel = three evaluations on step-weighted "
                        "acceptance; two_metric = one evaluation, per-turn "
                        "acceptance beside pooled tok/s")
    p.add_argument("--highlight", choices=["mid", "final"], default="final",
                   help="which of our checkpoints the two_metric chart features")
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
        print("matplotlib is not installed. Use:\n"
              "  uv run --with matplotlib python benchmark/plots/slide_final.py\n"
              "or re-run with --table-only.", file=sys.stderr)
        return 1
    if a.chart in ("three_panel", "all"):
        print("wrote", make_chart(a.out_dir, a.dpi, a.ext))
    if a.chart in ("two_metric", "all"):
        print("wrote", make_chart_two_metric(a.highlight, a.out_dir, a.dpi, a.ext))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
