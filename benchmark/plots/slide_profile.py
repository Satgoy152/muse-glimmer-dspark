#!/usr/bin/env python3
"""Profiling figures: where GPU time goes, and why acceptance is the lever.

Every number is HARD-CODED below, so this file runs anywhere with no access to
the GPU node, the repo's results/ tree, or any other script.

PROVENANCE
----------
docs/RESULTS-profiling-dflash2.md and benchmark/profiling/results/analysis/*.json.
Measured 2026-09-08 on the preemptible 1x H200, vLLM 0.28.1rc1.dev451+g1970f3ed4,
torch 2.13.0+cu130, driver 580.173.02.  Target meta-models/Muse-Glimmer-30B
(BF16), drafter /mnt/data/speculators/dflash2, num_speculative_tokens=15 (a
16-token speculative block).  V2 runner, FULL_AND_PIECEWISE CUDA graphs, prefix
caching on.  Nsight Systems, bounded 50-step steady-state decode window per case.

Twenty real Terminal-Bench prompts at reasoning strength `high`, replayed as raw
token IDs against /v1/completions so no chat parser sits in the timing path.
Ten short (1,568-2,281 tokens) and ten long (21,327-23,222).  Greedy,
ignore_eos=true, exactly 512 output tokens per request.

THIS IS A SPEED PROFILE, NOT A QUALITY EVALUATION.  Nothing here measures
whether the drafter proposes good tokens.

THE THREE CHARTS
----------------
  phases    Share of traced GPU kernel time by phase, for all four cases.
            Shows the drafter is ~13% of GPU time, so making it cheaper has
            almost nothing to recover.  MEASURED.
  prompts   Three short prompts at concurrency 1: ms/step is constant within
            3% while throughput varies 6x, and the spread is entirely
            accepted-length spread.  MEASURED, and the strongest single piece
            of evidence that acceptance is the knob.
  headroom  Arithmetic scenarios on tok/s = accept_len / t_step, using the
            measured phase shares.  DERIVED, NOT MEASURED -- the two "made
            free" bars are unreachable upper bounds, not proposals.

CAVEAT THAT MUST TRAVEL WITH THE `prompts` CHART
------------------------------------------------
`ignore_eos=true` inflates accepted length: forcing generation past the natural
stop drives the model into repetition, which the drafter predicts almost
perfectly.  That is what the 12.97 is.  Fixed-length greedy output is the right
control for a speed microbenchmark, but these accepted lengths are an UPPER
BOUND and are not a quality result.  The point the chart makes -- that step time
is flat while acceptance moves throughput -- does not depend on the inflation.

OTHER CAVEATS
-------------
* Absolute times come from a traced run.  Per-step time inside the capture is
  within 1% of unprofiled at concurrency 1 and within 6-14% at concurrency 10,
  so shares and shapes are trustworthy.  Whole-wave throughput during profiling
  is 43-77% lower, dominated by fixed Nsight start/stop cost -- do not quote
  that as tracing overhead.
* NVTX recorded inside a CUDA graph's capture does not replay, so there is no
  subphase attribution *within* either full graph.
* GPU busy is 96.9-97.7% of the capture window, so the ~13% drafter share is a
  share of a nearly saturated GPU; the headroom arithmetic assumes step time
  scales with GPU kernel time, which is a modelling assumption, not a measurement.

USAGE
-----
  uv run --with matplotlib python benchmark/plots/slide_profile.py
  uv run --with matplotlib python benchmark/plots/slide_profile.py --chart phases
  uv run python benchmark/plots/slide_profile.py --table-only
"""
from __future__ import annotations

import argparse
import os
import sys

# --------------------------------------------------------------------------
# Share of traced GPU kernel time (%), from analysis/{case}.json.
# The six rows below cover 99.65%+ of kernel time in every case; everything
# else (target_execute, prepare_inputs, prepare_attn, postprocess_sampled,
# sample_and_draft, target_logits_and_sampling, unclassified) is folded into
# "Scheduling / other".
# --------------------------------------------------------------------------
CASES = ["short-c1", "short-c10", "long-c1", "long-c10"]
CASE_LABELS = {
    "short-c1":  "short prompts\nconcurrency 1",
    "short-c10": "short prompts\nconcurrency 10",
    "long-c1":   "long prompts\nconcurrency 1",
    "long-c10":  "long prompts\nconcurrency 10",
}

PHASES = [
    ("Target forward (CUDA graph)", "#2C5F9E", "target"),
    ("Target LM head (202,048 vocab)", "#4F86C6", "target"),
    ("Rejection sampling", "#8FB4DC", "target"),
    ("Draft forward (CUDA graph, 1 pass)", "#D2691E", "draft"),
    ("Draft eager ops (proj + KV + propose)", "#E8A87C", "draft"),
    ("Scheduling / other", "#C9CDD2", "other"),
]

PHASE_SHARE = {
    "short-c1":  [82.881, 3.536, 0.239, 12.078, 0.968, 0.298],
    "short-c10": [82.937, 3.423, 0.279, 12.109, 0.967, 0.284],
    "long-c1":   [83.017, 3.522, 0.199, 11.995, 0.961, 0.305],
    "long-c10":  [83.736, 3.270, 0.288, 11.522, 0.914, 0.270],
}

# Headline per case: docs/RESULTS-profiling-dflash2.md
HEADLINE = {
    "short-c1":  dict(tok_s=305.7, accept=6.03, ms_step=19.64, gpu_busy=96.9),
    "short-c10": dict(tok_s=969.2, accept=6.04, ms_step=21.19, gpu_busy=97.7),
    "long-c1":   dict(tok_s=406.5, accept=8.23, ms_step=20.03, gpu_busy=96.9),
    "long-c10":  dict(tok_s=1540.5, accept=7.94, ms_step=23.68, gpu_busy=97.7),
}

# Three individual short prompts at concurrency 1, before averaging.
PROMPTS = [
    ("short-00", 5.22, 19.4, 266),
    ("short-01", 2.05, 19.3, 106),
    ("short-02", 12.97, 19.8, 645),
]

# Base case for the headroom arithmetic.
BASE = "short-c1"

# Measured acceptance gain of our DFlash2 fine-tune over native DFlash2, per
# turn, on the Terminal-Bench 64-256 bucket at concurrency 1 (see
# benchmark/plots/slide_bars.py): 7.350 vs 6.306.  Its t_step is 19.397 ms
# against 19.386, i.e. unchanged, so the acceptance gain passes straight
# through to throughput.
FINETUNE_ACCEPT_GAIN = 7.350 / 6.306


def drafter_share(case):
    """Draft forward + draft eager ops, as a share of GPU kernel time."""
    s = PHASE_SHARE[case]
    return (s[3] + s[4]) / 100.0


def lm_head_share(case):
    return PHASE_SHARE[case][1] / 100.0


def headroom_rows():
    """(label, multiplier, kind) for the headroom chart.

    tok/s = accept_len / t_step, so a cut of f in step time multiplies
    throughput by 1/(1-f), and a gain of g in acceptance multiplies it by g.
    """
    d = drafter_share(BASE)
    h = lm_head_share(BASE)
    return [
        (f"Delete the ENTIRE drafter\n(unreachable bound, {100*d:.1f}% of GPU time)",
         1.0 / (1.0 - d), "bound"),
        (f"Delete the drafter AND the LM head\n(unreachable bound, {100*(d+h):.1f}%)",
         1.0 / (1.0 - d - h), "bound"),
        ("Our DFlash2 fine-tune's measured\nacceptance gain (6.31 → 7.35/turn)",
         FINETUNE_ACCEPT_GAIN, "measured"),
        ("+2 accepted tokens\n(6.03 → 8.03)", 8.03 / 6.03, "hypothetical"),
    ]


def chart_phases(out_dir, dpi, ext):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    x = list(range(len(CASES)))
    bottom = [0.0] * len(CASES)
    for i, (name, color, _side) in enumerate(PHASES):
        vals = [PHASE_SHARE[c][i] for c in CASES]
        ax.bar(x, vals, bottom=bottom, width=0.58, color=color, label=name,
               edgecolor="white", linewidth=0.6, zorder=3)
        for j, v in enumerate(vals):
            if v >= 2.5:
                ax.text(j, bottom[j] + v / 2, f"{v:.1f}%", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white", zorder=4)
        bottom = [b + v for b, v in zip(bottom, vals)]

    # bracket the drafter's total
    for j, c in enumerate(CASES):
        tot = 100.0 * drafter_share(c)
        y = PHASE_SHARE[c][0] + PHASE_SHARE[c][1] + PHASE_SHARE[c][2] + tot / 2
        ax.annotate(f"drafter\n{tot:.1f}%", (j + 0.36, y), fontsize=9,
                    color="#8A4513", ha="left", va="center", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABELS[c] for c in CASES], fontsize=10)
    ax.set_ylabel("Share of traced GPU kernel time (%)", fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_title("DFlash2 forward-pass time split — target verification dominates",
                 fontsize=12.5, pad=14)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, fontsize=9,
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)

    fig.text(0.5, 0.012,
             "Nsight Systems, 50-step steady-state decode window per case; "
             "these six phases cover >99.6% of kernel time.\n"
             "The split moves <1 point across a 10x concurrency change and a 10x "
             "context-length change. Speed profile, not a quality evaluation.",
             ha="center", va="bottom", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"slide_profile_phases.{ext}")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def chart_prompts(out_dir, dpi, ext):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    names = [p[0] for p in PROMPTS]
    accepts = [p[1] for p in PROMPTS]
    steps = [p[2] for p in PROMPTS]
    toks = [p[3] for p in PROMPTS]
    x = list(range(len(PROMPTS)))

    ax.bar(x, toks, width=0.55, color="#4F86C6", zorder=3, label="throughput (tok/s)")
    for i, (t, a) in enumerate(zip(toks, accepts)):
        ax.text(i, t + 14, f"{t} tok/s", ha="center", fontsize=11, fontweight="bold")
        ax.text(i, t / 2, f"accept_len\n{a:.2f}", ha="center", va="center",
                fontsize=11, color="white", fontweight="bold")
    ax.set_ylim(0, max(toks) * 1.22)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Decode throughput (output tokens / s)", fontsize=11)
    ax.grid(axis="y", alpha=0.25, zorder=0)

    ax2 = ax.twinx()
    ax2.plot(x, steps, color="#D2691E", marker="o", markersize=8, linewidth=2.4,
             zorder=5, label="ms per engine step")
    for i, s in enumerate(steps):
        ax2.annotate(f"{s:.1f} ms", (i, s), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=10.5,
                     fontweight="bold", color="#D2691E")
    ax2.set_ylim(0, max(steps) * 2.1)
    ax2.set_ylabel("ms per engine step", fontsize=11, color="#D2691E")
    ax2.tick_params(axis="y", colors="#D2691E")

    for s in ("top",):
        ax.spines[s].set_visible(False)
        ax2.spines[s].set_visible(False)

    ax.set_title("Same server, same step cost — all the throughput spread is acceptance",
                 fontsize=12.5, pad=14)
    lo, hi = min(steps), max(steps)
    fig.text(0.5, 0.035,
             f"Three short prompts, concurrency 1. Step time spans {lo:.1f}-{hi:.1f} ms "
             f"({100*(hi-lo)/lo:.0f}%); throughput spans {min(toks)}-{max(toks)} tok/s "
             f"({max(toks)/min(toks):.1f}x).\n"
             "ignore_eos=true inflates accepted length (repetition drafts perfectly) — "
             "these are an UPPER BOUND, not a quality result.",
             ha="center", va="bottom", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"slide_profile_prompts.{ext}")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def chart_headroom(out_dir, dpi, ext):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = headroom_rows()
    colors = {"bound": "#C9CDD2", "measured": "#D2691E", "hypothetical": "#8FB4DC"}
    labels = [r[0] for r in rows]
    pcts = [100.0 * (r[1] - 1.0) for r in rows]
    kinds = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    y = list(range(len(rows)))[::-1]
    ax.barh(y, pcts, height=0.6, color=[colors[k] for k in kinds], zorder=3)
    base = HEADLINE[BASE]["tok_s"]
    for yi, (p, r) in zip(y, zip(pcts, rows)):
        ax.text(p + 0.6, yi, f"{p:+.1f}%   ({base * r[1]:.0f} tok/s)",
                va="center", fontsize=10.5, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlim(0, max(pcts) * 1.30)
    ax.set_xlabel("Change in decode throughput from the profiled base "
                  f"({base} tok/s, accept_len {HEADLINE[BASE]['accept']}, "
                  f"{HEADLINE[BASE]['ms_step']} ms/step)", fontsize=10)
    ax.set_title("Cost-side optimisation is capped; acceptance is not",
                 fontsize=12.5, pad=14)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    import matplotlib.patches as mpatches
    ax.legend(handles=[mpatches.Patch(color=colors["bound"], label="unreachable upper bound"),
                       mpatches.Patch(color=colors["measured"], label="measured (our fine-tune)"),
                       mpatches.Patch(color=colors["hypothetical"], label="hypothetical")],
              frameon=False, fontsize=9, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.18))

    fig.text(0.5, 0.015,
             "DERIVED, not measured: tok/s = accept_len / t_step, so cutting a "
             "fraction f of step time multiplies throughput by 1/(1-f).\n"
             "Assumes step time scales with GPU kernel time (GPU busy is 96.9% here). "
             "Deleting the drafter is not a proposal — it is the ceiling on cost-side work.",
             ha="center", va="bottom", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"slide_profile_headroom.{ext}")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


CHARTS = {"phases": chart_phases, "prompts": chart_prompts, "headroom": chart_headroom}


def tables():
    out = []
    head = "| phase | " + " | ".join(CASES) + " |"
    out.append("**Share of traced GPU kernel time (%)**\n")
    out.append(head)
    out.append("|---" * (len(CASES) + 1) + "|")
    for i, (name, _c, _s) in enumerate(PHASES):
        out.append(f"| {name} | " + " | ".join(f"{PHASE_SHARE[c][i]:.3f}" for c in CASES) + " |")
    out.append("| **drafter total** | " + " | ".join(
        f"**{100*drafter_share(c):.2f}**" for c in CASES) + " |")

    out.append("\n**Headline per case**\n")
    out.append("| case | tok/s | accept_len | ms/engine step | GPU busy |")
    out.append("|---|---:|---:|---:|---:|")
    for c in CASES:
        h = HEADLINE[c]
        out.append(f"| {c} | {h['tok_s']} | {h['accept']} | {h['ms_step']} | {h['gpu_busy']}% |")

    out.append("\n**Three short prompts, concurrency 1 (measured)**\n")
    out.append("| prompt | accept_len | ms/step | tok/s |")
    out.append("|---|---:|---:|---:|")
    for n, a, s, t in PROMPTS:
        out.append(f"| {n} | {a} | {s} | {t} |")

    out.append("\n**Headroom (DERIVED from the shares above, not measured)**\n")
    base = HEADLINE[BASE]["tok_s"]
    out.append(f"| scenario | multiplier | tok/s from {base} | change |")
    out.append("|---|---:|---:|---:|")
    for label, mult, kind in headroom_rows():
        out.append(f"| {label.replace(chr(10), ' ')} | {mult:.4f} | "
                   f"{base * mult:.0f} | {100*(mult-1):+.1f}% |")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chart", choices=sorted(CHARTS) + ["all"], default="all")
    p.add_argument("--out-dir", default="benchmark/plots/out")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--ext", default="png", choices=["png", "pdf", "svg"])
    p.add_argument("--table-only", action="store_true")
    a = p.parse_args(argv)

    print(tables())
    print()
    if a.table_only:
        return 0
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib is not installed. Either:\n"
              "  uv run --with matplotlib python benchmark/plots/slide_profile.py\n"
              "  pip install matplotlib\n"
              "Or re-run with --table-only.", file=sys.stderr)
        return 1
    names = sorted(CHARTS) if a.chart == "all" else [a.chart]
    for n in names:
        print("wrote", CHARTS[n](a.out_dir, a.dpi, a.ext))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
