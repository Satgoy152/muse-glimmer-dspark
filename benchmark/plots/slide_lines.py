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
    # The deck chart: three released baselines plus the best checkpoint from
    # each of our two families, rather than four near-identical fine-tune lines.
    # "Best" is by point estimate only -- DSpark 32K vs 49K (239.7 vs 238.6
    # pooled tok/s at c1) and DFlash2 final vs mid (265.9 vs 259.7) are both
    # inside each other's bootstrap intervals, so the pick is a presentation
    # choice, not a measured ranking.
    "best": ["dflash-official", "dflash2", "dspark-community",
             "dspark-run-a-32k", "dflash2-run-d-final"],
}

DEFAULT_REF = {"dspark": "dspark-community", "dflash2": "dflash2",
               "all": "dflash2", "best": "dflash-official"}

CONC_DATA = {
    "dflash-official": {
        1: dict(
            accept_pr=4.436, accept_pr_sem=0.1382,
            accept_sw=4.8242, accept_sw_ci=(4.5987, 5.1047),
            tps_pr=333.613, tps_pr_sem=7.018,
            tps=190.916, tps_ci=(182.711, 199.974),
            vs_dflash_pr=1.0, vs_dflash_pr_sem=0.0,
            vs_dflash=1.0, vs_dflash_ci=(1.0, 1.0),
            vs_nospec=4.0394, nospec_tps=62.352),
        2: dict(
            accept_pr=4.4302, accept_pr_sem=0.1405,
            accept_sw=4.7254, accept_sw_ci=(4.4817, 5.0311),
            tps_pr=304.176, tps_pr_sem=6.411,
            tps=175.669, tps_ci=(166.824, 185.7421),
            vs_dflash_pr=1.0, vs_dflash_pr_sem=0.0,
            vs_dflash=1.0, vs_dflash_ci=(1.0, 1.0),
            vs_nospec=3.6886, nospec_tps=61.905),
        8: dict(
            accept_pr=4.4259, accept_pr_sem=0.1376,
            accept_sw=4.8518, accept_sw_ci=(4.6134, 5.1471),
            tps_pr=260.389, tps_pr_sem=5.402,
            tps=148.084, tps_ci=(143.3091, 153.4756),
            vs_dflash_pr=1.0, vs_dflash_pr_sem=0.0,
            vs_dflash=1.0, vs_dflash_ci=(1.0, 1.0),
            vs_nospec=3.3379, nospec_tps=59.832),
        32: dict(
            accept_pr=4.4122, accept_pr_sem=0.1375,
            accept_sw=4.8497, accept_sw_ci=(4.6056, 5.1427),
            tps_pr=123.89, tps_pr_sem=2.534,
            tps=79.678, tps_ci=(76.0234, 83.969),
            vs_dflash_pr=1.0, vs_dflash_pr_sem=0.0,
            vs_dflash=1.0, vs_dflash_ci=(1.0, 1.0),
            vs_nospec=1.054, nospec_tps=54.447),
    },
    "dflash2": {
        1: dict(
            accept_pr=6.3063, accept_pr_sem=0.1309,
            accept_sw=4.839, accept_sw_ci=(4.6158, 5.1077),
            tps_pr=325.485, tps_pr_sem=6.762,
            tps=249.648, tps_ci=(238.0951, 263.5608),
            vs_dflash_pr=1.4216, vs_dflash_pr_sem=0.0135,
            vs_dflash=1.3076, vs_dflash_ci=(1.2576, 1.3575),
            vs_nospec=4.0039, nospec_tps=62.352),
        2: dict(
            accept_pr=6.3214, accept_pr_sem=0.1311,
            accept_sw=4.8703, accept_sw_ci=(4.6419, 5.1421),
            tps_pr=315.041, tps_pr_sem=6.511,
            tps=243.486, tps_ci=(232.1975, 257.0706),
            vs_dflash_pr=1.4269, vs_dflash_pr_sem=0.0141,
            vs_dflash=1.386, vs_dflash_ci=(1.3437, 1.4287),
            vs_nospec=3.9332, nospec_tps=61.905),
        8: dict(
            accept_pr=6.3375, accept_pr_sem=0.1285,
            accept_sw=4.9419, accept_sw_ci=(4.7162, 5.2162),
            tps_pr=257.142, tps_pr_sem=5.163,
            tps=202.617, tps_ci=(193.5587, 213.738),
            vs_dflash_pr=1.4319, vs_dflash_pr_sem=0.0128,
            vs_dflash=1.3683, vs_dflash_ci=(1.333, 1.3998),
            vs_nospec=3.3864, nospec_tps=59.832),
        32: dict(
            accept_pr=6.3251, accept_pr_sem=0.1312,
            accept_sw=4.9058, accept_sw_ci=(4.6785, 5.1727),
            tps_pr=122.05, tps_pr_sem=2.486,
            tps=97.618, tps_ci=(93.2197, 102.7684),
            vs_dflash_pr=1.4335, vs_dflash_pr_sem=0.0144,
            vs_dflash=1.2252, vs_dflash_ci=(1.1885, 1.2579),
            vs_nospec=1.7929, nospec_tps=54.447),
    },
    "dflash2-run-d-mid": {
        1: dict(
            accept_pr=7.266, accept_pr_sem=0.1751,
            accept_sw=5.0664, accept_sw_ci=(4.7901, 5.4174),
            tps_pr=370.425, tps_pr_sem=8.892,
            tps=259.707, tps_ci=(245.6374, 277.5883),
            vs_dflash_pr=1.638, vs_dflash_pr_sem=0.0162,
            vs_dflash=1.3603, vs_dflash_ci=(1.3187, 1.4074),
            vs_nospec=4.1652, nospec_tps=62.352),
        2: dict(
            accept_pr=7.2482, accept_pr_sem=0.1741,
            accept_sw=5.0282, accept_sw_ci=(4.7063, 5.4115),
            tps_pr=357.544, tps_pr_sem=8.525,
            tps=249.884, tps_ci=(234.0727, 268.7261),
            vs_dflash_pr=1.6361, vs_dflash_pr_sem=0.0186,
            vs_dflash=1.4225, vs_dflash_ci=(1.3719, 1.4795),
            vs_nospec=4.0366, nospec_tps=61.905),
        8: dict(
            accept_pr=7.2302, accept_pr_sem=0.176,
            accept_sw=5.0657, accept_sw_ci=(4.7828, 5.435),
            tps_pr=288.307, tps_pr_sem=6.884,
            tps=205.054, tps_ci=(193.9381, 219.6134),
            vs_dflash_pr=1.6336, vs_dflash_pr_sem=0.016,
            vs_dflash=1.3847, vs_dflash_ci=(1.3375, 1.4321),
            vs_nospec=3.4272, nospec_tps=59.832),
        32: dict(
            accept_pr=7.2484, accept_pr_sem=0.1766,
            accept_sw=5.1211, accept_sw_ci=(4.8346, 5.4736),
            tps_pr=138.497, tps_pr_sem=3.258,
            tps=100.934, tps_ci=(95.4615, 107.5263),
            vs_dflash_pr=1.6428, vs_dflash_pr_sem=0.0186,
            vs_dflash=1.2668, vs_dflash_ci=(1.2311, 1.3005),
            vs_nospec=1.8538, nospec_tps=54.447),
    },
    "dflash2-run-d-final": {
        1: dict(
            accept_pr=7.35, accept_pr_sem=0.1759,
            accept_sw=5.1896, accept_sw_ci=(4.8859, 5.5636),
            tps_pr=374.567, tps_pr_sem=8.91,
            tps=265.85, tps_ci=(250.4393, 284.7404),
            vs_dflash_pr=1.6569, vs_dflash_pr_sem=0.0167,
            vs_dflash=1.3925, vs_dflash_ci=(1.3332, 1.4525),
            vs_nospec=4.2637, nospec_tps=62.352),
        2: dict(
            accept_pr=7.3119, accept_pr_sem=0.1768,
            accept_sw=5.0055, accept_sw_ci=(4.7133, 5.3836),
            tps_pr=361.138, tps_pr_sem=8.671,
            tps=248.921, tps_ci=(234.5324, 267.4567),
            vs_dflash_pr=1.6505, vs_dflash_pr_sem=0.0187,
            vs_dflash=1.417, vs_dflash_ci=(1.3694, 1.4722),
            vs_nospec=4.021, nospec_tps=61.905),
        8: dict(
            accept_pr=7.3001, accept_pr_sem=0.1742,
            accept_sw=5.313, accept_sw_ci=(5.0318, 5.6701),
            tps_pr=289.994, tps_pr_sem=6.796,
            tps=213.881, tps_ci=(202.7342, 227.918),
            vs_dflash_pr=1.6494, vs_dflash_pr_sem=0.0158,
            vs_dflash=1.4443, vs_dflash_ci=(1.4021, 1.49),
            vs_nospec=3.5747, nospec_tps=59.832),
        32: dict(
            accept_pr=7.3083, accept_pr_sem=0.1792,
            accept_sw=4.9976, accept_sw_ci=(4.7252, 5.344),
            tps_pr=144.604, tps_pr_sem=3.396,
            tps=102.923, tps_ci=(97.4702, 109.6915),
            vs_dflash_pr=1.6564, vs_dflash_pr_sem=0.0195,
            vs_dflash=1.2917, vs_dflash_ci=(1.2444, 1.3366),
            vs_nospec=1.8903, nospec_tps=54.447),
    },
    "dspark-community": {
        1: dict(
            accept_pr=4.022, accept_pr_sem=0.0745,
            accept_sw=3.6199, accept_sw_ci=(3.5093, 3.7408),
            tps_pr=204.475, tps_pr_sem=3.751,
            tps=183.929, tps_ci=(178.4158, 190.0227),
            vs_dflash_pr=0.9067, vs_dflash_pr_sem=0.0135,
            vs_dflash=0.9634, vs_dflash_ci=(0.9248, 0.9976),
            vs_nospec=2.9499, nospec_tps=62.352),
        2: dict(
            accept_pr=4.0379, accept_pr_sem=0.0748,
            accept_sw=3.6478, accept_sw_ci=(3.5327, 3.7759),
            tps_pr=191.924, tps_pr_sem=3.503,
            tps=174.821, tps_ci=(169.2915, 180.875),
            vs_dflash_pr=0.9114, vs_dflash_pr_sem=0.0141,
            vs_dflash=0.9952, vs_dflash_ci=(0.9525, 1.0329),
            vs_nospec=2.824, nospec_tps=61.905),
        8: dict(
            accept_pr=4.046, accept_pr_sem=0.0745,
            accept_sw=3.6409, accept_sw_ci=(3.5221, 3.7762),
            tps_pr=163.52, tps_pr_sem=2.963,
            tps=148.084, tps_ci=(143.3091, 153.4756),
            vs_dflash_pr=0.9142, vs_dflash_pr_sem=0.0144,
            vs_dflash=1.0, vs_dflash_ci=(0.9616, 1.037),
            vs_nospec=2.475, nospec_tps=59.832),
        32: dict(
            accept_pr=4.0318, accept_pr_sem=0.075,
            accept_sw=3.6074, accept_sw_ci=(3.4916, 3.7444),
            tps_pr=77.763, tps_pr_sem=1.438,
            tps=71.551, tps_ci=(69.0234, 74.2306),
            vs_dflash_pr=0.9138, vs_dflash_pr_sem=0.014,
            vs_dflash=0.898, vs_dflash_ci=(0.8628, 0.931),
            vs_nospec=1.3141, nospec_tps=54.447),
    },
    "dspark-run-a-32k": {
        1: dict(
            accept_pr=6.9381, accept_pr_sem=0.1699,
            accept_sw=4.8065, accept_sw_ci=(4.5557, 5.1278),
            tps_pr=341.708, tps_pr_sem=8.273,
            tps=239.731, tps_ci=(227.5692, 255.0965),
            vs_dflash_pr=1.564, vs_dflash_pr_sem=0.0145,
            vs_dflash=1.2557, vs_dflash_ci=(1.2208, 1.2951),
            vs_nospec=3.8448, nospec_tps=62.352),
        2: dict(
            accept_pr=6.9345, accept_pr_sem=0.1684,
            accept_sw=4.8324, accept_sw_ci=(4.5371, 5.1803),
            tps_pr=314.484, tps_pr_sem=7.415,
            tps=224.126, tps_ci=(210.9219, 239.7517),
            vs_dflash_pr=1.5653, vs_dflash_pr_sem=0.017,
            vs_dflash=1.2758, vs_dflash_ci=(1.2333, 1.3183),
            vs_nospec=3.6205, nospec_tps=61.905),
        8: dict(
            accept_pr=6.9308, accept_pr_sem=0.1678,
            accept_sw=4.8147, accept_sw_ci=(4.5288, 5.1567),
            tps_pr=269.489, tps_pr_sem=6.325,
            tps=192.127, tps_ci=(181.1447, 205.279),
            vs_dflash_pr=1.566, vs_dflash_pr_sem=0.0145,
            vs_dflash=1.2974, vs_dflash_ci=(1.2611, 1.3376),
            vs_nospec=3.2111, nospec_tps=59.832),
        32: dict(
            accept_pr=6.9624, accept_pr_sem=0.1703,
            accept_sw=4.8237, accept_sw_ci=(4.5534, 5.1541),
            tps_pr=132.23, tps_pr_sem=3.044,
            tps=96.416, tps_ci=(91.389, 102.3929),
            vs_dflash_pr=1.578, vs_dflash_pr_sem=0.0166,
            vs_dflash=1.2101, vs_dflash_ci=(1.1789, 1.2432),
            vs_nospec=1.7708, nospec_tps=54.447),
    },
    "dspark-run-b-49k": {
        1: dict(
            accept_pr=6.919, accept_pr_sem=0.1699,
            accept_sw=4.7796, accept_sw_ci=(4.5197, 5.1),
            tps_pr=341.074, tps_pr_sem=8.251,
            tps=238.6, tps_ci=(226.0904, 254.1184),
            vs_dflash_pr=1.5597, vs_dflash_pr_sem=0.0144,
            vs_dflash=1.2498, vs_dflash_ci=(1.2138, 1.2892),
            vs_nospec=3.8267, nospec_tps=62.352),
        2: dict(
            accept_pr=6.9138, accept_pr_sem=0.1707,
            accept_sw=4.7371, accept_sw_ci=(4.4546, 5.0944),
            tps_pr=311.442, tps_pr_sem=7.405,
            tps=219.567, tps_ci=(207.1487, 235.3167),
            vs_dflash_pr=1.5606, vs_dflash_pr_sem=0.0142,
            vs_dflash=1.2499, vs_dflash_ci=(1.212, 1.2957),
            vs_nospec=3.5468, nospec_tps=61.905),
        8: dict(
            accept_pr=6.9627, accept_pr_sem=0.1689,
            accept_sw=4.8119, accept_sw_ci=(4.5234, 5.158),
            tps_pr=271.081, tps_pr_sem=6.355,
            tps=192.182, tps_ci=(181.2289, 205.588),
            vs_dflash_pr=1.5732, vs_dflash_pr_sem=0.0154,
            vs_dflash=1.2978, vs_dflash_ci=(1.2495, 1.3474),
            vs_nospec=3.212, nospec_tps=59.832),
        32: dict(
            accept_pr=6.929, accept_pr_sem=0.1695,
            accept_sw=4.707, accept_sw_ci=(4.4195, 5.0554),
            tps_pr=135.695, tps_pr_sem=3.094,
            tps=97.434, tps_ci=(91.8515, 104.1565),
            vs_dflash_pr=1.5704, vs_dflash_pr_sem=0.0148,
            vs_dflash=1.2228, vs_dflash_ci=(1.1765, 1.2711),
            vs_nospec=1.7895, nospec_tps=54.447),
    },
}

METRICS = {
    "accept": dict(
        default_value="per_turn",
        per_turn=dict(field="accept_pr", sem="accept_pr_sem", ci=None,
                      note="per-turn mean acceptance, "),
        pooled=dict(field="accept_sw", sem=None, ci="accept_sw_ci",
                    note="step-weighted, 1 + \u03a3accepted/\u03a3steps; "
                         "error bars = 95% paired bootstrap CI"),
        ylabel="Acceptance length (tokens per decode step)",
        title="Acceptance length vs concurrency",
        fmt="{:.2f}", ratio=False),
    "tps": dict(
        # Harmonic, not arithmetic.  `pooled` equals the token-weighted
        # HARMONIC mean of the per-call rates, exactly -- the correct mean for
        # a rate.  The arithmetic per-turn mean runs ~33% high because
        # corr(completion_tokens, per-call rate) is -0.35 to -0.42.
        default_value="pooled",
        per_turn=dict(field="tps_pr", sem="tps_pr_sem", ci=None,
                      note="ARITHMETIC per-turn mean of per-call rates "
                           "(runs ~33% high)"),
        pooled=dict(field="tps", sem=None, ci="tps_ci",
                    note="token-weighted harmonic mean\n"
                         "(pooled \u03a3output tokens / \u03a3decode time)"),
        ylabel="Decode throughput (output tokens / s)",
        title="Decode throughput vs concurrency",
        fmt="{:.0f}", ratio=False),
    "vs_dflash": dict(
        default_value="per_turn",
        per_turn=dict(field="vs_dflash_pr", sem="vs_dflash_pr_sem", ci=None,
                      note="mean acceptance / baseline DFlash"),
        pooled=dict(field="vs_dflash", sem=None, ci="vs_dflash_ci",
                    note="pooled tok/s / baseline DFlash"),
        ylabel="Decode throughput relative to DFlash (official)",
        title="Speed-up over the DFlash baseline vs concurrency",
        fmt="{:.2f}x", ratio=True),
    "vs_nospec": dict(
        default_value="pooled",
        per_turn=None,
        pooled=dict(field="vs_nospec", sem=None, ci=None,
                    note="pooled tok/s \u00f7 the no-spec control's; POINT "
                         "ESTIMATE ONLY -- the control cannot be paired or "
                         "resampled (see module docstring)"),
        ylabel="Decode throughput relative to no speculation",
        title="Speed-up over autoregressive decoding vs concurrency",
        fmt="{:.2f}x", ratio=True),
}


def resolve_value(metric, value):
    if value == "auto":
        return METRICS[metric]["default_value"]
    if METRICS[metric].get(value) is None:
        raise SystemExit(f"--metric {metric} has no '{value}' variant "
                         f"(only {METRICS[metric]['default_value']}).")
    return value


def spec_of(metric, value):
    return METRICS[metric][resolve_value(metric, value)]


def series(key, metric, value):
    spec = spec_of(metric, value)
    vals = [CONC_DATA[key][c][spec["field"]] for c in CONCURRENCIES]
    if spec["sem"]:
        return vals, [CONC_DATA[key][c][spec["sem"]] for c in CONCURRENCIES]
    if spec["ci"]:
        lo, hi = [], []
        for c, v in zip(CONCURRENCIES, vals):
            a, b = CONC_DATA[key][c][spec["ci"]]
            lo.append(v - a)
            hi.append(b - v)
        return vals, [lo, hi]
    return vals, None


def make_chart(family, metric, value, ref, out_dir, dpi, ext, annotate):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    value = resolve_value(metric, value)
    spec = dict(spec_of(metric, value))
    spec.update(ylabel=METRICS[metric]["ylabel"], title=METRICS[metric]["title"],
                fmt=METRICS[metric]["fmt"], ratio=METRICS[metric]["ratio"])
    keys = FAMILIES[family]
    if metric == "vs_dflash":
        keys = [k for k in keys if k != "dflash-official"]

    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    x = list(range(len(CONCURRENCIES)))
    # Series converge at high concurrency, so labels are staggered vertically
    # rather than all sitting at +8pt on top of each other.
    offsets = [(0, 12), (0, -26), (0, 36), (0, -50), (0, 58), (0, -72), (0, 80)]

    for si, k in enumerate(keys):
        vals, yerr = series(k, metric, value)
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

    # staggered labels sit up to ~80pt off their point; give them room
    ax.margins(y=0.16)
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
    caption = (f"{spec['note'].format(n=N_CALLS)}.{tail}\n"
               f"Greedy, NUM_SPEC_TOKENS=15, coding tasks replay.")
    fig.text(0.5, 0.02, caption,
             ha="center", va="bottom", fontsize=8.5, color="#555555")

    # reserve bottom space for however many lines the caption actually has
    fig.tight_layout(rect=(0, 0.055 + 0.031 * caption.count("\n"), 1, 1))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"slide_{family}_line_{metric}_{value}.{ext}")
    fig.savefig(path, dpi=dpi, transparent=True)
    plt.close(fig)
    return path


def table(family, metric, value):
    value = resolve_value(metric, value)
    spec = spec_of(metric, value)
    fmt = METRICS[metric]["fmt"]
    keys = FAMILIES[family]
    head = "| drafter | " + " | ".join(f"c{c}" for c in CONCURRENCIES) + " |"
    sep = "|---" * (len(CONCURRENCIES) + 1) + "|"
    rows = []
    for k in keys:
        vals, _ = series(k, metric, value)
        cells = []
        for c, v in zip(CONCURRENCIES, vals):
            cell = fmt.format(v)
            if spec["ci"]:
                a, b = CONC_DATA[k][c][spec["ci"]]
                cell += f" [{a:.3g}, {b:.3g}]"
            elif spec["sem"]:
                cell += f" ±{CONC_DATA[k][c][spec['sem']]:.3g}"
            cells.append(cell)
        rows.append(f"| {LABELS[k]} | " + " | ".join(cells) + " |")
    ns = "| *no speculation* | " + " | ".join(
        f"{CONC_DATA['dflash-official'][c]['nospec_tps']:.1f}" for c in CONCURRENCIES) + " |"
    out = [f"**{METRICS[metric]['title']}** ({value}) -- "
           f"{spec['note'].format(n=N_CALLS)}", "", head, sep] + rows
    if metric == "tps":
        out.append(ns)
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", choices=sorted(FAMILIES), default="dspark")
    p.add_argument("--metric", choices=sorted(METRICS) + ["all"], default="all")
    p.add_argument("--value", choices=["auto", "per_turn", "pooled"], default="auto",
                   help="auto = per_turn for accept/tps/vs_dflash, pooled for "
                        "vs_nospec (which has no per-turn variant)")
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
        print(table(a.family, m, a.value))
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
        print("wrote", make_chart(a.family, m, a.value, ref, a.out_dir,
                                  a.dpi, a.ext, a.annotate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
