# Deck figures

Self-contained. Every number is hard-coded in the script, so nothing here reads
`results/`, the GPU node, or any other module — edit the `DATA` dict and re-run.

Three scripts:

* `slide_bars.py` — one concurrency (1), acceptance and throughput per drafter.
* `slide_lines.py` — the same drafters across concurrency 1 / 2 / 8 / 32, plus a
  speed-up-vs-DFlash chart.
* `slide_profile.py` — the DFlash2 Nsight profile: where GPU time goes, and why
  acceptance rather than drafter cost is the lever.

## Commands

```bash
# DSpark family (slide 9): acceptance + throughput bars, plus the delta table
uv run --with matplotlib python benchmark/plots/slide_bars.py --family dspark

# DFlash2 family (slides 13-14)
uv run --with matplotlib python benchmark/plots/slide_bars.py --family dflash2

# All seven drafters on one chart
uv run --with matplotlib python benchmark/plots/slide_bars.py --family all

# The step-weighted acceptance view (the one that maps to deployment throughput)
uv run --with matplotlib python benchmark/plots/slide_bars.py \
    --family dspark --metric accept --value pooled

# Table only, no matplotlib needed
uv run python benchmark/plots/slide_bars.py --family dspark --table-only
```

Useful flags: `--ref <drafter-key>` changes what the `%` labels compare against,
`--metric accept|tokens|both`, `--ext png|pdf|svg`, `--dpi`, `--out-dir`.
Figures land in `benchmark/plots/out/` by default (git-ignored).

## What the numbers are

Terminal-Bench bucket x concurrency sweep, greedy, `NUM_SPEC_TOKENS=15`,
**concurrency 1**, output bucket **64-256** (280 calls per drafter). The tok/s
are measured on timed serial replays, not the normalised
`accept / t_step` estimates in `docs/RESULTS-turn-length.md`.

Bucket membership comes from the **original frozen recording's**
`completion_tokens`, so nothing selects on the replayed outcome.

Two aggregations are carried because they disagree:

| `--value` | acceptance | throughput | error bars |
|---|---|---|---|
| `per_request` | mean over the 280 calls | mean of per-call rates | SEM |
| `pooled` | `1 + Σaccepted / Σsteps` | `Σgen / Σ(steps × t_step)` | 95% bootstrap CI over calls (4,000 resamples, seed 20260830) |

`--value auto` (the default) picks `per_request` for `--metric accept` and
`pooled` for `--metric tokens`.

### Why throughput is pooled, not averaged over turns

Both are equally *measured* — same per-call counters, different weighting. For a
rate the weighting is not a free choice:

* `pooled` equals the **token-weighted harmonic mean** of the per-call rates,
  exactly, in every row. So it already is a per-turn average of per-turn rates —
  the harmonic one, which is the correct mean for a rate. (One mile at 20 mph
  and one at 60 mph averages 30 mph, not 40.)
* `per_request` is the **arithmetic** mean. Here `corr(completion_tokens,
  per-call rate)` is −0.35 to −0.42 for every DFlash/DSpark drafter — longer
  turns decode slower — so it weights a 64-token turn the same as a 256-token
  turn occupying 4x the wall-clock. That is the whole 334-vs-252 gap.
* It is also the only summary under which our DSpark fine-tune beats DFlash
  official on speed: arithmetic mean 333.6 vs 341.7 (ours +2.4%), median 324.9
  vs 316.6 (ours −2.6%), pooled 251.9 vs 239.7 (ours −4.8%).
* A per-call rate is just `accept_len_i / t_step`, and `t_step` moves under 0.7%
  at concurrency 1 — so the per-request throughput chart is the per-request
  acceptance chart with a rescaled y-axis, carrying no independent information.

For a genuinely per-turn speed statistic use the **median with an IQR**, not the
mean. Those are in `DATA` as `tps_med` / `tps_p25` / `tps_p75`:

```bash
uv run python benchmark/plots/slide_bars.py --list-dist
```

**Caveat that must stay attached:** the `64-256` bucket is 60.35% of
Terminal-Bench calls but only 24.19% of its decoded tokens. Per-request bars
flatter the fine-tunes relative to what a deployment sees; the step-weighted
numbers are much closer between drafters. Show both, or state which one is on
the slide.


# Line charts (`slide_lines.py`)

```bash
# all four line charts for the DSpark family: tps, accept_sw, vs_dflash, vs_nospec
uv run --with matplotlib python benchmark/plots/slide_lines.py --family dspark

# just the speed-up-over-DFlash chart
uv run --with matplotlib python benchmark/plots/slide_lines.py \
    --family dspark --metric vs_dflash

# DFlash2 family
uv run --with matplotlib python benchmark/plots/slide_lines.py --family dflash2

# tables only, no matplotlib
uv run python benchmark/plots/slide_lines.py --family all --table-only
```

`--annotate endpoints` (the default) labels only c1 and c32; `value` labels every
point and `none` labels nothing. The lines converge above c8, so `value` is only
readable on the ratio charts.

## Families

`--family best` is the deck chart: the three released baselines plus the best
checkpoint from each of our two families, instead of four near-identical
fine-tune lines. **"Best" is by point estimate only** — DSpark 32K vs 49K
(239.7 vs 238.6 pooled tok/s at c1) and DFlash2 final vs mid (265.9 vs 259.7)
each sit inside the other's bootstrap interval, so the pick is a presentation
choice, not a measured ranking. `--family all` shows every checkpoint.

## Metrics and `--value`

| `--metric` | `per_turn` | `pooled` |
|---|---|---|
| `accept` | mean over turns ± SEM | `1 + Σaccepted/Σsteps` ± bootstrap CI |
| `tps` | mean of per-call rates ± SEM | `Σgen / Σ(steps × t_step)` ± bootstrap CI |
| `vs_dflash` | mean over turns of that turn's rate ÷ DFlash's rate **on the same call** ± SEM | pooled ratio ± paired bootstrap CI |
| `vs_nospec` | — | pooled ratio, **point estimate only** |

`--value auto` (default) uses `per_turn` everywhere except `vs_nospec`, which
has no per-turn variant.

**The two weightings disagree in sign on `vs_dflash`, and both are real.** Our
DSpark 32K is 1.02x DFlash official per turn and 0.95x pooled at c1. Per turn it
wins the typical call; pooled it loses, because the fine-tune's acceptance gain
sits on short turns that carry few tokens (per-turn acceptance 6.94 vs 6.44,
step-weighted 4.81 vs 4.82). Quote whichever you mean and label it — do not mix
them in one claim.

The bootstrap is **paired**: every drafter replayed the byte-identical 280 calls
(verified — the per-call keys intersect 280/280), so one resample of call indices
is applied to every drafter and to both halves of a ratio. That is why the
`vs_dflash` intervals are far tighter than dividing two independent intervals
would give.

`vs_nospec` gets no interval: the no-speculation control ran as its own job with
its own call keys and reports `steps == 0` per request, so it cannot be paired or
resampled. Its throughput comes from the server counters and is a constant here.

**`accept_sw` is expected to be four flat lines.** Acceptance is a property of
the drafter and the prompt; the batch does not touch it. What moves with
concurrency is `t_step`. That flatness is the result, not a broken plot.


# Profiling figures (`slide_profile.py`)

```bash
# all three
uv run --with matplotlib python benchmark/plots/slide_profile.py

# just the stacked time split
uv run --with matplotlib python benchmark/plots/slide_profile.py --chart phases

# tables only, no matplotlib
uv run python benchmark/plots/slide_profile.py --table-only
```

Source: `docs/RESULTS-profiling-dflash2.md` and
`benchmark/profiling/results/analysis/*.json`. Nsight Systems, 50-step
steady-state decode window, four cases (short/long prompts x concurrency 1/10).

| `--chart` | what it shows | status |
|---|---|---|
| `phases` | GPU kernel time by phase; the whole drafter is 12.4-13.1% | **measured** |
| `prompts` | three prompts: step time flat within 3%, throughput spans 6.1x | **measured** |
| `headroom` | scenario arithmetic on `tok/s = accept_len / t_step` | **derived** |

The argument the three make together:

1. Target verification is 82.9-83.7% of GPU time and the entire drafter is ~13%,
   so **cost-side work on the drafter has at most ~15% to recover** even if you
   deleted it completely.
2. On the same server, at the same step cost (19.3-19.8 ms, a 3% spread),
   throughput ranges 106-645 tok/s across three prompts. **All of that spread is
   accepted length.**
3. Our DFlash2 fine-tune's measured acceptance gain is worth +16.6% throughput —
   more than deleting the entire drafter, and it cost no extra step time
   (`t_step` 19.397 ms vs native DFlash2's 19.386).

Caveats that must stay attached:

* This is a **speed profile, not a quality evaluation**. Nothing in it measures
  whether the drafter proposes good tokens.
* `ignore_eos=true` inflates accepted length — forcing generation past the
  natural stop drives repetition, which drafts almost perfectly. That is the
  12.97. Those accepted lengths are an **upper bound**, not a quality result.
  The flat-step-time point does not depend on the inflation.
* The `headroom` chart is **arithmetic, not measurement**. Its two grey bars are
  unreachable bounds, not proposals, and it assumes step time scales with GPU
  kernel time (GPU busy is 96.9% here).
* NVTX inside a CUDA graph's capture does not replay, so there is no subphase
  attribution *within* either full graph.
