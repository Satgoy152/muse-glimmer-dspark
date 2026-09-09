# Deck figures

Self-contained. Every number is hard-coded in the script, so nothing here reads
`results/`, the GPU node, or any other module — edit the `DATA` dict and re-run.

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
