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

# The step-weighted view (the one that maps to deployment throughput)
uv run --with matplotlib python benchmark/plots/slide_bars.py \
    --family dspark --value step_weighted

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

Three aggregations are carried because they disagree:

| | what it is | error bars? |
|---|---|---|
| `per_request` | mean over the 280 calls, equal weight each | yes, SEM |
| `step_weighted` | `1 + Σaccepted / Σsteps` | no — pooled ratio |
| `aggregate` | `Σgen / Σ(steps × t_step)` | no — pooled ratio |

**Caveat that must stay attached:** the `64-256` bucket is 60.35% of
Terminal-Bench calls but only 24.19% of its decoded tokens. Per-request bars
flatter the fine-tunes relative to what a deployment sees; the step-weighted
numbers are much closer between drafters. Show both, or state which one is on
the slide.
