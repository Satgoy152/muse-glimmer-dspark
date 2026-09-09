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

**Do not plot throughput as `per_request`.** A mean of per-call rates is each
call's acceptance divided by `t_step`, so it inherits the same short-call bias
that makes per-request acceptance (6.44) exceed step-weighted acceptance (4.82):
6.44 / 19.047 ms = 338 tok/s against a real pooled 252 tok/s, ~33% high. The
`pooled` column is the one in the deck's Table B. The per-request variant is
kept only so both charts can be drawn on a matched definition, and the figure
labels itself when you do.

**Caveat that must stay attached:** the `64-256` bucket is 60.35% of
Terminal-Bench calls but only 24.19% of its decoded tokens. Per-request bars
flatter the fine-tunes relative to what a deployment sees; the step-weighted
numbers are much closer between drafters. Show both, or state which one is on
the slide.
