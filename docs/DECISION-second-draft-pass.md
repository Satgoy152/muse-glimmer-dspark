# Would a second draft forward pass pay for itself?

Question: DFlash2 drafts the 16-token block in one forward pass. Would running
a second pass (a refinement/denoising step) be worth it if it raised acceptance?

Answer from the 2026-09-08 profile: **it needs a >12% relative gain in accepted
length to break even.** That is a low bar as speculative-decoding trades go, and
the reason is structural -- the drafter is cheap relative to the target.

## The arithmetic

Throughput per engine step is `accept_len / step_time`. A second draft pass
leaves the target side untouched (the block size does not change), so it adds
exactly the draft forward:

| case | draft forward, share of step | break-even multiplier |
|---|---|---|
| short, c1 | 12.08% | 1.121x |
| short, c10 | 12.11% | 1.121x |
| long, c1 | 12.00% | 1.120x |
| long, c10 | 11.52% | 1.115x |

The marginal cost is the draft *graph* only. Draft context-KV insertion (0.3%)
and feature projection (0.6%) are per-cycle setup over the verified tokens; a
refinement pass over the same block with context KV already resident does not
repeat them.

Accepted length required to break even, against a 16-token block:

| case | accept_len now | break-even | now, as % of block | needed |
|---|---|---|---|---|
| short, c1 | 6.03 | 6.76 | 37.7% | 42.2% |
| short, c10 | 6.04 | 6.77 | 37.8% | 42.3% |
| long, c1 | 8.23 | 9.22 | 51.4% | 57.6% |
| long, c10 | 7.94 | 8.86 | 49.6% | 55.4% |

Net throughput change for a given acceptance gain (all four cases agree within
half a point, so one row is enough):

| acceptance gain | +5% | +10% | +12% | +20% | +30% | +50% |
|---|---|---|---|---|---|---|
| net throughput | -6.3% | -1.9% | ~0 | +7.1% | +16.0% | +33.8% |

## Why the bar is this low, and what would raise it

The drafter is ~12% of GPU time against the target's ~83%. Doubling a component
that small costs little. Had the drafter been 40% of the step, break-even would
have been a +40% acceptance gain, which is a different proposition entirely.

The cost estimate is not optimistic. The second pass re-reads the drafter's
5.16 GiB of weights from HBM -- far beyond the 50 MB L2 -- so there is no
caching discount. And it is a sequential dependency on the first pass, so it
cannot overlap. At concurrency 1 the draft GEMMs are already skinny and
bandwidth-bound; a second pass does not improve their utilisation.

Two things would invalidate the estimate: a refinement pass with a different
shape or extra machinery (a second graph capture is fine, extra eager work is
not), and any change that grows the block, which would move cost onto the
target side where it is 7x more expensive.

## Where it is most likely to pay

Short prompts. They sit at 37.7% of the block versus 51.4% for long, so there is
more headroom before the 16-token ceiling compresses returns, and break-even
needs a smaller absolute gain (+0.72 tokens versus +0.99).

## How to measure it without fooling yourself

Accepted length varies **6x between prompts** (2.05 to 12.97 on short prompts at
concurrency 1) while step time holds constant within 3%. A 12% effect against
that spread will not survive an unpaired comparison.

Run one-pass and two-pass over the **same** prompt set and compare per prompt,
paired. The existing harness already supports this: the prompt set is fixed and
hashed, sampling is greedy, and `run_workload.py` records per-request
`num_spec_steps` and `num_accepted_draft_tokens`. Compare accept_len per prompt
id, then take the ratio distribution -- not the aggregate of one run against the
aggregate of the other.

Note also that `ignore_eos=true` inflates the accepted lengths above, because
forced continuation drives the model into repetition the drafter predicts almost
perfectly. Break-even is a *ratio*, so the 12% threshold is unaffected by that
inflation. What the inflation does affect is achievability: real accepted
lengths are lower, leaving more headroom to the ceiling, so these numbers if
anything understate the case for a second pass.

## The alternative lever the profile suggests

At concurrency 1 the target verification is a stack of N=16 GEMMs, several in
`splitK` variants -- the shape a kernel takes when it is too thin to fill the
GPU. That regime is bandwidth-bound on weight reads, so verifying a *larger*
block should cost sublinearly in the phase that dominates the step. Growing the
block is only useful if the drafter can fill it well, which is exactly what
extra refinement would buy. The two levers interact and are worth evaluating
together rather than separately.

This is a hypothesis from kernel shapes, not a measurement. Nothing here has
been tried.
