# Paired re-analysis of the runs already on disk

No new GPU time: this is the per-call metrics blocks from the traces in
`/mnt/data/traces/tb-*`, re-pooled two ways instead of one.

`accept_len` has been quoted **token-weighted** ("micro"): pool accepted and
steps over all calls, then divide. That is the throughput number. It is not the
only one — the **per-request** ("macro") mean, the average call's acceptance
length, answers a different and equally deployable question, and the two
disagree here.

Every comparison below is paired on `replay_of` (byte-identical prompt) and is
reported next to two controls: `dflash2` replayed against itself, and
`dflash-official` against `dflash2`.

## Per-request acceptance, Terminal-Bench replay, concurrency 10

| A vs B | micro A/B | macro A/B | macro Δ (95% CI) | call win rate | sign-test p |
|---|---|---|---|---|---|
| **ours `dflash2-run-d-step1976`** vs `dflash2` | 3.90 / 3.99 | **6.27 / 5.61** | **+0.659 ±0.105** | 59.3% | 5e-14 |
| **ours `dflash2-run-d-step1976`** vs `dflash-official` | 3.90 / 3.84 | **6.29 / 5.65** | **+0.638 ±0.098** | 61.3% | 7e-20 |
| **ours `dspark-run-a-32k`** vs `dflash-official` | 3.81 / 3.85 | **6.21 / 5.65** | **+0.559 ±0.101** | 58.5% | 7e-12 |
| **ours `dspark-run-a-32k`** vs `dspark-community` | 3.81 / 3.18 | **6.21 / 3.53** | **+2.674 ±0.128** | 89.1% | 1e-223 |
| *control* `dflash2` vs itself | 4.01 / 4.00 | 5.68 / 5.64 | +0.039 ±0.080 | 51.9% | 0.14 |
| *control* `dflash-official` vs `dflash2` | 3.85 / 3.99 | 5.72 / 5.67 | +0.045 ±0.089 | 50.4% | 0.73 |

Calls with fewer than 5 spec steps are excluded from the macro statistics on
both sides. Reproduces at concurrency 64: ours `dflash-run-d-mid` vs `dflash2`
macro +0.756 ±0.103, `dspark-run-a-32k` vs `dspark-community` +2.653 ±0.129,
baseline-vs-baseline +0.063 ±0.085.

**This is not a completion-length artifact.** The five runs draw near-identical
completion-length distributions (median 123–128 tokens, 58–61% of calls in
64–256), and restricting to calls where both sides land in the *same* length
bucket makes the gap larger, not smaller (+0.826 ±0.114 for `dflash2-run-d`
vs `dflash2`, control +0.038 ±0.086).

## Where the difference sits

Both sides restricted to the same completion-length bucket, so bucket
membership carries no selection bias.

**ours `dflash2-run-d-step1976` vs `dflash2`**

| completion tokens | calls | micro Δ | macro Δ | win rate |
|---|---:|---:|---:|---:|
| <64 | 81 | **+2.72** | +2.60 | 93.8% |
| 64–256 | 832 | **+0.35** | +0.94 | 67.9% |
| 256–1K | 266 | −0.03 | +0.02 | 44.0% |
| 1K–4K | 25 | −0.17 | −0.17 | 32.0% |

**ours `dspark-run-a-32k` vs `dflash-official`**

| completion tokens | calls | micro Δ | macro Δ | win rate |
|---|---:|---:|---:|---:|
| <64 | 93 | **+2.09** | +2.13 | 84.9% |
| 64–256 | 834 | **+0.32** | +0.69 | 64.4% |
| 256–1K | 261 | +0.01 | −0.02 | 44.4% |
| 1K–4K | 30 | −0.30 | −0.24 | 23.3% |

*control* `dflash2` vs itself, same table: −0.01 / +0.02 / +0.11 / −0.02 macro,
win rate 27–49%.

70% of the agent's turns are under 256 completion tokens. The fine-tunes win
those and lose the small number of long generations, which is why the
token-weighted pool moves the other way.

Bucketing by *prompt* length instead (identical across runs, so no bias at all)
shows the macro gain is flat in context: +0.60 to +0.67 in every bucket from
2K to 64K for `dflash2-run-d` vs `dflash2`, control ±0.17.

## What the token-weighted number can actually resolve

Paired bootstrap over calls, 2,000 resamples:

| pair | micro Δ | 95% CI |
|---|---|---|
| ours `dspark-run-a-32k` vs `dspark-community` | +0.623 | [+0.536, +0.722] |
| ours `dflash2-run-d-step1976` vs `dflash-official` | +0.061 | [−0.049, +0.162] |
| ours `dflash2-run-d-step1976` vs `dflash2` | −0.093 | [−0.210, +0.026] |
| *control* `dflash2` vs itself | +0.010 | [−0.101, +0.120] |

Only the DSpark fine-tune's gain clears the interval. The "0.005 noise floor" in
`RESULTS.md` is a single replay-vs-replay point estimate, not a confidence
interval; the CI is ±0.11. On the token-weighted metric **our DFlash2 fine-tune
is not separable from native `dflash2`**, and the earlier "0.091 below, ~17x the
noise floor" claim overstates what one repeat can support.

## Coding benchmarks: greedy, so the pairing is exact

Greedy decoding makes 499/500 MBPP and 163/164 HumanEval completions
byte-identical between drafters, so the per-problem pairing has essentially no
sampling noise.

| pair | HumanEval micro Δ | per-problem win | MBPP micro Δ | per-problem win |
|---|---|---|---|---|
| ours `dflash2-run-d` vs `dflash-official` | **+0.446** | **157/164 (96%)** | **+0.251** | **449/500 (90%)** |
| ours `dflash2-run-d` vs `dflash2` | −0.449 | 1/164 | −0.383 | 2/500 |
| ours `dspark-run-a-32k` vs `dspark-community` | −0.118 | 24/164 | −0.088 | 88/500 |

Our DFlash2 fine-tune beats the **official** DFlash drafter on both coding
benchmarks, off its training distribution, at a per-problem win rate that leaves
no room for noise.

## Draft-length sweep, simulated from the acceptance histograms

`accept_len(K) = 1 + mean(min(i, K))` over the recorded accepted-length
histogram. Exact at K=15 by construction; an estimate below it, since a shorter
draft shifts where later steps start.

| drafter | K=1 | K=3 | K=5 | K=7 | K=11 | K=15 |
|---|---|---|---|---|---|---|
| `dflash2` | 1.736 | 2.631 | 3.108 | 3.406 | 3.765 | 4.002 |
| `dflash-official` | 1.745 | 2.615 | 3.067 | 3.345 | 3.674 | 3.855 |
| ours `dflash2-run-d-step1976` | 1.733 | 2.583 | 3.013 | 3.281 | 3.643 | 3.909 |
| ours `dspark-run-a-32k` | 1.714 | 2.534 | 2.953 | 3.216 | 3.563 | 3.804 |
| `dspark-community` | 1.675 | 2.419 | 2.754 | 2.931 | 3.106 | 3.185 |

No crossover: nothing wins at a shorter draft that does not win at 15, so there
is no draft-budget setting that flips the ranking. Going the other way is not
available either — every checkpoint has `block_size` 16 baked into a
mask-token block, so K > 15 needs retraining.

What the sweep does show is that the fine-tune fixed a **saturating** drafter:
`dspark-community` gains only +0.43 from K=5 to K=15 while `dspark-run-a-32k`
gains +0.85, and the fraction of steps that accept the whole 15-token draft goes
from 1.3% to 5.2%. The community checkpoint was wasting most of the block.

# Throughput and TPOT, and how they are normalised

`RESULTS.md` says client TTFT/TPOT are invalid (the tool-call parser buffers
deltas) and that vLLM's ITL/TPOT *histogram buckets* are too coarse to separate
drafters. The histogram **`_sum`/`_count` counters** are neither. They are
accumulated inside the engine, so the parser cannot touch them, and a sum over
a whole run has no bucket resolution to lose.

## The identity everything normalises through

```
vllm:inter_token_latency_seconds_count  ==  number of speculative steps
vllm:inter_token_latency_seconds_sum    ==  total decode wall-clock

t_step   = inter_token_latency_sum / inter_token_latency_count      (ms per engine decode iteration)
TPOT     = request_decode_time_sum / (generation_tokens_sum - n)    == t_step / accept_len
tok/s    = generation_tokens_sum / request_decode_time_sum          == accept_len / t_step
```

So speed factors exactly into **how many tokens a step emits** (`accept_len`,
the drafter's quality) and **how long a step takes** (`t_step`, the drafter's
cost). Prefill is excluded by construction — `request_decode_time_seconds`
starts after the first token — which also removes the prefix-cache hit rate
from the comparison.

`t_step` is a function of the drafter *and the batch*, so a throughput number is
only comparable at a fixed concurrency. It is not a function of the workload:

| drafter | HumanEval | MBPP | 8-domain | spread |
|---|---|---|---|---|
| `dflash2` | 19.153 | 19.164 | 19.131 | **0.17%** |
| ours `dflash2-run-d-step1976` | 19.155 | 19.156 | — | 0.01% |
| `dflash-official` | 18.800 | 18.811 | — | 0.06% |
| ours `dspark-run-a-32k` | 19.472 | 19.471 | — | 0.01% |

Three different prompt sets, three different output-length distributions, same
`t_step` to a fraction of a percent. That is what makes a concurrency-1
throughput comparison trustworthy.

At concurrency 10 it is not. The same drafter replayed twice moves `t_step` by
**5.8%** (`dflash2` 35.418 vs 37.478 ms) and the scale-fix control moves the
same checkpoint by 3.3% (36.581 vs 35.414) at identical acceptance. So the
tok/s column on the Terminal-Bench replay table cannot resolve anything under
~6%, and acceptance length remains the only usable number there.

## Concurrency 1, greedy: the fully normalised comparison

Greedy decoding makes the completions byte-identical between drafters, so this
is the same token sequence produced at different speeds — nothing left to
normalise away.

**HumanEval (164 prompts)**

| drafter | accept_len | t_step ms | TPOT ms | decode tok/s |
|---|---:|---:|---:|---:|
| `dflash2` | 5.748 | 19.153 | 3.336 | **300.0** |
| ours `dflash2-run-d-step1976` | 5.299 | 19.155 | 3.619 | **276.5** |
| `dflash-official` | 4.853 | 18.800 | 3.877 | 258.1 |
| `dspark-community` | 4.658 | 19.470 | 4.190 | 238.9 |
| ours `dspark-run-a-32k` | 4.541 | 19.472 | 4.301 | 232.7 |

**MBPP (500 prompts)**

| drafter | accept_len | t_step ms | TPOT ms | decode tok/s |
|---|---:|---:|---:|---:|
| `dflash2` | 5.436 | 19.164 | 3.530 | **283.5** |
| ours `dflash2-run-d-step1976` | 5.054 | 19.156 | 3.796 | **263.7** |
| `dflash-official` | 4.803 | 18.811 | 3.921 | 255.2 |
| `dspark-community` | 4.528 | 19.483 | 4.313 | 232.0 |
| ours `dspark-run-a-32k` | 4.440 | 19.471 | 4.397 | 227.6 |

Our DFlash2 fine-tune is **+7.1% tok/s over the official DFlash drafter** on
HumanEval and **+3.3%** on MBPP, at a `t_step` within 2% of it — the gain is
acceptance, not a cheaper drafter.

**The fine-tune does not make the drafter more expensive.** `t_step` for
`dflash2-run-d-step1976` (19.155) matches native `dflash2` (19.153) to 0.01%,
and `dspark-run-a-32k` (19.472) matches `dspark-community` (19.470). Every
difference in the tables above is the acceptance term.

## Concurrency 1, Terminal-Bench subset: speedup over no speculation

100 stratified calls from the frozen set, replayed serially, with a
target-only server as the denominator. This is the multiple the deck should
quote; it did not exist before.

| drafter | accept_len | t_step ms | TPOT ms | decode tok/s | vs no-spec |
|---|---:|---:|---:|---:|---:|
| *no speculation* | 1.000 | 16.357 | 16.357 | 61.3 | 1.00x |
| `dflash-official` | 4.137 | 19.374 | 4.713 | 213.0 | **3.47x** |
| ours `dspark-run-a-32k` | 3.892 | 20.024 | 5.193 | 193.2 | **3.15x** |
| `dspark-community` | 3.197 | 20.000 | 6.263 | 160.2 | 2.61x |

`dflash2` and our DFlash2 fine-tune are queued — the first attempt passed
`method: "dflash2"`, which vLLM rejects; DFlash2 checkpoints are served as
`dflash` and routed to the V2 runner by their `architectures` field.

**The DSpark fine-tune is worth +20.6% decode throughput over the checkpoint it
started from** — 193.2 vs 160.2 tok/s, 2.61x to 3.15x over autoregressive.

Subtracting the no-spec step gives the drafter's own cost per step: **3.02 ms
for DFlash, 3.65 ms for DSpark** on top of a 16.36 ms target step. DSpark's
drafter is ~21% more expensive to run, a fixed 3.3% TPOT handicap that its
acceptance has to pay back before it can match DFlash — `dspark-run-a-32k`
needs `accept_len` ≥ 4.27 to tie `dflash-official`'s 4.14, and reaches 3.89.
That, not the fine-tune, is why the DSpark family trails.

# What the turn-length theory turned out to be

The per-request win sits on short turns, so the obvious hypothesis was that the
fine-tune saw shorter turns than the eval decodes. Measured, that is not the
lever.

## Turn length is not the difference

Turn length, from the recorded calls on both sides (`completion_tokens`, the
same field the eval quotes):

| | training (SWE-Gym) | eval (Terminal-Bench) |
|---|---|---|
| median turn | 93 tok | 129 tok |
| mean / p90 | 178 / 405 | 297 / 598 |
| share of turns 64-256 | 69.1% | 60.3% |
| share of turns >=1K | 0.9% | 4.8% |

Reweighting the 1,753 TB calls to the training mix (same prompts, same runs,
weights 1.18 / 1.15 / 0.73 / 0.18 by output-length bucket) lifts every drafter
by ~+0.16 and **changes no ranking**:

| drafter | accept as measured | reweighted | TPOT ms | tok/s |
|---|---|---|---|---|
| `dflash2` | 4.002 | 4.159 | 4.606 | 217.1 |
| ours `dflash2-run-d` | 3.909 | 4.090 | 4.683 | 213.5 |
| `dflash-official` | 3.855 | 4.020 | 4.819 | 207.5 |
| ours `dspark-run-a` | 3.804 | 3.999 | 5.007 | 199.7 |
| `dspark-community` | 3.185 | 3.188 | 6.274 | 159.4 |

Paired bootstrap on the reweighted value: ours `dflash2-run-d` vs `dflash2`
-0.071 [-0.193, +0.057]; vs `dflash-official` +0.073 [-0.043, +0.192]; control
+0.043. The earlier +0.35 in the 64-256 bucket was conditional on both drafters
drawing a similar-length completion for that prompt, which a deployment cannot
condition on.

## Reasoning share is the difference

Every turn is `reasoning` + a `bash` tool call; `content` is empty on 99.5% of
them because the tool parser routes the whole message into `tool_calls`.
`completion_tokens` counts all of it, and vLLM reports
`completion_tokens_details.reasoning_tokens` as 0 on 100% of turns, so the split
comes from the parsed fields.

At **fixed** output length, acceptance falls 1.5x across the reasoning axis --
more than it falls along the length axis at fixed reasoning share. `dflash2`,
absolute accept_len:

| | Q1 least reasoning | Q2 | Q3 | Q4 most reasoning |
|---|---|---|---|---|
| out 64-256 | 5.719 | 4.334 | 3.945 | 3.704 |
| out 256-1K | 5.571 | 4.814 | 3.791 | 3.583 |
| out >=1K | — | — | 3.780 | 3.284 |

The bash command is patterned and drafts well; free-form reasoning prose does
not. And the fine-tune's gain is concentrated in the command-heavy half, split
at the median reasoning share inside each length bucket:

| ours `dflash2-run-d` vs `dflash2` | low reasoning | high reasoning |
|---|---|---|
| out <64 | +2.600 (n=71) | — |
| out 64-256 | **+0.663** (n=426) | **+0.214** (n=406) |
| out 256-1K | +0.035 (n=101) | -0.056 (n=165) |
| *control, same cells* | -0.03 / +0.01 / -0.32 | +0.07 / +0.07 |

Which is exactly where the corpora differ:

| | reasoning share of generated chars | turns with no reasoning at all |
|---|---|---|
| training (SWE-Gym) | 56.8% | 11.6% |
| eval (Terminal-Bench) | **75.3%** | 2.2% |

At the same nominal `reasoning_strength` the model reasons 74 -> 87 median chars
on SWE-Gym and 88 -> 294 on Terminal-Bench: the strength knob barely moves it on
formulaic issue-fixing, and moves it a lot on open-ended terminal tasks. That is
a task property, not a sampling weight.

So the fine-tune improved the half of the output that was already easy, and 75%
of what the eval decodes is the half it did not improve. The retrain target is
reasoning-token share -- harder tasks that make the model actually reason, or a
loss weighting toward reasoning positions -- not turn length.

Shares are of characters in the parsed fields, not tokens, and the parsed fields
do not recover structural tokens. Chars per `completion_token` is 2.50 on the
eval against 2.06 in training, so the eval's reasoning share is if anything
higher in token terms than 75.3%.

## Concurrency 1, 100 TB calls, with the no-spec control

| drafter | accept_len | t_step ms | TPOT ms | decode tok/s | vs no-spec |
|---|---:|---:|---:|---:|---:|
| *no speculation* | 1.000 | 16.357 | 16.357 | 61.3 | 1.00x |
| ours `dflash2-run-d-step1976` | 4.223 | 19.704 | 4.692 | 213.8 | 3.49x |
| `dflash-official` | 4.137 | 19.374 | 4.713 | 213.0 | 3.47x |
| `dflash2` | 4.169 | 19.709 | 4.743 | 211.6 | 3.45x |
| ours `dspark-run-a-32k` | 3.892 | 20.024 | 5.193 | 193.2 | 3.15x |
| `dspark-community` | 3.197 | 20.000 | 6.263 | 160.2 | 2.61x |

**Do not quote the ordering of the top four.** At n=100 and temperature 1.0 the
CI on accept_len is roughly +-0.45; they are one cluster. What is solid is the
drafter cost over the 16.357 ms bare target step: **+3.02 ms DFlash, +3.35 ms
DFlash2, +3.65 ms DSpark**, and each fine-tune matches its own base to within
0.03 ms. The DSpark family's 0.63 ms handicap against DFlash is a fixed 3.3% of
TPOT that acceptance has to pay back before it can win.

The DSpark fine-tune's gain over its warm start is the one that clears
everything: **160.2 -> 193.2 tok/s, 2.61x -> 3.15x over autoregressive.**

## Domain breakdown (partial)

`RedHatAI/speculator_benchmarks`, greedy, concurrency 1, so every drafter emits
identical tokens. Two and a half drafters in before the run was stopped for the
holdout eval; `question` and `translation` are missing for our checkpoint.

| accept_len / tok-s | tool_call | qa | rag | summ. | writing | math |
|---|---|---|---|---|---|---|
| `dflash2` | 4.44 / 232 | 3.64 / 190 | 9.44 / 493 | 5.47 / 286 | 4.17 / 218 | 6.00 / 313 |
| ours `dflash2-run-d` | 4.32 / 225 | 3.49 / 182 | 9.19 / 480 | 5.27 / 275 | 4.09 / 213 | 5.81 / 303 |
| `dflash-official` | 4.11 / 219 | 3.26 / 173 | 8.72 / 464 | 4.77 / 254 | 3.71 / 197 | 5.31 / 282 |

Ours is above the official DFlash drafter and below native `dflash2` on every
domain, with no flip anywhere. Note the median output on the `tool_call` split
is 484 tokens and 388-742 across the others: Muse Glimmer reasons at length
before emitting a function call, so "tool call" is not a short-generation
regime for this model. Terminal-Bench's 129-token median is the shortest
workload measured.
