# Turn-length analysis of the drafter evaluations

This note consolidates the post-hoc turn-length analysis for the frozen
Terminal-Bench replay and HumanEval. It supplements, rather than replaces, the
full pooled results in [`RESULTS.md`](RESULTS.md).

## Definitions

- A **turn** is one model completion/API call.
- **Turn share** gives every call equal weight.
- **Token share** is the fraction of all decoded tokens contributed by a group
  of turns. Long turns therefore contribute much more token mass.
- **Per-turn acceptance** is the mean of each call's acceptance length. Every
  call receives equal weight.
- **Step-weighted acceptance** is `1 + sum(accepted draft tokens) / sum(decode
  steps)`. Long turns contain more decode steps, so this is the acceptance
  measure that factors into aggregate decode throughput.

For example, if a 10-step turn has acceptance 3 and a 100-step turn has
acceptance 5, the per-turn mean is 4.0 while the step-weighted value is 4.82.

## Correction to the original short-turn diagnosis

The earlier claim that 83.2% of the SWE-Gym training turns were shorter than 64
tokens was not a measurement of the full supervised turn. That exact number is
reproduced by tokenizing only `content` plus tool arguments. It excludes
`reasoning_content` and the rendered reasoning/tool protocol tokens, all of
which are supervised by the training template.

Using the recorded target-model `completion_tokens`, the 32K-eligible training
pool and Terminal-Bench have the following distributions:

| completion length | SWE-Gym 32K: turns | SWE-Gym 32K: tokens | TB: turns | TB: tokens |
|---|---:|---:|---:|---:|
| `<64` | 12.2% | 4.2% | 9.9% | 1.9% |
| `64-256` | 70.3% | 45.1% | 60.3% | 24.2% |
| `256-1K` | 16.7% | 44.6% | 25.0% | 39.4% |
| `1K-4K` | 0.7% | 6.1% | 4.4% | 27.8% |
| `>=4K` | 0.0% | 0.0% | 0.4% | 6.7% |

The SWE-Gym 32K-eligible median is 90 tokens and its mean is 168 tokens. The
training mismatch is therefore not that nearly every supervised turn is below
64 tokens. The clearer gap is the long tail: turns of at least 1K tokens supply
6.1% of SWE-Gym supervision but 34.6% of Terminal-Bench decode tokens.

Reasoning composition also differs. Reasoning accounts for 57.9% of generated
characters in the SWE-Gym 32K pool and 75.3% in Terminal-Bench. This is a
character-based decomposition of the recorded response's reasoning versus
content/tool arguments; it is not a tokenizer-level loss decomposition.

## Terminal-Bench: turns near the SWE-Gym median and mean

The filter below uses the completion length from the **original frozen
Terminal-Bench recording**, before any drafter replay. It does not condition on
the sampled output produced by each drafter. This keeps bucket membership fixed
across models.

Coverage in the complete 1,753-call replay:

| original completion length | calls | share of TB calls | share of TB decode tokens |
|---|---:|---:|---:|
| `64-128` | 699 | 39.87% | 11.72% |
| `128-256` | 359 | 20.48% | 12.47% |
| `64-256` | 1,058 | **60.35%** | **24.19%** |

The comparison table uses the 962 `64-256` calls with non-empty per-request
speculative metrics for every listed replay: 642 calls in `64-128` and 320 in
`128-256`. Acceptance columns are per-turn means except for the final column.

The bucket tok/s values are normalized decode-throughput estimates:
`step-weighted accept_len / measured serial t_step`. They are not independently
timed subset replays. Serial `t_step` is measured for each architecture and is
stable across workloads to less than 0.2%; the DFlash2 final and DSpark 49K rows
use the measured cost of their matching architecture.

| Drafter | 64-128 accept | 64-128 tok/s | 128-256 accept | 128-256 tok/s | 64-256 overall | Step-weighted |
|---|---:|---:|---:|---:|---:|---:|
| DFlash2 | 6.142 | 219.1 | 5.177 | 207.6 | 5.821 | 4.213 |
| DFlash2 repeat | 6.081 | 221.1 | 5.098 | 199.3 | 5.754 | 4.141 |
| Ours DFlash2 mid, step 1,976 | 7.232 | 231.5 | 5.577 | 202.8 | 6.682 | 4.267 |
| Ours DFlash2 final, step 3,956 | 7.314 | 231.7 | 5.485 | 205.1 | 6.705 | 4.304 |
| DFlash official | 6.377 | 221.4 | 5.144 | 208.5 | 5.967 | 4.168 |
| Ours DSpark 32K | 7.245 | 229.7 | 5.355 | 193.5 | 6.616 | 4.231 |
| Ours DSpark 49K | 7.223 | 226.9 | 5.227 | 195.1 | 6.559 | 4.214 |
| DSpark community | 3.519 | 158.6 | 3.466 | 155.4 | 3.502 | 3.140 |

The fine-tunes show large gains when each request receives equal weight. The
step-weighted gains are much smaller because the selected turns account for
60.35% of requests but only 24.19% of decoded tokens.

## Later checkpoints and longer-context runs

Both later variants have complete 1,753-call Terminal-Bench replays. The full
column uses the canonical server counters; the length slices use paired
per-request metrics because server counters cannot be filtered by call:

| checkpoint | full TB server accept | paired 64-256 per-turn | paired 64-256 step-weighted |
|---|---:|---:|---:|
| DFlash2 mid, step 1,976 | 3.914 | 6.707 | 4.297 |
| DFlash2 final, step 3,956 | 3.862 | 6.693 | 4.281 |
| DSpark 32K | 3.810 | 6.586 | 4.197 |
| DSpark 49K | 3.772 | 6.534 | 4.203 |

The DFlash final replay contains one 58,019-token generation. In per-request
pooling, removing that single outlier changes acceptance from 3.854 to
approximately 3.931.
On the paired `64-256` slice, the midpoint and final checkpoint are effectively
tied. The DSpark 49K run is a longer-context run, not a separately identified
later-step checkpoint; it is also effectively tied with DSpark 32K on the
training-length slice.

## Reasoning share versus turn and context length

Prompt/context length has little relationship with acceptance after output
length is controlled. Output length is predictive, but it is not cleanly
separable from response composition. At fixed original output length, native
DFlash2 acceptance still changes substantially with reasoning share:

| original output bucket | least-reasoning quartile | most-reasoning quartile |
|---|---:|---:|
| `64-256` | 5.70 | 3.69 |
| `256-1K` | 5.54 | 3.61 |

Therefore the supported conclusion is that **context length is weak, while
output length and reasoning composition are both important**. The data does not
support claiming that output length is more important than reasoning share.

The earlier large low-reasoning advantage conditioned on both drafter replays
landing in the same sampled output-length bucket. Because that bucket is an
outcome of each drafter, it is a selected comparison. Using the original
recording's fixed covariates and reweighting Terminal-Bench to the SWE-Gym 32K
token distribution does not establish a DFlash2 improvement:

| comparison | training-token-reweighted delta | 95% bootstrap interval |
|---|---:|---:|
| ours DFlash2 vs native replay 1 | +0.074 | [-0.063, +0.208] |
| ours DFlash2 vs native replay 2 | -0.036 | [-0.186, +0.146] |

## HumanEval

HumanEval cannot reproduce the same median/mean slice: none of its 164 greedy
completions are below 256 tokens. Its median is 1,019 and its mean is 1,141.
The closest available band is `256-1K`, containing 82 problems:

| Drafter | per-turn accept | step-weighted accept | decode tok/s |
|---|---:|---:|---:|
| DFlash2 | 6.336 | 6.291 | 328.4 |
| Ours DFlash2 mid | 5.816 | 5.772 | 301.3 |
| DFlash official | 4.995 | 4.984 | 265.1 |
| Ours DSpark 32K | 4.855 | 4.828 | 248.0 |
| DSpark community | 5.046 | 5.014 | 257.5 |

No HumanEval replay exists for the DFlash2 final or DSpark 49K checkpoints.
Within HumanEval's shortest available outputs, the DFlash2 fine-tune beats the
official DFlash checkpoint but not native DFlash2.

## Retraining implication

An equal number of turns from each length bucket is not the appropriate target:
deployment throughput is step-weighted. A follow-up dataset should instead
move supervised **token mass** toward the evaluation distribution and jointly
track reasoning share.

Resampling the current SWE-Gym corpus is insufficient. It can discard
overrepresented `64-256` supervision, but it cannot create the missing `>=1K`
tail, especially the absent `>=4K` examples. A defensible additional run would
require newly generated long, reasoning-heavy SWE-Gym trajectories. If time
permits one controlled ablation, continue from the DFlash2 midpoint with a
lower learning rate and evaluate several early checkpoints against the full
Terminal-Bench replay. A full run using only downsampled existing data is not
well supported by these results.

## Recommended presentation framing

Lead with the complete pooled evaluation, then present the fixed-bucket table
as a diagnostic. Do not introduce the filtered result as the primary fairness
correction: the model is deployed on the full step-weighted workload, and a
post-hoc subset alone can look like metric selection.

A concise sequence is:

1. The existing drafters were weakest on agentic coding, motivating on-policy
   SWE-Gym fine-tuning.
2. DSpark improved substantially over its community warm start; DFlash2 did not
   beat the stronger native DFlash2 baseline on the full workload.
3. Per-request analysis showed that the fine-tunes improved the majority of
   ordinary-length calls, but the aggregate speed metric is dominated by a
   smaller long, reasoning-heavy tail.
4. The data audit corrected an initially misleading action-only token count and
   isolated the actual coverage gap: 6.1% versus 34.6% of token mass in `>=1K`
   turns.
5. Longer training and a 49K context window did not resolve the gap, motivating
   a targeted long-reasoning data experiment rather than more undirected
   training.

This framing reports the full result before the favorable slice, demonstrates
the evaluation discipline, and shows the diagnostic iteration without claiming
that the original experiment anticipated a mismatch it did not measure.
