# HumanEval and MBPP acceptance for five speculators

Companion to the Terminal-Bench replay eval, not a replacement. **The two use
different decoding regimes and their numbers must not be put in one table.**

> **One row is invalid: the DFlash2 fine-tune.** It was served through the
> unpatched `docker/serve_patched.sh` that was on the node, which drops
> DFlash2's `output_multiplier` (0.196) and `final_logit_softcapping` (20.0)
> for *converted* speculators-format checkpoints — a 5.1x logit-scale error.
> Its baseline `dflash2` is a native z-lab checkpoint and is **not** affected,
> so that comparison is asymmetric and its apparent regression is not
> attributable to the fine-tune. Details in "The DFlash2 fine-tune rows are
> confounded" below. Every other row is unaffected.

## Decoding regime

| | this eval | Terminal-Bench replay (`eval_replay.sh`) |
|---|---|---|
| temperature | **0 (greedy)** | 1.0 |
| top_k | n/a | 64 |
| **concurrency** | **1** | 10 |
| num_speculative_tokens | 15 | 15 |
| max_tokens | 2048 | unset |
| prompts | HumanEval 164, MBPP 500 | frozen TB call set |

Concurrency is pinned and reported because acceptance is **not**
contention-independent: we measured it rising 0.04–0.07 going from concurrency
10 to 64. Every number below is greedy @ concurrency 1 on one H200, vLLM
`specd:latest`, target `meta-models/Muse-Glimmer-30B`.

## Results — HumanEval (164 problems)

`accept_len = 1 + num_accepted_draft_tokens / num_spec_steps`, pooled over all
164 requests. **Δ** is the cross-check against the server's own
`/metrics` counters (`1 + spec_decode_num_accepted_tokens_total /
spec_decode_num_drafts_total`), taken as a before/after delta so warmup traffic
cannot leak in.

| Drafter | method | accept_len | Δ vs `/metrics` | draft rate | tok/s | pass@1 |
|---|---|---|---|---|---|---|
| Official DFlash (`dflash-official`) | dflash | 4.853 | +0.0000 | 0.257 | 255.3 | 0.878 |
| DSpark (`dspark-community`) | dspark | 4.658 | +0.0000 | 0.244 | 236.4 | 0.878 |
| **DFlash 2** (`dflash2`) | dflash | **5.748** | +0.0000 | 0.317 | 296.1 | 0.878 |
| DSpark fine-tuned (`dspark-run-a-32k`) | dspark | 4.541 | +0.0000 | 0.236 | 230.4 | 0.878 |
| DFlash2 fine-tuned (`dflash2-run-d-32k-mix-step1976`) ⚠️ | dflash | _5.299_ | +0.0000 | 0.287 | 273.2 | 0.878 |

⚠️ = served at the wrong logit scale; see the banner above. The number is a
faithful measurement of what was served, not of the checkpoint.

The two acceptance paths agree exactly, not merely to ~0.01: for `dflash2` the
server counters read 154,620 accepted / 32,567 drafts and the summed
per-request blocks read 154,620 / 32,567. Baseline counters were 0.0 in all
five runs.

## Results — MBPP (500 problems, the standard test split)

Same regime, same driver, same drafter order; only the prompt set differs.

| Drafter | method | accept_len | Δ vs `/metrics` | draft rate | tok/s | pass@1 |
|---|---|---|---|---|---|---|
| Official DFlash (`dflash-official`) | dflash | 4.803 | +0.0000 | 0.254 | 253.0 | 0.710 |
| DSpark (`dspark-community`) | dspark | 4.528 | +0.0000 | 0.235 | 230.2 | 0.710 |
| **DFlash 2** (`dflash2`) | dflash | **5.436** | +0.0000 | 0.296 | 280.7 | 0.712 |
| DSpark fine-tuned (`dspark-run-a-32k`) | dspark | 4.440 | +0.0000 | 0.229 | 225.8 | 0.712 |
| DFlash2 fine-tuned (`dflash2-run-d-32k-mix-step1976`) ⚠️ | dflash | _5.054_ | +0.0000 | 0.270 | 261.2 | 0.710 |

**Read the MBPP numbers with more caution than the HumanEval ones.** 133 of 500
generations hit the 2048-token cap and they carry **43.4%** of all tokens
(HumanEval: 20 of 164, 21.9%). The pooled number is therefore dominated to a
much greater degree by long rambling generation rather than by code. Excluding
truncated rows moves MBPP by +0.12 to +0.35:

| Drafter | all 500 | stop-only (367) | shift |
|---|---|---|---|
| dflash-official | 4.803 | 4.921 | +0.118 |
| dspark-community | 4.528 | 4.763 | +0.235 |
| dflash2 | 5.436 | 5.789 | +0.353 |
| dspark-run-a-32k | 4.440 | 4.626 | +0.186 |
| dflash2-run-d-32k-mix-step1976 | 5.054 | 5.350 | +0.296 |

Ranking is identical under both, and both fine-tune regressions keep their sign
and roughly their size. For comparison against a *published* MBPP number the
stop-only column is the better choice, since a published harness would not have
had a quarter of its answers cut off mid-generation.

MBPP pass@1 (0.710) is not comparable to HumanEval's: 3 of MBPP's own reference
solutions fail their own tests (2 `RecursionError`, 1 `IndentationError`), so
the ceiling is 0.994, and 114 of 500 responses return empty content for the
parser reason described below.

## Against the published reference

| Task | | Official DFlash | DSpark | DFlash 2 |
|---|---|---|---|---|
| HumanEval | published | 4.11 | 4.33 | 5.66 |
| HumanEval | ours | 4.85 | 4.66 | 5.75 |
| MBPP | published | 3.74 | 4.02 | 5.30 |
| MBPP | ours | 4.80 | 4.53 | 5.44 |

DFlash 2 is ahead of both DFlash 1 and DSpark on both tasks, in the published
table and in ours — that is the part that replicates (+0.63 to +0.89 over
DFlash 1 here, +1.55 and +1.56 published, so the size of the lead does not
transfer, only its direction). The DFlash-vs-DSpark
ordering is inverted in ours on **both** tasks (published has DSpark ahead by
0.22 and 0.28; we have DFlash ahead by 0.19 and 0.28). A consistent inversion
across two independent benchmarks is a property of this target model, not
noise.

## The finding that stands: the DSpark fine-tune regresses

| task | comparison | baseline | fine-tune | Δ | verdict |
|---|---|---|---|---|---|
| HumanEval | DSpark | 4.658 | 4.541 | **−0.118** | real |
| MBPP | DSpark | 4.528 | 4.440 | **−0.088** | real |
| HumanEval | DFlash 2 | 5.748 | 5.299 | −0.449 | **confounded** |
| MBPP | DFlash 2 | 5.436 | 5.054 | −0.383 | **confounded** |

The two DSpark rows are the ones to trust. Both drafters are speculators-format
(`speculators_model_type: dspark`), neither carries the output-shaping knobs,
and no DSpark code path reads them — the comparison is symmetric. Both deltas
are far outside the 0.005 noise floor, both survive the truncation sensitivity
check, and the effect reproduces on two independent benchmarks.

The two DFlash 2 rows cannot be read as a fine-tune result until the row is
re-run with a serve script carrying the knob patch. Our fine-tunes were trained on Terminal-Bench agent traces, so
losing acceptance on standalone HumanEval-style code generation is consistent
with specialisation to that distribution — but on the evidence that survives,
that claim is established only for DSpark. `dflash2` (unmodified, native) is
the best drafter on both tasks by a wide margin and is a clean measurement.

## The DFlash2 fine-tune rows are confounded

Found after the sweeps, while committing: `docker/serve_patched.sh` in the
working tree carries a patch that the copy on the node did not have, and the
difference falls asymmetrically across exactly one comparison.

`vllm/transformers_utils/configs/speculators/algos.py::update_dflash2` rebuilds
a native `dflash_config` from a speculators-format checkpoint and forwards only
four keys — `conv_kernel_size`, `conv_group_size`, `selector_rank`,
`selector_top_k`. It does **not** forward `output_multiplier` or
`final_logit_softcapping`. `qwen3_dflash2.py:275-279` then reads both straight
back out of that dict to build the candidate `LogitsProcessor`, defaulting to
`scale=1.0` and `soft_cap=None`.

Which side of that each drafter lands on:

| drafter | format | knobs reach the model? | served at |
|---|---|---|---|
| `dflash2` (baseline) | native z-lab, ships its own `dflash_config` | yes, bypasses `update_dflash2` | scale 0.196, cap 20.0 ✅ |
| `dflash2-run-d-32k-mix-step1976` | speculators-format, knobs at top level | **no, silently dropped** | scale 1.0, no cap ❌ |

The fine-tune's weights were fit for scale 0.196 (it was trained with
`patches/speculators-dflash2-output-shaping.patch`), so it was served with a
5.1x logit mismatch while its baseline was served correctly. Nothing in the
stack errors on this. Lower acceptance is the expected symptom, which is
exactly what the two rows show — so the regression cannot be separated from the
bug.

The DSpark pair is not affected: both are speculators-format, symmetric, and
neither `config.json` carries the knobs (the only `final_logit_softcapping`
reader outside DFlash2 is `gemma4_dspark.py`, a different model family).
`dflash-official` is a plain DFlash1 checkpoint and never touches this path.

This is the trap `docs/HANDOFF-dflash2.md` already names: its step 1 is
`grep -c output_multiplier .../speculators/algos.py`, with "`0` means the eval
measured a 5.1x logit mismatch, not the fine-tune". The count is 0 on the image
these sweeps used, so run D's Terminal-Bench regression (4.02 baseline -> 3.80
final) and the two rows here are all the same unresolved confound rather than
three independent results. That is one more reason to treat the DFlash2
fine-tune as unmeasured: nothing has yet compared it against its baseline on an
even footing.

**To resolve it:** re-run that one drafter on both tasks with a serve script
carrying the vLLM-side fix (the patched `docker/serve_patched.sh`, or
`patches/vllm-dflash2-output-shaping.patch`), and confirm
`dflash2 knob patch OK` in the server log alongside the existing
`Using V2 Model Runner` check. About one GPU-hour for both tasks. Until then the
DFlash2 fine-tune is unmeasured, not worse.

## Caveats, in order of how much they could move a number

**1. Truncation at the 2048-token cap, on both tasks.** HumanEval: 20 of 164
generations, 21.9% of all tokens. MBPP: 133 of 500, 43.4% of tokens. Excluding
them raises every row, by more than the gaps between some rows — so the
absolute values are cap-dependent. The ranking is not. HumanEval:

| Drafter | all 164 | stop-only | shift |
|---|---|---|---|
| dflash-official | 4.853 | 4.954 | +0.101 |
| dspark-community | 4.658 | 4.816 | +0.158 |
| dflash2 | 5.748 | 5.999 | +0.251 |
| dspark-run-a-32k | 4.541 | 4.672 | +0.131 |
| dflash2-run-d-32k-mix-step1976 | 5.299 | 5.517 | +0.218 |

Ranking is identical under both, and every fine-tune-vs-baseline gap keeps its
sign (−0.144 and −0.482 stop-only; the MBPP table is in the MBPP section).
`max(completion_tokens)` is 2048 on every row of both tasks — the cap, not a
runaway; no generation ran away the way the 58K-token one did on
Terminal-Bench.

The cap is not really the problem: median completion length is 1024 tokens on
HumanEval and 1142 on MBPP, for tasks whose answers are a few dozen tokens of
code. The model is verbose by default at this serving config, and raising the
cap would likely buy more prose rather than fewer truncations. Re-running with
a larger cap (~5 GPU-hours for both tasks) would settle it, and is the obvious
next step if these absolute numbers need to be quotable rather than
comparative.

**2. pass@1 is a property of the target, not the drafter.** All five rows give
0.878 because greedy speculative decoding is lossless: 161–163 of 164 outputs
are byte-identical across drafters, and every difference sits in a generation
at the 2048 cap. Treat pass@1 here as a harness sanity check, not a
differentiator.

15 of the 20 pass@1 failures are a serving artifact rather than bad code: those
responses have `finish_reason=length` and **empty** content, because the
`muse_glimmer` tool-call parser buffers deltas and the buffer is discarded when
generation is cut at the cap. Among the 144 cleanly-terminated generations,
pass@1 is 142/144 = 0.986. The headline 0.878 keeps the standard 164
denominator; the gap between the two is the cap plus the parser, not the model.

**3. Client-side TTFT/TPOT are invalid on both tasks, as suspected.** Median
`ttft/e2e` is 0.86–0.88 on all ten runs — the first delta arrives around 87% of
the way through the response, which is the tool-call parser buffering, not a
real time-to-first-token. It is a systematic delay rather than a total
collapse: only 2–3% of calls land within 5% of their own e2e (the
Terminal-Bench workload, where nearly every call ends in tool_calls, collapses
much harder at 86.6%).

Server-side histograms are recorded inside the engine, before the parser, and
are unaffected:

| task | TTFT p50 | TTFT p90 | ITL p50 | TPOT p50 |
|---|---|---|---|---|
| HumanEval | 0.043–0.046 s | 0.057–0.058 s | 0.0175 | 0.0050 |
| MBPP | 0.035–0.036 s | 0.054–0.055 s | 0.0175 | 0.0050 |

Two orders of magnitude below the client-side figure, which settles it. But
vLLM's ITL/TPOT buckets are too coarse to separate drafters — all ten runs read
an identical ITL p50 of 0.0175 and TPOT p50 of 0.0050 — so **server-side
latency cannot rank these drafters either**. The tok/s column in the results
tables (output tokens over summed e2e latency at concurrency 1) is the usable
speed comparison: `dflash2` is 1.16× `dflash-official` on HumanEval and 1.11×
on MBPP.

Regenerate with `benchmark/prom_latency.py <run>.prom ...` for the server side
and `benchmark/humaneval/analyze.py` for the client-side validity check.

**4. Scorer validated against canonical solutions** before use: HumanEval
164/164 pass, MBPP 497/500. The 3 MBPP failures are defects in the dataset's
own reference solutions (2 `RecursionError`, 1 `IndentationError`), not in the
harness — they put a 0.994 ceiling on any MBPP pass@1 measured this way.

## Reproducing

```bash
NAME=dflash2 SPEC=/mnt/data/speculators/dflash2 METHOD=dflash \
  TASK=HumanEval CONC=1 TEMP=0 bash scripts/eval_humaneval.sh
```

Note the serve script matters: the runs above used the node's copy of
`docker/serve_patched.sh`, which predates the DFlash2 knob patch now in the
working tree. Any re-run of a **converted** DFlash2 checkpoint must use the
patched version and check for `dflash2 knob patch OK` in the log.

`scripts/eval_humaneval.sh` keeps `eval_replay.sh`'s guards (port-free check,
container-alive check before and after, `dspark patch OK`, log captured to a
variable rather than piped into `grep -q`) and adds two: it refuses to reuse a
NAME that already has results, and the DFlash2 V2-runner check is **fatal**
rather than a warning, since a DFlash2 checkpoint on the V1 runner degrades to
DFlash1 silently and still reports a believable number. `Using V2 Model Runner`
was confirmed for both DFlash2 rows.

Prompts: HumanEval from `RedHatAI/speculator_benchmarks` `HumanEval.jsonl` (the
set the upstream speculators HumanEval example uses); MBPP from
`google-research-datasets/mbpp` `full` **test** split (500 problems, task_id
11–510), normalised by `benchmark/humaneval/prep_mbpp.py` into the same shape.
Both are sent as a single user message
with the raw stub as content, no `chat_template_kwargs` — `reasoning_strength`
is a Muse-Glimmer knob with no counterpart in the published table, and the
model emitted 0 reasoning tokens at its served default.

- driver `benchmark/humaneval/gen.py`, scorer `benchmark/humaneval/metrics.py`
- MBPP prep `benchmark/humaneval/prep_mbpp.py`
- whole-sweep driver `scripts/sweep_coding_bench.sh HumanEval|MBPP`
- cross-drafter analysis `benchmark/humaneval/analyze.py` — regenerates every
  comparison in this document (acceptance sensitivity, output identity,
  truncation, client-latency validity) from the traces
- server-side latency `benchmark/prom_latency.py`
- per-drafter result JSONs committed under `results/coding_benchmarks/`
- traces `/mnt/data/traces/he-<name>/gen.jsonl` and `he-mbpp-<name>/`, results `/mnt/data/eval/he-<name>.json` and `he-mbpp-<name>.json`
