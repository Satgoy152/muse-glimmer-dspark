# Three eval sweeps: bucket x concurrency, SWE-bench Multilingual, replay repeats

Results from the sweeps described in `docs/HANDOFF-eval-sweep.md`, run on the
preemptible 1xH200 starting 2026-09-08 09:26 UTC. Nothing here is appended to
`docs/RESULTS.md` or `docs/RESULTS-paired.md`; where a number is comparable to
one of those it says so, and where it is not it says why.

**Status: IN PROGRESS.** Sections are filled in as cells land. Anything still
marked *pending* has not been measured.

---

## What is and is not comparable to the existing tables

| | existing TB rows (`RESULTS-paired.md`) | task 1 here | task 3 here |
|---|---|---|---|
| sampling | temperature 1.0, top_p 0.95, top_k 64 | **greedy** (temperature 0, no top_p/top_k) | temperature 1.0, top_p 0.95, top_k 64 |
| call set | all 1,753 replayable | 360-call bucket manifests | all 1,753 replayable |
| concurrency | 10 (and 64) | 1, 2, 8, 32 | 10 |
| `NUM_SPEC_TOKENS` | 15 | 15 | 15 |

**Task 1's rows are greedy and the existing Terminal-Bench rows are temperature
1.0. They are not comparable.** Greedy was chosen so that every drafter decodes
the same token sequence for the same prompt, which removes sampling noise from
the drafter comparison entirely — the same reason the HumanEval and MBPP tables
in `RESULTS-paired.md` are exact. It also changes the workload: greedy draws
longer completions than the temperature-1.0 recording did (the `64-128` bucket
generated 17,483 tokens against 13,654 recorded), so the buckets describe the
*recorded* call, not the replayed one.

Task 3's rows keep the original settings, because their whole purpose is to
repeat runs that already exist.

## How the numbers are computed

Server-side Prometheus counters, taken as an **after-minus-before delta** over
each cell's own `/metrics` snapshots:

```
t_step = inter_token_latency_seconds_sum / inter_token_latency_seconds_count   # _count == spec steps
TPOT   = request_decode_time_seconds_sum / (request_generation_tokens_sum - n) # == t_step / accept_len
tok/s  = request_generation_tokens_sum   / request_decode_time_seconds_sum     # decode only, prefill excluded
```

The delta is not optional here. Sixteen cells share one server, so the after
snapshot alone would report cell 16 as the sum of cells 1 through 16 plus the
warmup. `scripts/sweep_cell.sh` snapshots immediately before and immediately
after each replay; `scripts/eval_replay.sh` has been given the same baseline
snapshot for the same reason.

Client-side TTFT/TPOT are still invalid — `--enable-auto-tool-choice` buffers
deltas until a tool call is whole — so TTFT here is the server histogram's
`time_to_first_token_seconds_sum / e2e_request_latency_seconds_count`. The
difference is large and one-directional: on the no-spec `64-128` cell at
concurrency 1 the server reports a 0.212 s mean TTFT and the client's p50 is
1.432 s.

Acceptance is reported under **both** poolings, because they disagree and both
are real:

* **step-weighted**, `1 + sum(accepted)/sum(steps)` pooled over the cell. This
  is the one throughput is made of.
* **per-request**, the mean over calls of that call's `1 + acc/steps`, excluding
  calls with fewer than 5 spec steps. This is the average call. It is the same
  cut `benchmark/analysis/macro.py` uses, so it stays comparable to the macro
  columns in `RESULTS-paired.md`.

### Noise floor, restated

From `RESULTS-paired.md`: `t_step` reproduces to <0.2% across workloads at
concurrency 1 and swings **5.8%** run-to-run at concurrency 10. **No throughput
difference below ~6% is quoted as a result at concurrency 10 or above.** The
paired bootstrap CI on pooled acceptance over the full 1,753-call set is ±0.11.
Per-cell samples here are much smaller than that, so per-cell acceptance
differences are read against the repeat control, not against zero.

## The bucket manifests

Four immutable manifests, built once by
`benchmark/terminal_bench/make_bucket_manifests.py` and reused byte-for-byte for
every drafter and every concurrency. Bucketing is on the **original recording's**
`completion_tokens`, never the replayed output: bucketing on the replay selects
on the outcome, and `docs/RESULTS-paired.md` shows the control then swings as
hard as the effect.

Per-bucket sample sizes were fixed on 2026-09-08 **before any cell was run**, to
fit ~100K output tokens per (drafter, concurrency) pass, weighted toward the
cheap buckets so the `>=1K` bucket did not consume the whole night.

| bucket | pool | sampled | trajectories | recorded out tok | recorded max | `max_tokens` | sha256[:16] |
|---|---:|---:|---:|---:|---:|---:|---|
| `64-128` | 699 | **160** | 35 | 13,654 | 125 | 512 | `93e93121a2559f97` |
| `128-256` | 359 | **120** | 34 | 21,432 | 255 | 1024 | `ee7d9f657eeb008e` |
| `256-1K` | 439 | **60** | 34 | 25,589 | 1,019 | 4096 | `1834f35b5a7085e7` |
| `>=1K` | 85 | **20** | 20 | 46,863 | 7,538 | 8192 | `082f365264c9a9eb` |
| total | 1,582 | **360** | 36 | 107,538 | | | |

### The generation cap, and the run that was thrown away

The recorded requests carry no `max_tokens`, and the first attempt at this sweep
ran without one. It produced this, in the `>=1K` bucket at concurrency 1 with no
speculation:

| | |
|---|---|
| calls in the cell | 20 |
| output tokens in the cell | 157,876 |
| **output tokens in the single largest call** | **129,888** (82%) |
| that call's recorded length | within 1,019-7,538 |
| that call's wall time | 2,125 s of the cell's 2,584 s |

Greedy decoding walked into a repetition loop and ran to the 131,072-token
context limit. Temperature 1.0 does not do this, which is why the existing
Terminal-Bench runs never hit it; the worst they drew was 58,019 tokens, and
`docs/RESULTS-paired.md` already records what one call that long does to a
token-weighted pool.

Three things made this fatal rather than merely slow:

1. A repetition loop is near-perfectly predictable, so it drafts extremely well.
   That one call would have set the bucket's step-weighted acceptance almost on
   its own, and the number would have looked plausible.
2. At concurrency 32 the other 19 calls finish in the first couple of minutes and
   the cell then measures one request decoding alone. That is a concurrency-1
   measurement wearing a concurrency-32 label.
3. `replay.py`'s 900 s timeout does not bound it — the timeout is on the socket,
   not on the length of a stream that keeps producing bytes.

So each bucket now caps generation at **4x its upper edge**, and the open-ended
`>=1K` bucket at **8192** — just above the 7,538-token maximum the recording
actually drew there. The cap is a property of the request, identical for every
drafter and every concurrency, fixed before any drafter ran; it selects on
nothing. The share of calls that reach it is reported per cell.

**The four-and-a-bit uncapped cells measured before this was caught were
discarded, not reused.** They are under `/mnt/data/eval/discarded-uncapped/`
with their original manifests, and no number in this document comes from them.

Calls with fewer than 64 recorded completion tokens (171 of the 1,753) are in no
bucket by construction.

Sampling within a bucket is round-robin across trajectories rather than uniform,
because `replay.py` gives one worker per trajectory: a cell containing 12
trajectories cannot run at concurrency 32 whatever is asked for. **The `>=1K`
bucket contains 20 calls over 20 trajectories, so its concurrency-32 cell runs
at an effective concurrency of at most 20.** Every reported concurrency is the
requested one; this is the one cell where the achieved concurrency is capped
below it.

## Cell naming and the resume trap

Every cell is `<drafter>__<bucket>__c<concurrency>__r<repeat>`, and
`scripts/sweep_cell.sh` refuses to run a cell whose result json already exists.
`replay.py` resumes an existing output path by `replay_of`, so a reused cell name
would have silently skipped every call and handed back the previous cell's
trace.

## Server integrity checks

`scripts/sweep_serve.sh` refuses to hand a server to the sweep unless the run's
own preconditions are in the container log:

* `dspark patch OK` for any DSpark drafter (already fatal in `eval_replay.sh`),
* `Using V2 Model Runner` for any checkpoint whose `config.json` declares
  `DFlash2DraftModel` — **made fatal**. It was a warning, matched on a bare
  `"V2"` that appears in unrelated lines; a DFlash2 checkpoint served on the V1
  runner degrades silently to DFlash1 and produces a clean, plausible, wrong
  trace. `scripts/eval_replay.sh` has been given the same fatal check.
* `no-spec control: target only` for the `SPEC_METHOD=none` denominator.

---

## Task 1 — Terminal-Bench bucket x concurrency sweep

*pending — cells are still running.*

## Task 2 — SWE-bench Multilingual

### Task selection and disjointness

32 instances, selected by `scripts/build_swebench_ml.py` from
`swe-bench/SWE-Bench_Multilingual` (300 instances, 41 repositories).

Disjointness was **enforced and measured at both levels**, not assumed:

| | overlap with `data/training/train_instances.jsonl` |
|---|---|
| instance ids | **0** of 300 |
| repositories | **0** of 41 |

The training corpus is 2,000 SWE-Gym instances over 11 repositories
(`Project-MONAI/MONAI`, `pandas-dev/pandas`, `getmoto/moto`, `python/mypy`,
`iterative/dvc`, `dask/dask`, `modin-project/modin`, `pydantic/pydantic`,
`conan-io/conan`, `facebookresearch/hydra`, `bokeh/bokeh`), all Python. SWE-bench
Multilingual contains none of them, so repository-level disjointness holds for
the whole benchmark and the 32-instance sample inherits it. The selection is
round-robin over (build-tool family, repository), giving **32 instances in 32
distinct repositories**, and every instance's evaluation image was confirmed to
resolve on Docker Hub before selection closed. All 32 images are pulled.

Because the dataset has no `language` column, the language family is taken from
`log_parser` (`parse_log_maven` -> Java, `parse_log_gotest` -> Go, and so on).
Distribution over the 32: 5 Go, 3 Java (Maven) + 2 Gradle + 1 Ant, 3 JavaScript
+ 1 Karma + 1 immutable-js, 3 RSpec + 2 Ruby-unit, 2 PHP, 2 TAP, 2 Redis,
1 each googletest / doctest / jq / jekyll / micropython.

Resolved rate is reported with a **Wilson score interval**: at n=32 the normal
approximation is unusable, and Wilson is the only reason a rate on a sample this
size is quotable.

### Recording

*pending — the target-only recording starts when the task 1 sweep releases the GPU.*

`scripts/swegym_holdout.sh` was a structural reference only. Its converted output
contained one row: the `hi` health probe, which it sent **through the recording
proxy**, so the probe was captured and no agent call was. Reconverting the trace
that script actually left on disk confirms the diagnosis exactly —
`scripts/convert_recorded_calls.py` keeps 1,013 replayable calls over 16
trajectories and drops precisely one row for having no `reasoning_strength`.

`scripts/swebench_ml_record.sh` fixes all three: the health probe goes to the
upstream port and never touches the proxy, the converter drops any row that is
not an agent call (no tools, no reasoning strength, or an errored response), and
`VALIDATE_FIRST=1` records one instance and refuses to launch the other 31
unless that instance produced replayable calls.

## Task 3 — Terminal-Bench replay repeats

*partial — the two extra replays per checkpoint are queued behind task 1. The
table below is the existing single runs, re-pooled, and is what the extra runs
are being added to.*

### Where the question stands before the repeats

Full-set replays, temperature 1.0 / top_k 64 / concurrency 10 /
`NUM_SPEC_TOKENS=15`, re-pooled by `benchmark/analysis/task3_repeats.py`:

| checkpoint | run | calls | out tok | max completion | accept_len (step-w) | accept_len (per-req) | step-w, longest call dropped |
|---|---|---:|---:|---:|---:|---:|---:|
| `dflash2` | `tb-dflash2` | 1,753 | 509,389 | 9,900 | 4.0020 | 5.6748 | 4.0325 |
| `dflash2` | `tb-dflash2-repeat2` | 1,753 | 493,449 | 7,885 | 4.0045 | 5.6497 | 4.0270 |
| run-D midpoint (step 1976) | `tb-dflash2-run-d-32k-mix-step1976` | 1,753 | 513,780 | 8,204 | 3.9085 | 6.2888 | 3.9349 |
| run-D final (step 3956) | `tb-dflash2-run-d-32k-mix` | 1,753 | 574,004 | **58,019** | 3.8537 | 6.2957 | 3.9312 |

**The single 58,019-token completion is most of the gap.** Removing each run's
own longest call — a diagnostic, not a correction — moves the final checkpoint
from 3.8537 to 3.9312 and the midpoint from 3.9085 to 3.9349, and the paired
bootstrap over calls goes from

* as measured: mid − final = **+0.0548**, 95% CI [−0.0967, +0.2077]
* longest call dropped: mid − final = **−0.0306**, 95% CI [−0.1354, +0.0673]

Neither interval excludes zero, and the sign flips. On the **per-request**
pooling the two checkpoints are 6.2888 and 6.2957 — the final checkpoint is
nominally *ahead*. So even before the repeats, the midpoint-beats-final ordering
is a single draw on a token-weighted metric and not a separable difference. The
repeats are being run to say whether that holds across three runs each; three
runs is a run-to-run stability check, not statistical power.

