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

### Greedy does not make the drafters emit the same tokens

The handoff's reason for preferring greedy was that every drafter would then
decode the same token sequence, making the comparison exact. It asked for the
completion hashes to be verified before pooling. They were, and **the assumption
is false at this prompt and output length.**

`replay.py` reconstructs `content` from the streamed deltas but never reassembles
`tool_call` deltas, and this model routes ~99.5% of a turn into `tool_calls`, so
a content hash is empty on almost every call and matches vacuously. The check
that bites is per-call `completion_tokens`, which under identical greedy decoding
must be identical.

Share of calls with an identical completion length, concurrency 1:

| bucket | `dflash2` vs no-spec | **`dflash2` vs itself (r1 vs r2)** | median relative difference where they differ |
|---|---:|---:|---:|
| `64-128` | 83.8% | **96.2%** | 9 tokens |
| `128-256` | 53.3% | **83.3%** | 47 tokens |
| `256-1K` | 43.3% | **80.0%** | 86 tokens |
| `>=1K` | 20.0% | **70.0%** | 262 tokens |

The second column is the one that matters. **The same drafter, same config, same
manifest, replayed twice, diverges too** — so most of what the first column shows
is not "drafter A decodes differently from drafter B", it is vLLM not being
bitwise reproducible run to run at this scale. Divergence probability compounds
per token, which is why agreement falls monotonically with output length.

This does not invalidate anything, but it changes what the greedy rows are:

* Pairing on `replay_of` is still exact — both sides saw a byte-identical prompt.
* The comparison is **paired-with-noise, not exact.** A per-call acceptance
  difference is partly a drafter difference and partly a different continuation.
* So every task-1 comparison is read against the `dflash2` r1-vs-r2 repeat
  control in the same cell, never against zero. That control is why it was run.

Greedy is still the better choice than temperature 1.0 — 96% agreement on short
turns beats 0% by construction — it just does not deliver the exactness the
HumanEval and MBPP tables in `docs/RESULTS-paired.md` have. Those are single-turn
code completions with short prompts and short outputs, where the per-token
divergence probability has far less room to compound.

### TTFT and the prefix cache

Every drafter's server is started cold, and the four bucket manifests are then
replayed against it at concurrency 1, 2, 8 and 32 in that order. So:

* **TTFT is comparable across drafters** at a fixed (bucket, concurrency) — every
  server reaches that cell having served exactly the same traffic before it.
* **TTFT is not comparable across concurrencies within a drafter.** The
  concurrency-2 pass replays prompts the concurrency-1 pass has already served,
  so it runs against a warmer prefix cache. The effect is large: an early
  no-spec cell run against a server that had already served the same manifest
  reported a 0.045 s mean TTFT where the same cell on a cold server reported
  0.212 s.

`request_decode_time_seconds` starts after the first token, so `accept_len`,
`t_step`, TPOT and decode tok/s are untouched by any of this. Only TTFT and the
wall-clock throughput column see the cache.

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

**Complete.** 142 cells: 8 drafters x 4 buckets x 4 concurrencies, plus the
`dflash2` repeat control at concurrencies 1 and 32, plus task 3's six full-set
replays. **Zero errored calls across all 142 cells.** Greedy, `NUM_SPEC_TOKENS`
15, one server per drafter, every server-side number an after-minus-before
delta. 4 h 44 min of GPU.

### The headline: speculation's advantage collapses with concurrency

**Decode throughput over the no-spec control (x), greedy**

| drafter | bucket | c1 | c2 | c8 | c32 |
|---|---|---|---|---|---|
| `dflash-official` | 64-128 | 4.40x | 3.94x | 3.61x | 1.80x |
| `dflash-official` | 128-256 | 3.85x | 3.55x | 3.19x | 1.76x |
| `dflash-official` | 256-1K | 4.04x | 3.82x | 3.33x | 1.91x |
| `dflash-official` | >=1K | 3.46x | 3.44x | 2.89x | 2.38x |
| `dflash2` | 64-128 | 4.41x | 4.26x | 3.53x | 1.80x |
| `dflash2` | 128-256 | 3.81x | 3.76x | 3.31x | 1.79x |
| `dflash2` | 256-1K | 4.10x | 4.00x | 3.49x | 1.97x |
| `dflash2` | >=1K | 3.34x | 3.96x | 3.15x | 2.56x |
| `dflash2-run-d-mid` | 64-128 | 4.71x | 4.53x | 3.82x | 1.95x |
| `dflash2-run-d-mid` | 128-256 | 3.89x | 3.78x | 3.22x | 1.79x |
| `dflash2-run-d-mid` | 256-1K | 3.76x | 3.75x | 3.29x | 2.19x |
| `dflash2-run-d-mid` | >=1K | 3.45x | 3.12x | 2.89x | 2.47x |
| `dflash2-run-d-final` | 64-128 | 4.90x | 4.67x | 3.88x | 1.96x |
| `dflash2-run-d-final` | 128-256 | 3.94x | 3.72x | 3.40x | 1.85x |
| `dflash2-run-d-final` | 256-1K | 4.05x | 3.76x | 3.37x | 1.90x |
| `dflash2-run-d-final` | >=1K | 3.25x | 3.15x | 2.97x | 2.44x |
| `dspark-community` | 64-128 | 3.02x | 2.81x | 2.54x | 1.22x |
| `dspark-community` | 128-256 | 2.91x | 2.83x | 2.44x | 1.38x |
| `dspark-community` | 256-1K | 3.28x | 3.12x | 2.74x | 1.60x |
| `dspark-community` | >=1K | 2.89x | 2.91x | 2.62x | 2.09x |
| `dspark-run-a-32k` | 64-128 | 4.45x | 3.96x | 3.61x | 1.86x |
| `dspark-run-a-32k` | 128-256 | 3.55x | 3.43x | 3.01x | 1.72x |
| `dspark-run-a-32k` | 256-1K | 3.73x | 3.58x | 3.09x | 1.77x |
| `dspark-run-a-32k` | >=1K | 3.01x | 2.81x | 2.59x | 2.16x |
| `dspark-run-b-49k` | 64-128 | 4.34x | 3.89x | 3.61x | 1.86x |
| `dspark-run-b-49k` | 128-256 | 3.56x | 3.36x | 3.01x | 1.75x |
| `dspark-run-b-49k` | 256-1K | 3.88x | 3.47x | 3.05x | 1.79x |
| `dspark-run-b-49k` | >=1K | 2.94x | 2.89x | 2.67x | 2.10x |

Every drafter loses roughly half its advantage between concurrency 1 and 32.
This is new — `docs/RESULTS-paired.md` only had concurrency 1, where the
3.15x-3.47x figures it quotes are reproduced here.

**Acceptance is not what changes.** It is nearly flat in concurrency: `dflash2`
in the `64-128` bucket scores 5.346 / 5.343 / 5.326 / 5.379 at concurrency
1 / 2 / 8 / 32. Acceptance is a property of the drafter and the prompt, and the
batch does not touch it.

What changes is the cost of a step:

**`t_step` (ms), pooled over the four buckets, with the per-bucket spread**

| drafter | c1 | c2 | c8 | c32 |
|---|---|---|---|---|
| `nospec` | 16.13 (±0.1%) | 16.26 (±0.3%) | 16.75 (±1.2%) | 17.86 (±7.7%) |
| `dflash-official` | 19.00 (±0.7%) | 20.01 (±9.1%) | 23.05 (±12.5%) | 40.33 (±61.9%) |
| `dflash2` | 19.35 (±0.4%) | 19.93 (±1.8%) | 23.27 (±14.0%) | 40.33 (±67.6%) |
| `dflash2-run-d-mid` | 19.33 (±0.5%) | 19.97 (±1.0%) | 23.39 (±13.3%) | 38.82 (±66.8%) |
| `dflash2-run-d-final` | 19.36 (±0.4%) | 19.89 (±1.4%) | 23.44 (±12.8%) | 41.44 (±56.4%) |
| `dspark-community` | 19.66 (±0.5%) | 20.52 (±6.2%) | 24.13 (±6.4%) | 41.77 (±62.8%) |
| `dspark-run-a-32k` | 19.67 (±0.7%) | 20.66 (±6.9%) | 23.59 (±13.1%) | 40.31 (±62.8%) |
| `dspark-run-b-49k` | 19.67 (±0.7%) | 20.67 (±9.1%) | 23.66 (±12.3%) | 39.04 (±60.6%) |

The no-spec target step grows 16.13 -> 17.86 ms from concurrency 1 to 32, about
11%. Every speculative step grows 19.0-19.7 -> ~40 ms, more than twice. A
speculative step verifies `NUM_SPEC_TOKENS + 1` positions per sequence instead
of one, so at batch 32 it is asking the target for ~16x more positions per
iteration, and that lands on a GPU that is no longer memory-bound. The drafter
that was nearly free at batch 1 is not free at batch 32.

Subtracting the no-spec step at concurrency 1 gives each family's drafter cost:
**+2.87 ms for DFlash, +3.22 ms for DFlash2, +3.54 ms for DSpark**, against
+3.02 / +3.35 / +3.65 in `docs/RESULTS-paired.md`. That reproduces on a
different workload, under greedy rather than temperature 1.0, to within 0.15 ms.

**The `t_step` spread column is a warning, not a feature.** At concurrency 1 the
four buckets agree to 0.1-0.7%, which is the workload-independence
`RESULTS-paired.md` relies on. At concurrency 32 they disagree by 56-68%,
because the `>=1K` manifest holds 20 calls over **20 trajectories** and
`replay.py` runs one worker per trajectory — that cell cannot reach concurrency
32 and in practice runs far below it. `dflash2`'s `t_step` by bucket at
concurrency 32 is 54.85 / 47.79 / 46.04 / **27.57** ms. So the `>=1K` row of the
speedup table at concurrencies 8 and 32 is measuring a smaller batch than its
label claims, and its higher apparent speedup there is that shortfall, **not**
long generations holding up better under load. Do not read it as the latter.

### Acceptance by output-length bucket

Bucketed on the **original recording's** `completion_tokens`, so nothing selects
on the replayed outcome.

**Acceptance length, step-weighted — greedy, concurrency 1**

| drafter | 64-128 | 128-256 | 256-1K | >=1K | reweighted to the full mix |
|---|---|---|---|---|---|
| `dflash-official` | 5.306 | 4.571 | 4.766 | 4.071 | 4.561 |
| `dflash2` | 5.346 | 4.590 | 4.932 | 4.003 | 4.611 |
| `dflash2-run-d-mid` | 5.782 | 4.703 | 4.530 | 4.132 | 4.561 |
| `dflash2-run-d-final` | 6.019 | 4.768 | 4.889 | 3.896 | 4.659 |
| `dspark-community` | 3.719 | 3.563 | 4.006 | 3.516 | 3.743 |
| `dspark-run-a-32k` | 5.650 | 4.402 | 4.589 | 3.667 | 4.367 |
| `dspark-run-b-49k` | 5.486 | 4.416 | 4.765 | 3.587 | 4.392 |

**Acceptance length, per-request — greedy, concurrency 1**

| drafter | 64-128 | 128-256 | 256-1K | >=1K | reweighted to the full mix |
|---|---|---|---|---|---|
| `dflash-official` | 7.012 | 5.668 | 6.029 | 4.575 | 5.589 |
| `dflash2` | 6.770 | 5.688 | 6.172 | 4.834 | 5.711 |
| `dflash2-run-d-mid` | 8.074 | 6.150 | 6.276 | 4.767 | 5.943 |
| `dflash2-run-d-final` | 8.164 | 6.227 | 6.287 | 4.763 | 5.967 |
| `dspark-community` | 4.164 | 3.832 | 4.770 | 3.950 | 4.290 |
| `dspark-run-a-32k` | 7.828 | 5.705 | 6.014 | 4.269 | 5.577 |
| `dspark-run-b-49k` | 7.783 | 5.720 | 5.950 | 4.306 | 5.561 |

The `reweighted` column weights each bucket by that bucket's share of output
tokens in the full 1,753-call recording, because the per-bucket sample sizes
(160/120/60/20) were chosen to fit a time budget and pooling them raw would
report the budget rather than the workload. It excludes the 171 calls under 64
recorded tokens, which are in no bucket, and it uses recorded token shares while
the replay generated its own — so treat it as an approximate full-mix number,
not as a measurement of the full set.

### Paired against `dflash2`, with the repeat control

Concurrency 1, paired on `replay_of`, 2,000-resample bootstrap. The last row of
each block is the same drafter replayed twice — the only honest yardstick at
these sample sizes.

```
== output bucket 128-256, concurrency 1 -- A vs dflash2 (greedy) ==
A                            n     sw A     sw B  sw delta                95% CI  per-req delta    win   tied
dflash-official            120    4.571    4.590    -0.019      [-0.256, +0.222]  -0.021+-0.166    46%     5%
dflash2-run-d-final        120    4.768    4.590    +0.178      [-0.007, +0.413]  +0.539+-0.199    67%     3%
dflash2-run-d-mid          120    4.703    4.590    +0.113      [-0.056, +0.301]  +0.462+-0.188    66%     4%
dspark-community           120    3.563    4.590    -1.027      [-1.259, -0.841]  -1.856+-0.252     1%     0%
dspark-run-a-32k           120    4.402    4.590    -0.188      [-0.411, +0.040]  +0.016+-0.186    41%     1%
dspark-run-b-49k           120    4.416    4.590    -0.174      [-0.377, +0.037]  +0.031+-0.196    45%     1%
*control* dflash2 r2       120    4.610    4.590    +0.020      [-0.077, +0.128]  -0.012+-0.050    47%    73%

== output bucket 256-1K, concurrency 1 -- A vs dflash2 (greedy) ==
A                            n     sw A     sw B  sw delta                95% CI  per-req delta    win   tied
dflash-official             60    4.766    4.932    -0.166      [-0.453, +0.117]  -0.143+-0.210    31%     3%
dflash2-run-d-final         60    4.889    4.932    -0.042      [-0.354, +0.315]  +0.115+-0.173    53%     3%
dflash2-run-d-mid           60    4.530    4.932    -0.402      [-0.752, -0.094]  +0.104+-0.162    44%     2%
dspark-community            60    4.006    4.932    -0.926      [-1.314, -0.564]  -1.402+-0.291     5%     0%
dspark-run-a-32k            60    4.589    4.932    -0.343      [-0.741, +0.092]  -0.158+-0.266    32%     0%
dspark-run-b-49k            60    4.765    4.932    -0.166      [-0.553, +0.264]  -0.222+-0.249    31%     2%
*control* dflash2 r2        60    4.908    4.932    -0.024      [-0.200, +0.126]  +0.046+-0.062    53%    72%

== output bucket 64-128, concurrency 1 -- A vs dflash2 (greedy) ==
A                            n     sw A     sw B  sw delta                95% CI  per-req delta    win   tied
dflash-official            160    5.306    5.346    -0.040      [-0.277, +0.190]  +0.243+-0.176    57%     6%
dflash2-run-d-final        160    6.019    5.346    +0.673      [+0.361, +1.119]  +1.416+-0.227    85%     4%
dflash2-run-d-mid          160    5.782    5.346    +0.436      [+0.193, +0.746]  +1.326+-0.226    84%     4%
dspark-community           160    3.719    5.346    -1.627      [-1.978, -1.363]  -2.605+-0.265     1%     0%
dspark-run-a-32k           160    5.650    5.346    +0.304      [+0.107, +0.549]  +1.080+-0.240    74%     2%
dspark-run-b-49k           160    5.486    5.346    +0.140      [-0.097, +0.431]  +1.035+-0.244    73%     1%
*control* dflash2 r2       160    5.349    5.346    +0.004      [-0.015, +0.027]  +0.022+-0.043    53%    88%

== output bucket >=1K, concurrency 1 -- A vs dflash2 (greedy) ==
A                            n     sw A     sw B  sw delta                95% CI  per-req delta    win   tied
dflash-official             20    4.071    4.003    +0.068      [-0.244, +0.211]  -0.259+-0.252    25%     0%
dflash2-run-d-final         20    3.896    4.003    -0.106      [-0.343, +0.148]  -0.071+-0.224    45%     0%
dflash2-run-d-mid           20    4.132    4.003    +0.129      [-0.082, +0.237]  -0.068+-0.172    47%     5%
dspark-community            20    3.516    4.003    -0.487      [-0.764, -0.344]  -0.884+-0.279     0%     0%
dspark-run-a-32k            20    3.667    4.003    -0.335      [-0.562, -0.166]  -0.565+-0.167     0%     0%
dspark-run-b-49k            20    3.587    4.003    -0.415      [-0.584, -0.321]  -0.529+-0.154     5%     0%
*control* dflash2 r2        20    4.081    4.003    +0.078      [-0.007, +0.228]  +0.042+-0.124    57%    65%
```

**The fine-tunes win short turns and lose long ones, and the effect is large
where it exists.** In `64-128`, run-D final is +0.673 [+0.361, +1.119] over
`dflash2` step-weighted and +1.416 per request, winning 85% of the calls that
differ, against a control of +0.004 [-0.015, +0.027]. `dspark-run-a-32k` is
+0.304 [+0.107, +0.549] there. In `>=1K` the DSpark fine-tunes go the other way
and clear the control doing it: -0.335 [-0.562, -0.166] for run-a and -0.415 for
run-b, against a control of +0.078.

This is the same shape `docs/RESULTS-paired.md` found by splitting calls after
the fact. It now holds with output length as the **design** variable, bucketed
on the recording rather than the replay, so the earlier concern that the split
was selecting on the outcome does not apply.

**`dspark-community` is far behind every other checkpoint** — -1.627 against
`dflash2` in `64-128`, winning 1% of calls, and behind in every bucket at every
concurrency. Its warm-started fine-tune `dspark-run-a-32k` scores 5.650 against
its 3.719 in that bucket. That gap is the largest and least ambiguous result in
the sweep, and it agrees with the +2.674 per-request gap already reported at
temperature 1.0.

**What the sweep does not resolve.** The `>=1K` bucket holds 20 calls; its
repeat control is +0.078 while several of its effects are +-0.1 to -0.4, so only
the DSpark deficits there clear it. `256-1K` holds 60 calls and its control is
-0.024, so run-D midpoint's -0.402 [-0.752, -0.094] clears, but run-D final's
-0.042 does not. Nothing in the `128-256` bucket clears its control except
`dspark-community`.

### Ranking on the reweighted full mix, concurrency 1

| drafter | step-weighted | per-request |
|---|---:|---:|
| `dflash2-run-d-final` | **4.659** | **5.967** |
| `dflash2` | 4.611 | 5.711 |
| `dflash2-run-d-mid` | 4.561 | 5.943 |
| `dflash-official` | 4.561 | 5.589 |
| `dspark-run-b-49k` | 4.392 | 5.561 |
| `dspark-run-a-32k` | 4.367 | 5.577 |
| `dspark-community` | 3.743 | 4.290 |

Note that on this mix, under greedy, run-D **final** edges `dflash2` on the
token-weighted metric — the opposite of the temperature-1.0 full-set ordering in
`docs/RESULTS-paired.md`, and consistent with task 3's finding that that ordering
was never separable from zero. The margin is 0.048, well inside what the repeat
control and the between-run spread in task 3 (sd 0.064-0.072) can support, so
**this is not a claim that the final checkpoint is better** — it is a statement
that the two orderings disagree and neither is resolved.

## Task 2 — SWE-bench Multilingual

**Selection, recording, all 8 replays and resolved rate complete.** Two controls
— a second target-only rollout and a same-drafter replay repeat — were still
running when this was written, and are marked where they matter.

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

**Complete.** Target-only (`SPEC_METHOD=none`, `no-spec control: target only`
confirmed in the server log), 32 instances, 8 workers, 37 minutes.

| | |
|---|---|
| replayable calls | **2,742** |
| rows dropped by the converter | **0** |
| trajectories | 32 — every task recorded |
| output tokens | 516,208 (mean 188.3, max 3,248) |
| calls per task | min 43, median 97.5, max 100 |
| reasoning strengths | 644 low / 654 medium / 656 high / 788 xhigh |

For scale, the frozen Terminal-Bench set is 1,753 calls and ~520K output tokens,
so this is a comparable replay workload on a benchmark with no repository overlap
with training.

Agent exit statuses: **16 Submitted, 15 LimitsExceeded, 1 RepeatedFormatError**,
and 16 of 32 predictions carry a non-empty patch. Half the tasks run to the
100-step limit without submitting. That is worth stating before any resolved
rate is quoted: **the ceiling on this sample is 16/32 even before grading**, so a
low resolved rate here is as much a statement about a 30B model on 9-language
repositories at a 100-step budget as it is about any drafter.

The validation gate did its job: instance 0 was recorded alone first and produced
100 replayable calls before the other 31 were launched.

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

### Frozen-replay acceptance — the transfer result

All drafters replay the same 2,742 recorded calls, paired on `replay_of`,
temperature 1.0 / top_k 64 / concurrency 10 / `NUM_SPEC_TOKENS` 15 — the same
settings as the Terminal-Bench full-set rows, so the two benchmarks are directly
comparable to each other (and, unlike task 1, to `docs/RESULTS-paired.md`).

All eight drafters complete.

| drafter | accept_len (step-w) | accept_len (per-req) | t_step ms | decode tok/s | vs no-spec |
|---|---:|---:|---:|---:|---:|
| *no speculation* | 1.000 | — | 19.42 | 51.8 | 1.00x |
| `dflash-official` | 4.708 | 6.483 | 36.88 | 127.0 | 2.45x |
| `dflash2` | 4.796 | 6.362 | 37.87 | 126.7 | 2.45x |
| **ours `dflash2-run-d-mid`** | **5.081** | **7.500** | 37.10 | **136.2** | **2.63x** |
| **ours `dflash2-run-d-final`** | **5.058** | **7.549** | 37.80 | 133.0 | **2.57x** |
| `dspark-community` | 3.306 | 3.387 | 35.62 | 93.2 | 1.80x |
| ours `dspark-run-a-32k` | 4.780 | 7.286 | 37.86 | 124.7 | 2.41x |
| ours `dspark-run-b-49k` | 4.795 | 7.348 | 37.70 | 125.6 | 2.43x |

Paired bootstrap against `dflash2`, 2,000 resamples, n=2,742:

| A vs `dflash2` | step-w Δ | 95% CI | per-req Δ | win |
|---|---:|---|---:|---:|
| ours `dflash2-run-d-mid` | **+0.300** | **[+0.189, +0.418]** | +1.243 ±0.094 | 71% |
| ours `dflash2-run-d-final` | **+0.272** | **[+0.160, +0.387]** | +1.311 ±0.089 | 73% |
| ours `dspark-run-b-49k` | +0.003 | [−0.113, +0.121] | +1.116 ±0.092 | 67% |
| ours `dspark-run-a-32k` | +0.008 | [−0.098, +0.112] | +1.041 ±0.093 | 66% |
| `dflash-official` | −0.075 | [−0.177, +0.022] | +0.058 ±0.068 | 51% |
| `dspark-community` | **−1.481** | [−1.580, −1.393] | −2.971 ±0.087 | 4% |

**The two poolings say different things about the DSpark fine-tunes, and both
are worth having.** `dspark-run-a-32k` and `dspark-run-b-49k` are indis­tinguishable
from `dflash2` step-weighted (+0.008 and +0.003) yet **+1.04 and +1.12 per
request**, winning two calls in three. They win the typical call and lose the
tokens, which is the same split `docs/RESULTS-paired.md` found on Terminal-Bench.
Against the checkpoint they were warm-started from they are ahead on everything:
`dspark-community` sits at 3.306 step-weighted and 3.387 per request, −1.481 and
−2.971 against `dflash2`, and it is the only drafter here under 2.0x on
throughput.

**Both DFlash2 fine-tunes beat native `dflash2` on this benchmark, on both
poolings, with intervals clear of zero.** On Terminal-Bench at identical
settings the same checkpoints are −0.09 step-weighted and not separable from
zero. The fine-tune transfers *better* to the benchmark it was never trained on
than to the one already being used to evaluate it.

The throughput column agrees but resolves less: 136.2 against 126.7 tok/s is
+7.5%, only just past the ~6% that `t_step` run-to-run movement at concurrency 10
allows anyone to quote. The mechanism is not in dispute though — `t_step` differs
by 2% (37.10 vs 37.87) while acceptance differs by 6%, so the gain is acceptance,
not a cheaper drafter.

**Caveat: this comparison has no same-drafter repeat on this call set yet.** The
Terminal-Bench repeat control at concurrency 10 is +0.010 [−0.101, +0.120], which
would put +0.300 well clear — but borrowing a noise floor from a different
workload is exactly the shortcut this document has twice found to be wrong. A
`dflash2` repeat over the same 2,742 calls is queued.

### Why it transfers better — the reasoning-share prediction holds

`docs/RESULTS-paired.md` concluded that the fine-tune improved the
command-heavy, low-reasoning half of a turn, and that Terminal-Bench decodes
mostly the other half. That predicts the gain should appear on a benchmark whose
reasoning share sits nearer the training corpus. Measured on the generated
characters, by the same method:

| corpus | reasoning share | turns with no reasoning at all |
|---|---:|---:|
| training (SWE-Gym) | 56.8% | 11.6% |
| **SWE-bench Multilingual** | **62.9%** | **11.3%** |
| Terminal-Bench | 75.3% | 2.2% |

SWE-bench Multilingual sits near the training corpus and far from Terminal-Bench,
and its no-reasoning share (11.3%) is almost exactly training's (11.6%). The
fine-tune's step-weighted gain over `dflash2` is +0.300 here and −0.09 there.

**These 32 instances were selected for repository disjointness, not for reasoning
share** — the share was measured afterwards — and the prediction was written down
in `RESULTS-paired.md` before this benchmark was recorded. That is stronger than
a post-hoc fit. It is still two benchmarks and a training corpus: a consistent
direction, not a dose-response curve. A third point (the SWE-Gym holdout, which
should sit at the training end) is queued.

### Resolved rate

On-policy agent runs over the same 32 instances, graded with the SWE-bench
harness (`swebench` 5.0.2). The denominator is all 32 — an instance whose agent
produced nothing is unresolved, not absent.

| arm | resolved | n | rate | 95% Wilson | empty patches |
|---|---:|---:|---:|---|---:|
| target model only | 13 | 32 | 40.6% | [25.5%, 57.7%] | 16 |
| ours `dspark-run-a-32k` | 13 | 32 | 40.6% | [25.5%, 57.7%] | 13 |
| ours `dflash2-run-d-mid` | 12 | 32 | 37.5% | [22.9%, 54.8%] | 13 |
| `dflash2` | 6 | 32 | 18.8% | [8.9%, 35.3%] | 19 |

**No difference here is quotable, and the table should not be read as one.**
Every interval overlaps every other. Three of the four arms sit at 12–13 and only
`dflash2` is low, which is the wrong shape for a drafter effect — if speculation
were costing task resolution, all three drafters would be low, not one.

The first two arms measured were target-only (13) and `dflash2` (6), paired
exact McNemar p=0.065, and that pair on its own looks like a finding. It is the
same trap task 3 turned on: **both arms are single stochastic rollouts at
temperature 1.0, so two runs of the same model differ by construction and
nothing here says by how much.** A second target-only rollout is queued purely to
measure that floor. Until it lands, no delta in this table is interpretable.

Two further limits, independent of any of that. Between 13 and 19 of the 32
predictions in every arm are **empty patches**, so most of the denominator is the
agent never producing a patch at all; and the agent hits its 100-step limit on
roughly half the tasks. The ceiling on this sample was 16/32 before grading
began. This is a measurement of a 30B model on nine-language repositories at a
100-step budget at least as much as it is a measurement of any drafter.

## Task 3 — Terminal-Bench replay repeats

**Complete.**

Two more full-set replays were run for each of the three checkpoints, at the
original settings (temperature 1.0, top_k 64, concurrency 10,
`NUM_SPEC_TOKENS=15`, **no** generation cap — task 1's cap deliberately does not
apply here, because a long draw recurring is the thing being measured). With the
runs already on disk that is four runs of `dflash2` and three each of the two
run-D checkpoints.

### Answer: the ordering does not survive

| pooling | midpoint | final | delta | 95% CI | verdict |
|---|---:|---:|---:|---|---|
| step-weighted, as measured | 3.9871 | 3.9206 | **+0.0664** | [−0.0299, +0.1618] | not separable from zero |
| step-weighted, longest call per run dropped | 4.0159 | 3.9790 | **+0.0369** | [−0.0278, +0.1003] | not separable from zero |
| **per-request** | 6.3491 | **6.3570** | **−0.0079** | — | **final is nominally ahead** |

Paired bootstrap over calls, 2,000 resamples, three runs per checkpoint,
balanced. On the token-weighted metric the sign is preserved but the interval
covers zero; on the per-request metric the sign reverses. **The
midpoint-beats-final ordering is not a result at three runs each.**

Three runs is a run-to-run stability check, not statistical power, and the
repeats show why that distinction matters:

| checkpoint | runs | mean | **between-run sd** | min | max |
|---|---:|---:|---:|---:|---:|
| `dflash2` | 4 | 4.0075 | **0.0151** | 3.9944 | 4.0292 |
| run-D midpoint | 3 | 3.9890 | **0.0716** | 3.9085 | 4.0459 |
| run-D final | 3 | 3.9234 | **0.0639** | 3.8537 | 3.9793 |

The paired bootstrap resamples calls *within* a run, so it says how tightly one
run pins its own number down — not where the next run will land. For the two
run-D checkpoints the between-run spread is roughly 4x `dflash2`'s and of the
same order as the ±0.11 bootstrap interval quoted in `docs/RESULTS-paired.md`.

### The 58,019-token completion was not bad luck

The handoff describes the final checkpoint's regression as resting on "one run
that drew a single 58,019-token completion". The repeats say otherwise.

Longest completion in each run, and how many calls exceeded 16K tokens:

| checkpoint | run | longest call | calls >16K |
|---|---|---:|---:|
| `dflash2` | four runs | 10,127 / 9,900 / 7,885 / 7,223 | 0 / 0 / 0 / 0 |
| run-D midpoint | three runs | 10,318 / 9,394 / 8,204 | 0 / 0 / 0 |
| **run-D final** | three runs | **58,019 / 21,577 / 18,163** | **1 / 1 / 1** |

Every one of the final checkpoint's three runs produced a call over 16K tokens.
**None of the other seven runs produced one at all.** And it is the same
trajectory every time — `c8be87c4ba67de0b` — which supplied the longest call in
all three final runs and, in one of them, the two longest (18,163 and 16,238).
That trajectory appears in every run of every checkpoint; only under the final
checkpoint does it run away.

So the correct statement is not that one run was unlucky. **The final checkpoint
reproducibly drives one trajectory into a very long generation, and that is what
depresses its token-weighted acceptance.** Dropping the longest call per run,
which the table above still reports, is therefore not outlier removal — it
deletes a real and repeatable behaviour of the checkpoint, and understates what a
deployment would see.

This also says something the acceptance numbers do not: speculative decoding is
supposed to be output-preserving, so the drafter should not change *what* the
target generates at all. It does. The same effect shows up in the greedy
divergence measured above, and here it has a direction.

**Read this as an observation, not a finding.** The pattern was noticed in the
data and then quantified, which is exactly the setting where a p-value
overstates. Treating the ten runs as exchangeable, the probability that one
group of three holds all three of the top-three maxima is 1/C(10,3) = 0.008 —
suggestive, and the same-trajectory detail is stronger than that number, but the
honest test is a targeted one: replay `c8be87c4ba67de0b` alone, say twenty times
per checkpoint, and see whether the runaway rate separates. That has not been
run.

### Per-run detail

Full-set replays, temperature 1.0 / top_k 64 / concurrency 10 /
`NUM_SPEC_TOKENS=15`, re-pooled by `benchmark/analysis/task3_repeats.py`:

| checkpoint | run | calls | out tok | max completion | accept_len (step-w) | accept_len (per-req) | step-w, longest call dropped |
|---|---|---:|---:|---:|---:|---:|---:|
| `dflash2` | `tb-dflash2` (existing) | 1,753 | 509,389 | 9,900 | 4.0020 | 5.6748 | 4.0325 |
| `dflash2` | `tb-dflash2-repeat2` (existing) | 1,753 | 493,449 | 7,885 | 4.0045 | 5.6497 | 4.0270 |
| `dflash2` | repeat r2 (new) | 1,753 | 500,768 | 7,223 | 3.9944 | 5.6480 | 4.0153 |
| `dflash2` | repeat r3 (new) | 1,753 | 515,233 | 10,127 | 4.0292 | 5.6822 | 4.0510 |
| run-D midpoint (step 1976) | `tb-dflash2-run-d-32k-mix-step1976` (existing) | 1,753 | 513,780 | 8,204 | 3.9085 | 6.2888 | 3.9349 |
| run-D midpoint | repeat r2 (new) | 1,753 | 500,623 | 10,318 | 4.0124 | 6.3782 | 4.0469 |
| run-D midpoint | repeat r3 (new) | 1,753 | 499,925 | 9,394 | 4.0459 | 6.3805 | 4.0719 |
| run-D final (step 3956) | `tb-dflash2-run-d-32k-mix` (existing) | 1,753 | 574,004 | **58,019** | 3.8537 | 6.2957 | 3.9312 |
| run-D final | repeat r2 (new) | 1,753 | 508,234 | **18,163** | 3.9793 | 6.3892 | 4.0227 |
| run-D final | repeat r3 (new) | 1,753 | 542,703 | **21,577** | 3.9370 | 6.3853 | 3.9850 |

Both run-D checkpoints' *original* runs are the lowest of their three, which is
worth noting and not over-reading: `dflash2`'s two original runs sit in the
middle of its four, so there is no consistent old-harness-versus-new-harness
shift to explain it.

