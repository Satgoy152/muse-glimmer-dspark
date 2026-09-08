# DFlash2 profile: where the GPU time goes

Measured 2026-09-08 on the preemptible 1x H200 node, vLLM
`0.28.1rc1.dev451+g1970f3ed4`, torch `2.13.0+cu130`, driver `580.173.02`.
Target `meta-models/Muse-Glimmer-30B` (BF16), drafter
`/mnt/data/speculators/dflash2`, `num_speculative_tokens=15` (16-token
speculative block). V2 model runner, `FULL_AND_PIECEWISE` CUDA graphs, prefix
caching on. Reproduce with `benchmark/profiling/run_all.sh`.

This is a **speed profile, not a quality evaluation**. Nothing here measures
whether the drafter proposes *good* tokens.

## Workload

Twenty real Terminal-Bench requests (`/mnt/data/eval/raw.parquet`, reasoning
strength `high`), re-rendered through the server's own `/tokenize` and replayed
as raw prompt token IDs against `/v1/completions` so no chat parser sits in the
timing path. Ten short prompts (1568-2281 rendered tokens) and ten long ones
(21327-23222). Greedy, `ignore_eos=true`, exactly 512 output tokens per
request, prefixes warmed before every timed wave.

An "engine step" below is one target verification of a 16-token speculative
block, which emits `accept_len` accepted tokens.

## Headline numbers

| case | tok/s | accept_len | p50 latency | ms/engine step | GPU busy |
|---|---|---|---|---|---|
| short, concurrency 1 | 305.7 | 6.03 | 1.241 s | 19.64 | 96.9% |
| short, concurrency 10 | 969.2 | 6.04 | 1.670 s | 21.19 | 97.7% |
| long, concurrency 1 | 406.5 | 8.23 | 1.061 s | 20.03 | 96.9% |
| long, concurrency 10 | 1540.5 | 7.94 | 1.553 s | 23.68 | 97.7% |

Throughput is output tokens over wave makespan, not the sum of request
latencies.

## Where the time goes

Share of traced GPU kernel time, from a bounded 50-step steady-state decode
window per case. These six phases cover 99.65% of kernel time in every case.

| phase | short-c1 | short-c10 | long-c1 | long-c10 |
|---|---|---|---|---|
| target forward (full CUDA graph) | 82.9% | 82.9% | 83.0% | 83.7% |
| draft forward (full CUDA graph, one pass) | 12.1% | 12.1% | 12.0% | 11.5% |
| target LM head | 3.5% | 3.4% | 3.5% | 3.3% |
| draft feature projection (eager) | 0.62% | 0.55% | 0.61% | 0.53% |
| draft context KV insert (eager) | 0.30% | 0.35% | 0.29% | 0.33% |
| rejection sampling | 0.24% | 0.28% | 0.20% | 0.29% |

The split barely moves across a 10x concurrency change and a 10x context-length
change. **Target verification dominates; the entire drafter is about an eighth
of GPU time.**

The drafter proposes the whole 16-token block in a **single** forward pass, not
15 sequential ones: the draft graph is launched once per engine step and
contains 115 kernels, about 23 per drafter layer. Fifteen sequential passes
would be roughly 1700 kernels, and 115/15 = 7.7 kernels per pass is impossible
for a 5-layer model. This is what makes a second draft pass cheap -- see
`docs/DECISION-second-draft-pass.md`.

The two eager phases the earlier handoff flagged as suspects -- draft context KV
insertion and feature projection -- together account for under 1% of GPU time.
They are not worth optimising. Their *host* NVTX ranges are much longer (about
20 ms and 6 ms per 50 steps), but host range duration includes waiting and is
not GPU execution time.

## Two things that look like problems and are not

**Throughput varies 6x between prompts; per-step time does not.** Three short
prompts at concurrency 1, before averaging:

| prompt | accept_len | ms/step | tok/s |
|---|---|---|---|
| short-00 | 5.22 | 19.4 | 266 |
| short-01 | 2.05 | 19.3 | 106 |
| short-02 | 12.97 | 19.8 | 645 |

Step time is constant within 3%. All the throughput spread is accepted-length
spread, which belongs to the drafter and the prompt, not to the serving stack.
Quote ms/step when discussing system speed; quote accept_len separately.

Note that `ignore_eos=true` inflates accept_len: forcing generation past the
natural stop drives the model into repetition, which the drafter predicts almost
perfectly (that is the 12.97). Fixed-length greedy output is the right control
for a speed microbenchmark, but **the accepted lengths in this document are an
upper bound and are not a quality result.**

**The GPU is not starved.** GPU busy is 96.9-97.7% of the capture window, so
roughly 3% idle. CPU launch overhead is not the bottleneck at either
concurrency, and CUDA graphs are doing their job: 95-96% of traced kernels are
graph nodes. The residual gap is a trace-activity gap; it does not by itself
establish a cause.

## Kernel shapes explain the concurrency scaling

At concurrency 1 the target forward is a stack of very skinny GEMMs -- N=16, the
speculative block -- and several appear in `splitK` variants, which is what a
kernel does to find parallelism when the shape is too thin:

- `nvjet_sm90_tst_192x16_...` 39.1% of kernel time
- `nvjet_sm90_tst_256x16_..._splitK` 20.8%
- `nvjet_sm90_tst_512x16_...` 6.9% (this one is the LM head, 202048 vocabulary)

At concurrency 10 the same GEMMs become N=160 (10 requests x 16 tokens) and
attention grows with context:

- `nvjet_sm90_tst_192x160_..._coopB` 20.1%
- `nvjet_sm90_tst_256x160_..._coopA_splitK` 10.7%
- flash-attention cutlass kernels ~11% combined

So batching costs only 8-18% more per engine step while doing 10x the work, and
throughput rises about 3.2x (306 -> 969 short, 406 -> 1541 long). **Occupancy at
low concurrency is the structural inefficiency here, not any single phase.**

## What this implies for optimisation

Hypotheses, not measured wins. Nothing below has been tried.

1. Making the drafter cheaper has at most ~12% of GPU time to recover, and only
   if accepted length is unchanged. Effort is better spent on accepted length,
   which moves throughput linearly and is already worth 6x between prompts.
2. The target LM head is 3.3-3.5% on its own, outside the target CUDA graph, on
   a 202048-token vocabulary. That is the largest single non-graph phase.
3. Shrinking the speculative block would make the batch-1 GEMMs thinner still,
   in the direction the shape data says is already bad. Growing it costs more
   per step and only pays if acceptance holds up.

## Caveats

- **Absolute times are from a traced run.** Per-step time inside the capture is
  within 1% of unprofiled at concurrency 1 and within 6-14% at concurrency 10,
  so shares and shapes are trustworthy. Whole-wave throughput during profiling
  is 43-77% lower, but that is dominated by fixed Nsight start/stop cost, not by
  per-step slowdown -- do not quote it as tracing overhead.
- `target_graph` appears as 100 NVTX ranges per 50 steps because
  `ModelCudaGraphManager.run_fullgraph` nests inside the inherited
  `CudaGraphManager.run_fullgraph`. Both carry the same label and the analyser
  attributes to the innermost match, so the nesting does not double-count GPU
  time. It is not two graph replays per step.
- NVTX recorded inside a CUDA graph's capture does not replay, so subphase
  attribution *within* either full graph is not available from this trace.
- 4-16 kernels per capture (<0.05 ms total) had no matching runtime launch and
  are reported as `unclassified`.
- Concurrency-10 waves finish ragged: with fixed 512 tokens and unequal
  acceptance, requests need different step counts, so the batch shrinks toward
  the end of a wave. Engine steps per wave are taken from the slowest request.

## Artifacts

On the node, under `/mnt/data/profiles/dflash2-20260908/`:

- `capture.{1..4}.nsys-rep` and `.sqlite` -- one report per capture range, in
  order short-c1, short-c10, long-c1, long-c10. Nsight splits captures itself
  when `--capture-range-end=repeat` is used; each holds a single burst.
- `prompts.json` -- rendered prompt token IDs. **Contains prompt content; not
  committed.** `prompts.metadata.json` holds the non-content audit trail.
- `runs/<case>/` -- full per-request records and Prometheus snapshots.

In this repo, under `benchmark/profiling/results/`: the per-case analysis JSON
and the run summaries, with no prompt content.
