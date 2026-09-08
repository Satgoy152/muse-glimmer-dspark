# All speculator results to date

See [`RESULTS-turn-length.md`](RESULTS-turn-length.md) for the consolidated
turn-length, reasoning-share, later-checkpoint, and length-aligned throughput
analysis.

Everything measured on the preemptible 1x H200 node against target
`meta-models/Muse-Glimmer-30B` (BF16), `num_speculative_tokens=15` (16-token
block), vLLM `specd:latest`. `accept_len = 1 + accepted/steps`, pooled.

Three regimes, **not comparable to each other**:

| eval | prompts | temperature | concurrency |
|---|---|---|---|
| Terminal-Bench replay | frozen 1,753-call set | 1.0, top_k 64 | 10 |
| HumanEval | 164 | 0 (greedy) | 1 |
| MBPP | 500 (standard test split) | 0 (greedy) | 1 |

## Acceptance length

Terminal-Bench is quoted from the **server `/metrics` counters**, which cover
every call; per-request pooling silently omits calls that returned an empty
metrics block, and on these runs that is worth up to 0.03. The coding
benchmarks are quoted from per-request pooling, which agreed with the counters
to +0.0000 on all ten runs.

| Drafter | Terminal-Bench | HumanEval | MBPP |
|---|---|---|---|
| `dflash-official` | 3.859 | 4.853 | 4.803 |
| **`dflash2`** (native, unmodified) | **4.005** | **5.748** | **5.436** |
| `dspark-community` | 3.185 † | 4.658 | 4.528 |
| `dspark-run-a-32k` *(ours)* | 3.810 | 4.541 | 4.440 |
| `dspark-run-b-49k` *(ours)* | 3.772 | — | — |
| `dflash2-run-d-32k-mix` *(ours)* | 3.862 | — | — |
| `dflash2-run-d-...-step1976` *(ours)* | 3.914 | 5.299 | 5.054 |
| └ same checkpoint, scale-fixed (control) | 3.919 | not run | not run |

† per-request pooling; this run has no `/metrics` scrape.

**Unmodified `dflash2` wins on all three.** Published reference for context
(different target model, so only direction transfers): HumanEval 4.11 / 4.33 /
5.66 and MBPP 3.74 / 4.02 / 5.30 for DFlash / DSpark / DFlash2.

## What the fine-tunes actually did

**DSpark fine-tune is a large win on the deployment workload and a small loss
off it.** `dspark-run-a-32k` beats `dspark-community` by **+0.619** on
Terminal-Bench (3.804 vs 3.185) while losing 0.118 on HumanEval and 0.088 on
MBPP. That is textbook specialisation toward the agentic trace distribution it
was trained on, and Terminal-Bench is the workload we care about — so this
fine-tune is a keeper. The 49k variant is slightly worse than the 32k one
(3.777).

**DFlash2 fine-tune is not.** It sits 0.091 below the native baseline on
Terminal-Bench (3.914 vs 4.005), ~17x the noise floor, and 0.45 / 0.38 below on
HumanEval / MBPP. The logit-scale bug does not explain it: see below.

## The DFlash2 logit-scale bug is real, and does not move acceptance

vLLM's `update_dflash2` drops `output_multiplier` (0.196) and
`final_logit_softcapping` (20.0) when rebuilding config for a *converted*
speculators-format checkpoint; `qwen3_dflash2.py` then defaults to `scale=1.0`,
no cap. Native `dflash2` ships its own `dflash_config` and is unaffected, so
every comparison of our converted fine-tune against it was served asymmetrically.
Fixed in `docker/serve_patched.sh`.

**It is nonetheless acceptance-neutral, so the rows above stand.** Re-serving
`step1976` with both values restored moved it 3.9139 -> 3.9191 on server
counters: **+0.005, at the 0.005 noise floor.** That is the expected result --
drafting selects by argmax/top-k, and a positive scale and a tanh soft cap are
both monotonic, so they cannot change which tokens are proposed. The
measurement was never the problem; the checkpoint is simply worse than its
baseline.

The bug still matters for **training**, where the loss is not invariant to it
(`patches/apply_dflash2_output_shaping.py` records 0.406 vs 2.155 at step 0),
and it is worth fixing upstream regardless.

## pass@1

HumanEval **0.878**, MBPP **0.710** — and identical across all five drafters,
because greedy speculation is lossless: 161–163/164 and 490–493/500 outputs are
byte-identical between drafters. pass@1 is a property of the target model here,
not a drafter differentiator. MBPP's ceiling is 0.994 (3 of its own reference
solutions fail their own tests).

## Latency

**Client-side TTFT/TPOT are invalid on every eval** — `--enable-auto-tool-choice`
makes the tool-call parser buffer deltas, so the first delta lands near the end
(TB: client TTFT p50 1.27 s against e2e 1.30 s; HumanEval 3.0 s against 3.8 s).
Server-side histograms are recorded inside the engine and are unaffected.

Server-side p50, `benchmark/prom_latency.py` (full table in
`results/latency/`):

| eval | TTFT p50 | ITL p50 | TPOT p50 |
|---|---|---|---|
| Terminal-Bench, concurrency 10 | 0.238–0.252 s | 0.035 s | 0.0068 s |
| Terminal-Bench, concurrency 64 | 0.347–0.373 s | 0.044 s | 0.0099–0.0181 s |
| HumanEval, concurrency 1 | 0.043–0.046 s | 0.0175 s | 0.0050 s |
| MBPP, concurrency 1 | 0.035–0.036 s | 0.0175 s | 0.0050 s |

vLLM's ITL/TPOT buckets are too coarse to separate drafters within one regime;
end-to-end throughput is the usable speed comparison (`dflash2` is 1.16x
`dflash-official` on HumanEval, 296 vs 255 tok/s).

## Where the GPU time goes (`dflash2`, separate greedy 512-token profile)

| phase | share of kernel time |
|---|---|
| target forward | 83% |
| draft forward (one pass for the whole 16-token block) | 12% |
| target LM head | 3.5% |
| everything else (projection, KV insert, rejection sampling) | ~1.2% |

Stable across 10x concurrency and 10x context length. Throughput 306 tok/s
(short, c1) to 1,541 tok/s (long, c10); accept_len 6.03–8.23 on that workload.
**Target verification dominates; the entire drafter is an eighth of GPU time**,
so a second draft pass breaks even at only a +12% acceptance gain — see
`docs/DECISION-second-draft-pass.md`.

## Things that would change a number

- **Noise floor ~0.005** on pooled acceptance, measured by replaying `dflash2`
  twice unchanged (4.0047 vs 4.0101 on server counters).
- **A single runaway generation outweighs that noise.**
  `dflash2-run-d-32k-mix` drew one 58,019-token completion where every other
  run peaks at 8-10K; it alone pulls that row's per-request pooled acceptance
  from 3.931 to 3.854. Check `max(completion_tokens)` before believing a
  sub-0.01 difference, and prefer the server counters where a run has many
  empty per-request metrics blocks.
- **Acceptance is not contention-independent.** Concurrency 10 -> 64 adds
  +0.014 to +0.070 depending on drafter, so concurrency is pinned and reported.
- **Truncation at the 2048-token cap** on the coding benchmarks: 20/164
  HumanEval generations (21.9% of tokens) and 133/500 MBPP (43.4%). Excluding
  them raises every row by +0.10 to +0.35 and changes no ranking. The model is
  simply verbose here — median completion is 1,024 (HumanEval) and 1,142 (MBPP)
  tokens for answers needing a few dozen.
- **Acceptance cross-check**: per-request blocks vs the server's `/metrics`
  counters agreed to +0.0000 on all ten coding-benchmark runs.

## Where things live

- `docs/coding_benchmarks.md` — HumanEval/MBPP detail; `docs/RESULTS-profiling-dflash2.md` — profile detail
- `results/terminal_bench/`, `results/coding_benchmarks/`, `results/latency/`
- harnesses: `scripts/eval_replay.sh` (TB), `scripts/eval_humaneval.sh` (HumanEval/MBPP), `benchmark/profiling/run_all.sh`
