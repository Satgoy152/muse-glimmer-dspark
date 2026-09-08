# All speculator results to date

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

| Drafter | Terminal-Bench | HumanEval | MBPP |
|---|---|---|---|
| `dflash-official` | 3.854 | 4.853 | 4.803 |
| **`dflash2`** (native, unmodified) | **4.002** | **5.748** | **5.436** |
| `dspark-community` | 3.185 | 4.658 | 4.528 |
| `dspark-run-a-32k` *(ours)* | 3.804 | 4.541 | 4.440 |
| `dspark-run-b-49k` *(ours)* | 3.777 | — | — |
| `dflash2-run-d-32k-mix` *(ours)* | 3.854 | — | — |
| `dflash2-run-d-...-step1976` *(ours)* | 3.909 | 5.299 ⚠️ | 5.054 ⚠️ |
| └ same checkpoint, **scale-fixed** | **3.949** | not run | not run |

⚠️ = served at the wrong logit scale (see "The DFlash2 confound").

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

**DFlash2 fine-tune is not.** Even with the logit-scale bug fixed it reaches
3.949 against the 4.002 native baseline — still 0.053 short, ~20x the noise
floor. The scale fix is real but only recovers +0.040 of the 0.093 gap.

## The DFlash2 confound

vLLM's `update_dflash2` drops `output_multiplier` (0.196) and
`final_logit_softcapping` (20.0) when rebuilding config for a *converted*
speculators-format checkpoint; `qwen3_dflash2.py` then defaults to `scale=1.0`,
no cap. Native `dflash2` ships its own `dflash_config` and is unaffected — so
any comparison of our converted fine-tune against it was asymmetric, at a 5.1x
logit mismatch. Fixed in `docker/serve_patched.sh`.

The Terminal-Bench control (`control-dflash2-run-d-step1976-scalefix`) is the
only re-run done with the fix. **The two coding-benchmark rows were never
re-run and remain unmeasured** — about one GPU-hour to settle.

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

- **Noise floor 0.0025** on pooled acceptance, measured by replaying `dflash2`
  twice (4.0020 vs 4.0045). Differences below ~0.005 are not real.
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
