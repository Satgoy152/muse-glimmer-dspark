# Speculator comparison — Terminal-Bench frozen 40

## How these were produced

Each drafter served the **same frozen set of 1,753 recorded Terminal-Bench calls**, replayed verbatim against its own vLLM server. The call set is the `raw` split of `Satgoy152/Muse-Glimmer-Terminal-Bench-Eval`: the frozen 40 pinned to `harbor-framework/terminal-bench-1` at `d28711d0da2675d0bb1d56de45ae5df6082438a3`, 36 trajectories, 42.3 M prompt tokens.

`NUM_SPEC_TOKENS=15` for every run — acceptance length is only comparable at the same draft length. Every run got its own `TRACE_DIR`; the DSpark patch was confirmed in the log of every `dspark` server; `--per-request-spec-decode-metrics summary` supplied the per-request block that acceptance is pooled from.

> **This is a replay, not an agentic re-run.** It measures acceptance on a fixed, representative agentic prompt distribution — which removes trajectory drift as a confound, since every drafter sees byte-identical prompts. It does **not** produce `is_resolved`: task completion needs the real harness, so the `resolved` column is *not measured* for every row here. The published on-policy baseline's 19/39 is carried for reference only.

## Pooled

| drafter | calls | out_tok | accept_len | draft_rate | tok/s | TTFT mean | resolved |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dflash2` | 1753 | 509,389 | **4.002** | 0.200 | 100.0 | 0.319 s | not measured |
| `dflash2-run-d-32k-mix-step1976` | 1753 | 513,780 | **3.909** | 0.194 | 95.0 | 0.324 s | not measured |
| `dflash-official` | 1753 | 525,505 | **3.854** | 0.190 | 98.9 | 0.338 s | not measured |
| `dflash2-run-d-32k-mix` | 1753 | 574,004 | **3.854** | 0.190 | 100.0 | 0.336 s | not measured |
| `dspark-run-a-32k` | 1753 | 495,315 | **3.804** | 0.187 | 92.7 | 0.336 s | not measured |
| `dspark-run-b-49k` | 1753 | 491,830 | **3.777** | 0.185 | 93.0 | 0.310 s | not measured |
| `dspark-community` | 1753 | 513,386 | **3.185** | 0.146 | 84.7 | — | not measured |
| *published on-policy baseline* | *1753* | *519,902* | *3.913* | *0.194* | *151.1* | *n/a* | *19/39 (48.7%)* |

**Control.** `dflash-official` replays at **3.854** against the published on-policy **3.913** (-1.5%). Close enough to trust the rest: the residual is replay-vs-on-policy, a newer vLLM nightly than the baseline was measured on, and temperature-1.0 sampling noise.

## The fine-tune's effect, isolated

`dspark-run-a-32k` against `dspark-community`, the warm start it was fine-tuned from — same architecture, same `block_size` 15, same `aux_hidden_state_layer_ids`. The only difference is the fine-tune.

| strength | community | run-a-32k | Δ | |
|---|---:|---:|---:|---|
| low | 3.295 | 4.455 | +1.159 | +35.2% |
| medium | 3.328 | 3.925 | +0.597 | +17.9% |
| high | 3.219 | 3.699 | +0.481 | +14.9% |
| xhigh | 3.040 | 3.619 | +0.578 | +19.0% |
| **pooled** | **3.185** | **3.804** | **+0.619** | +19.4% |

The fine-tune improves acceptance at **every reasoning strength**, and closes **92%** of the gap between the community warm start and the official DFlash drafter.

## Per reasoning strength

| drafter | low | medium | high | xhigh | pooled |
|---|---:|---:|---:|---:|---:|
| `dflash2` | 4.653 | 4.069 | 3.904 | 3.870 | **4.002** |
| `dflash2-run-d-32k-mix-step1976` | 4.569 | 3.927 | 3.846 | 3.769 | **3.909** |
| `dflash-official` | 4.420 | 3.872 | 3.713 | 3.792 | **3.854** |
| `dflash2-run-d-32k-mix` | 4.677 | 3.666 | 3.902 | 3.843 | **3.854** |
| `dspark-run-a-32k` | 4.455 | 3.925 | 3.699 | 3.619 | **3.804** |
| `dspark-run-b-49k` | 4.343 | 3.779 | 3.640 | 3.722 | **3.777** |
| `dspark-community` | 3.295 | 3.328 | 3.219 | 3.040 | **3.185** |
| *published baseline* | *4.345* | *4.034* | *3.796* | *3.797* | *3.913* |

Acceptance falls as reasoning strength rises, as in the baseline.

### Full breakdown per drafter

**`dflash2`**

| strength | calls | out_tok | accept_len | draft_rate | tok/s |
|---|---:|---:|---:|---:|---:|
| low | 350 | 54,945 | 4.653 | 0.243 | 98.9 |
| medium | 369 | 129,178 | 4.069 | 0.205 | 107.0 |
| high | 333 | 142,033 | 3.904 | 0.194 | 103.8 |
| xhigh | 701 | 183,233 | 3.870 | 0.191 | 93.3 |
| pooled | 1753 | 509,389 | 4.002 | 0.200 | 100.0 |

**No calls excluded** — every one of the 1,753 returned a response with a per-request metrics block. Server-counter cross-check: 4.005. Prefix-cache hit rate 95.9%.

**`dflash2-run-d-32k-mix-step1976`**

| strength | calls | out_tok | accept_len | draft_rate | tok/s |
|---|---:|---:|---:|---:|---:|
| low | 350 | 55,646 | 4.569 | 0.238 | 94.4 |
| medium | 369 | 149,268 | 3.927 | 0.195 | 104.6 |
| high | 333 | 127,021 | 3.846 | 0.190 | 94.1 |
| xhigh | 701 | 181,845 | 3.769 | 0.185 | 89.1 |
| pooled | 1753 | 513,780 | 3.909 | 0.194 | 95.0 |

**No calls excluded** — every one of the 1,753 returned a response with a per-request metrics block. Server-counter cross-check: 3.914. Prefix-cache hit rate 95.9%.

**`dflash-official`**

| strength | calls | out_tok | accept_len | draft_rate | tok/s |
|---|---:|---:|---:|---:|---:|
| low | 350 | 56,472 | 4.420 | 0.228 | 97.6 |
| medium | 369 | 158,744 | 3.872 | 0.191 | 111.4 |
| high | 333 | 135,362 | 3.713 | 0.181 | 97.0 |
| xhigh | 701 | 174,927 | 3.792 | 0.186 | 91.5 |
| pooled | 1753 | 525,505 | 3.854 | 0.190 | 98.9 |

**No calls excluded** — every one of the 1,753 returned a response with a per-request metrics block. Server-counter cross-check: 3.859. Prefix-cache hit rate 95.9%.

**`dflash2-run-d-32k-mix`**

| strength | calls | out_tok | accept_len | draft_rate | tok/s |
|---|---:|---:|---:|---:|---:|
| low | 350 | 54,673 | 4.677 | 0.245 | 95.1 |
| medium | 369 | 209,429 | 3.666 | 0.178 | 116.7 |
| high | 333 | 128,615 | 3.902 | 0.194 | 95.0 |
| xhigh | 701 | 181,287 | 3.843 | 0.190 | 90.0 |
| pooled | 1753 | 574,004 | 3.854 | 0.190 | 100.0 |

**No calls excluded** — every one of the 1,753 returned a response with a per-request metrics block. Server-counter cross-check: 3.862. Prefix-cache hit rate 95.9%.

**`dspark-run-a-32k`**

| strength | calls | out_tok | accept_len | draft_rate | tok/s |
|---|---:|---:|---:|---:|---:|
| low | 350 | 57,589 | 4.455 | 0.230 | 97.3 |
| medium | 369 | 136,451 | 3.925 | 0.195 | 100.4 |
| high | 333 | 123,715 | 3.699 | 0.180 | 91.9 |
| xhigh | 701 | 177,560 | 3.619 | 0.175 | 86.7 |
| pooled | 1753 | 495,315 | 3.804 | 0.187 | 92.7 |

**No calls excluded** — every one of the 1,753 returned a response with a per-request metrics block. Server-counter cross-check: 3.810. Prefix-cache hit rate 95.9%.

**`dspark-run-b-49k`**

| strength | calls | out_tok | accept_len | draft_rate | tok/s |
|---|---:|---:|---:|---:|---:|
| low | 350 | 55,532 | 4.343 | 0.223 | 93.8 |
| medium | 369 | 127,824 | 3.779 | 0.185 | 97.5 |
| high | 333 | 127,769 | 3.640 | 0.176 | 94.1 |
| xhigh | 701 | 180,705 | 3.722 | 0.181 | 89.2 |
| pooled | 1753 | 491,830 | 3.777 | 0.185 | 93.0 |

**No calls excluded** — every one of the 1,753 returned a response with a per-request metrics block. Server-counter cross-check: 3.772. Prefix-cache hit rate 96.0%.

**`dspark-community`**

| strength | calls | out_tok | accept_len | draft_rate | tok/s |
|---|---:|---:|---:|---:|---:|
| low | 350 | 55,663 | 3.295 | 0.153 | 78.8 |
| medium | 369 | 134,291 | 3.328 | 0.155 | 91.8 |
| high | 333 | 138,994 | 3.219 | 0.148 | 89.3 |
| xhigh | 701 | 184,438 | 3.040 | 0.136 | 79.0 |
| pooled | 1753 | 513,386 | 3.185 | 0.146 | 84.7 |

**No calls excluded** — every one of the 1,753 returned a response with a per-request metrics block.

## Acceptance by context length × reasoning strength

**`dflash2`**

| ctx | low | medium | high | xhigh |
|---|---:|---:|---:|---:|
| <2K | 5.502 | 3.248 | 3.261 | 5.470 |
| 2-8K | 4.945 | 4.102 | 4.345 | 4.051 |
| 8-16K | 5.162 | 3.784 | 3.618 | 3.659 |
| 16-32K | 4.518 | 4.111 | 3.881 | 3.726 |
| 32-64K | 4.202 | 5.310 | 5.469 | 3.927 |
| 64K+ | — | — | — | 4.042 |

**`dflash2-run-d-32k-mix-step1976`**

| ctx | low | medium | high | xhigh |
|---|---:|---:|---:|---:|
| <2K | 5.775 | 4.184 | 3.702 | 7.953 |
| 2-8K | 4.552 | 3.491 | 4.029 | 3.881 |
| 8-16K | 5.264 | 3.698 | 3.697 | 3.492 |
| 16-32K | 4.517 | 4.043 | 3.607 | 3.580 |
| 32-64K | 4.144 | 5.032 | 5.050 | 3.884 |
| 64K+ | — | — | — | 3.858 |

**`dflash-official`**

| ctx | low | medium | high | xhigh |
|---|---:|---:|---:|---:|
| <2K | 5.692 | 3.782 | 3.496 | 6.184 |
| 2-8K | 4.941 | 3.742 | 3.882 | 3.704 |
| 8-16K | 5.098 | 3.683 | 3.493 | 3.353 |
| 16-32K | 4.146 | 3.695 | 3.729 | 3.770 |
| 32-64K | 3.875 | 5.140 | 4.416 | 3.893 |
| 64K+ | — | — | — | 3.807 |

**`dflash2-run-d-32k-mix`**

| ctx | low | medium | high | xhigh |
|---|---:|---:|---:|---:|
| <2K | 5.918 | 3.330 | 3.568 | 6.050 |
| 2-8K | 4.956 | 3.538 | 4.292 | 3.690 |
| 8-16K | 5.164 | 3.433 | 3.758 | 3.516 |
| 16-32K | 4.422 | 4.045 | 3.680 | 3.742 |
| 32-64K | 4.320 | 5.471 | 5.071 | 3.997 |
| 64K+ | — | — | — | 3.917 |

**`dspark-run-a-32k`**

| ctx | low | medium | high | xhigh |
|---|---:|---:|---:|---:|
| <2K | 6.329 | 4.822 | 3.673 | 5.838 |
| 2-8K | 4.696 | 3.556 | 3.947 | 3.575 |
| 8-16K | 4.674 | 3.603 | 3.653 | 3.293 |
| 16-32K | 4.502 | 3.940 | 3.403 | 3.519 |
| 32-64K | 4.003 | 5.382 | 5.245 | 3.776 |
| 64K+ | — | — | — | 3.579 |

**`dspark-run-b-49k`**

| ctx | low | medium | high | xhigh |
|---|---:|---:|---:|---:|
| <2K | 5.397 | 3.277 | 3.287 | 6.025 |
| 2-8K | 4.417 | 3.572 | 3.904 | 3.533 |
| 8-16K | 4.541 | 3.580 | 3.320 | 3.301 |
| 16-32K | 4.476 | 3.682 | 3.577 | 3.649 |
| 32-64K | 3.967 | 5.355 | 4.972 | 3.899 |
| 64K+ | — | — | — | 3.920 |

**`dspark-community`**

| ctx | low | medium | high | xhigh |
|---|---:|---:|---:|---:|
| <2K | 4.546 | 3.244 | 3.026 | 5.062 |
| 2-8K | 3.670 | 3.040 | 3.605 | 3.228 |
| 8-16K | 3.795 | 3.248 | 2.925 | 2.843 |
| 16-32K | 3.051 | 3.287 | 3.219 | 3.004 |
| 32-64K | 2.862 | 4.012 | 3.726 | 3.013 |
| 64K+ | — | — | — | 2.955 |

## Caveats

These are carried from `benchmark/terminal_bench/README.md` and must not be dropped when these numbers are quoted.

**Two denominators, quoted separately.** Task completion is over **39 tasks**; traces are over **36 trajectories**, of which 38 tasks issued any call at all. They are not the same denominator and averaging them is wrong.

* `word2vec-from-scratch` **cannot build at the pinned commit** — an unpinned `huggingface_hub` now rejects the Dockerfile's `load_dataset('wikitext', …)` (the dataset moved to `Salesforce/wikitext`). The agent never runs, so this is upstream task rot that fails for *any* model, and it is excluded from the denominator (`resolved: null`).
* `extract-safely` issued **zero model calls** across the whole recording. It failed before any inference, but TB recorded it as an ordinary unresolved task (`failure_mode: unset`) rather than an agent error — so it counts as a failure in 19/39 *without having exercised the model at all*.
* `llm-inference-batching-scheduler` and `amuse-install` were removed by the canary filter. They ran, and still count toward completion.

**tok/s reflects contention as much as the drafter.** These runs used concurrency **10**, not the baseline's `--concurrent 8`, so the tok/s and TTFT columns are **not comparable to the published 151.1**. Acceptance length is the contention-independent number and is unaffected.

**Prefix caching was on** (`--enable-prefix-caching`), hit rates ~96%. That flatters TTFT and tok/s relative to a cold run, identically across drafters. Acceptance length is unaffected either way.

**TTFT is the server-side histogram**, not the client's first delta. `--enable-auto-tool-choice` makes the tool-call parser buffer deltas until a call is whole, so a client-measured TTFT is ≈ end-to-end on tool-calling requests and is wrong. The published baseline never measured TTFT at all.

**Server-global counters are used only as a cross-check.** They cannot be attributed in general, but each endpoint here served exactly one drafter and one workload. Where they appear they agree with the per-request pooling to within 0.01.

**Canary: 0 for every trace.** `filter_canary.py` reported 0 contaminated trajectories and 0 remaining occurrences on all runs; the published `raw` split was already canary-clean at source.

**Heavy-tail output lengths move the pooled number more than run-to-run noise does.** Replaying `dflash2` twice, unchanged, gives 4.0047 vs 4.0101 on server counters -- a noise floor of ~0.005. A single runaway generation moves it much further: `dflash2-run-d-32k-mix` drew one 58,019-token completion (every other run peaks at 8-10K), which by itself explains its 574,004 out_tok and pulls its pooled acceptance from 3.931 down to 3.854. Trimming is not the fix -- across the two `dflash2` replays the untrimmed number is the most stable (delta 0.0025) while drop-1 and drop-5 are worse (0.0055 and 0.0252). Quote the untrimmed pooled value, and check max(completion_tokens) before calling a sub-0.01 difference real. Where a run has an elevated count of responses with an empty per-request metrics block, prefer the server counters: they cover every call, while per-request pooling silently omits those.

**The two `dflash2-run-d-*` rows were served without `output_multiplier` and `final_logit_softcapping`.** vLLM rebuilds `dflash_config` from scratch when it loads a speculators-format checkpoint (`transformers_utils/configs/speculators/algos.py::update_dflash`), forwarding only `mask_token_id`, `target_layer_ids`, `sample_from_anchor` and `causal`, plus four conv/selector keys added by `update_dflash2`. Both fields are present in those checkpoints own `config.json` but never reach the model, so `qwen3_dflash2.py` falls back to `scale=1.0` (against 0.196) and `softcap=0.0` (against 20.0). The native `dflash2` row carries its own `dflash_config` and is unaffected, so this table compares native-with-scale against converted-without. Measured cost: re-serving `step1976` with both restored moved it 3.9139 to 3.9191 on server counters, i.e. +0.005, at the noise floor. That is the expected result -- the proposal method is `greedy` and both transforms are monotonic, so argmax drafting is invariant to them -- so the rows remain comparable. The asymmetry is still a real bug and worth fixing upstream.

**Adding a fifth drafter** is one invocation and a regenerate:

```bash
NAME=dspark-run-b-49k SPEC=/mnt/data/speculators/dspark-run-b-49k \
  METHOD=dspark bash scripts/eval_replay.sh
python benchmark/terminal_bench/buckets.py
python benchmark/terminal_bench/make_summary.py
```
