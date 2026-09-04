# Coding-Specialized DSpark Speculator for Muse Glimmer 30B

## Motivation

Existing speculators for Muse Glimmer are trained on general chat data (OpenPerfectBlend). None have been trained for long-horizon agentic coding, and the community DSpark model card lists agentic decoding as unbenchmarked. Its weakest published subset is tool-call (3.087 → 3.805 accepted length).

This project fine-tunes a speculator on curated on-policy coding and agentic traces, with the goal of raising acceptance length and throughput on that workload relative to generic speculators. Terminal-Bench is used as the evaluation proxy for a coding customer.

## Models

- Target: `meta-models/Muse-Glimmer-30B` (BF16, 30B)
- Official DFlash drafter (baseline): `meta-models/Muse-Glimmer-30B-assistant` (5-layer)
- Community DSpark (warm start): [DaoCloud/Muse-Glimmer-30B-DSpark](https://huggingface.co/DaoCloud/Muse-Glimmer-30B-DSpark)
- DFlash2 (baseline): [z-lab/Muse-Glimmer-30B-DFlash2](https://huggingface.co/z-lab/Muse-Glimmer-30B-DFlash2)

## Data

- Replay: [DaoCloud/Muse-Glimmer-OPB-100K](https://huggingface.co/datasets/DaoCloud/Muse-Glimmer-OPB-100K) — 99,984 conversations / 148,900 rows, pre-tokenized, on-policy Glimmer
- New training traces: [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) over SWE-Gym, repos disjoint from eval (To be released)
  - 2000 instances, balanced round-robin across SWE-Gym's 11 repos.
  - Every instance is verified with a pre-built docker image.
  - **Limitation:** SWE-bench-extra would have added ~1100 more repos (increased data variation), but it has no published images.
- Eval: Terminal-Bench — [Satgoy152/Muse-Glimmer-Terminal-Bench-Eval](https://huggingface.co/datasets/Satgoy152/Muse-Glimmer-Terminal-Bench-Eval) (private)
  - 40 of 241 tasks, mixed across difficulty (11 easy / 20 medium / 9 hard).
  - Within each difficulty band, tasks are drawn **round-robin over category** (e.g., `games`, `math`, `file_operations`).
- Leakage control: the training and eval sets are checked for repo overlap before sampling.

## Baseline

Official DFlash drafter at 15 speculative tokens, measured on the frozen 40 with
`--concurrent 8`: **pooled acceptance length 3.913**, draft acceptance rate
0.194, 151.1 tok/s over 1,753 calls, and 19/39 tasks resolved (48.7%).
Acceptance falls as reasoning strength rises, from 4.345 at `low` to 3.797 at
`xhigh`. Per-segment numbers, caveats and coverage:
[benchmark/terminal_bench/README.md](benchmark/terminal_bench/README.md).

## Gathering Traces

- **Trace Generation Workflow:** See [docs/generate.md](docs/generate.md) for environment setup, container management, and running the recording proxy.
- **Inference Endpoint Hosting:** See [docs/inference_hosting.md](docs/inference_hosting.md) for self-hosting the vLLM endpoint on Nebius (1x H200 GPU) with speculative decoding.
- **Terminal-Bench Eval Run:** See [benchmark/terminal_bench/README.md](benchmark/terminal_bench/README.md) for the eval harness,
  the in-container agent, and the canary check.
- **Datasets & Manifests:** See [docs/data.md](docs/data.md) for details on benchmark tasks and training instances.

## Training

- **Speculator Training:** See [docs/train.md](docs/train.md) for the seq-length
  budget, the render/extraction server, and the DSpark fine-tune. Warm start is
  `DaoCloud/Muse-Glimmer-30B-DSpark`; `block_size` is pinned to 15 so acceptance
  length stays comparable to the baseline above.
