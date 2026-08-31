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
- New training traces: agent scaffold over SWE-Gym / SWE-bench-extra, repos disjoint from eval (To be released)
  - 2000 instances across 1106 repos, 1000 slots each from SWE-Gym and SWE-bench-extra.
  - Deduplicated on `instance_id` before sampling.
- Eval: Terminal-Bench, frozen 40-task subset (To be released)
  - 40 of 241 tasks, mixed across difficulty (11 easy / 20 medium / 9 hard).
  - Within each difficulty band, tasks are drawn **round-robin over category** (e.g., `games`,
    `math`, `file_operations`). 21 of the 26 categories are represented.
- Leakage control: the training and eval sets are checked for repo overlap before sampling. Every
  GitHub repo referenced by any of the 241 Terminal-Bench task instructions are excluded from the training pool.
