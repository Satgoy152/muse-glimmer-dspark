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
  - 2000 instances across 1106 repos, sampled under a **per-source quota** — 1000 slots each to
    SWE-Gym and SWE-bench-extra. Round-robin over repos alone collapses onto SWE-bench-extra's
    1974-repo long tail and starves SWE-Gym, which contributes the large mature repos (pandas,
    mypy, MONAI, moto) that produce long-horizon, deep-context trajectories.
  - Within each source, a **per-repo cap** of 5% of that source's quota keeps any single repo from
    dominating. Small repos that exhaust below the cap are backfilled, preferring
    under-represented repos.
  - Deduplicated on `instance_id` before sampling: the two sources overlap on `iterative/dvc` and
    `pydantic/pydantic`, and 111 duplicates were dropped.
- Eval: Terminal-Bench, frozen 40-task subset (To be released)
  - 40 of 241 tasks, allocated proportionally across difficulty — 11 easy / 20 medium / 9 hard —
    so the subset keeps the full benchmark's difficulty mix rather than over-weighting one band.
  - Within each difficulty band, tasks are drawn **round-robin over category**, cycling through
    categories one pick at a time so no single category dominates a band. 21 of the 26 categories
    are represented. Category labels are normalized first, since the source uses several spellings
    for the same category (`game`/`games`, `math`/`mathematics`, `file_operations`/`file-operations`).
- Leakage control: the training and eval sets are checked for repo overlap before sampling. Every
  GitHub repo referenced by any of the 241 Terminal-Bench task instructions (9 repos) is excluded
  from the training pool. Zero instances matched, so the sets were already disjoint; the check
  stays in the pipeline so it holds if either set changes. Repo overlap is a weak signal on its
  own — Terminal-Bench tasks are self-contained container tasks, not repo-derived — so the
  stronger check is the Terminal-Bench canary GUID carried by 240/241 tasks: any generated trace
  containing it is a leak and gets dropped.
