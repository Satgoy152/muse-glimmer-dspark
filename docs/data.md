# Datasets

All manifests and dataset files are frozen by `SEED = 20260830` and included directly in the repository.

## Benchmark — `benchmark/terminal_bench/tasks_frozen40.json`

40 of 241 Terminal-Bench tasks, pinned to tree `d28711d0` of `harbor-framework/terminal-bench-1`
(the `laude-institute/terminal-bench` URL now redirects there).

- Stratified proportionally by difficulty (11 easy / 20 medium / 9 hard), then round-robin over
  category so no category dominates a difficulty band. 21 categories represented.
- Category labels in the source are inconsistent (`game`/`games`, `math`/`mathematics`,
  `file_operations`/`file-operations`); normalized before sampling.
- 240/241 tasks carry the Terminal-Bench canary string. Any generated trace containing that GUID
  is a leak and must be dropped.

The 40 frozen tasks are tracked directly in `benchmark/terminal_bench/tasks_frozen40.json`.

## Training — `data/training/train_instances.jsonl`

2000 instances for trace generation, from 1106 repos.

| Source | Instances | Repos | Role |
|---|---|---|---|
| SWE-Gym | 887 | 11 | large mature repos — long-horizon, deep-context trajectories |
| SWE-bench-extra | 1113 | ~1100 | long tail — breadth |

- 111 duplicate `instance_id`s dropped.
- Leakage exclusion: every repo referenced by *any* of the 241 Terminal-Bench tasks (9 repos), not
  just the frozen 40.

The instance metadata is tracked directly in `data/training/train_instances.jsonl`, and the full problem dataset formatted for `mini-swe-agent` (`datasets.load_dataset`) is tracked in `data/training/swegym_2k/train.jsonl`.

## Getting traces onto the GPU box

Recommendation: **push traces to a private HF dataset repo from the CPU generation box, pull with
`hf download` on the Nebius node.**

- Survives preemption and re-provisioning — the node is cattle, the dataset repo is not.
- Resumable and content-addressed, so a half-finished pull costs nothing to retry.
- Versioned, so a training run can pin a revision SHA and stay reproducible.
- It is the same artifact released publicly at the end, so the release path gets exercised early.
- Size is not a problem: ~2000 trajectories at ~60k context is ~120M tokens, roughly 0.6 GB of
  `int32` `input_ids` plus a `uint8` `loss_mask`, with raw JSON in single-digit GB.

Push tokenized shards as parquet, and the raw request/response JSON as a second config in the same
repo — the raw side is needed for the leakage audit and for re-tokenizing without regenerating.

Two things this does *not* cover:

- **Offline hidden-state extraction (~2 TB).** Do not put that on HF. Write it to Nebius Object
  Storage in the same region as the GPU node and mount it, or keep it on the persistent volume.
  It is regenerable from the traces, so it never needs to be durable across projects.
- **Checkpoints.** Persistent volume with `--checkpoint-freq 0.1`, mirrored to HF only at the end.

Create the dataset repo and prove the round trip before generation starts, not after.
