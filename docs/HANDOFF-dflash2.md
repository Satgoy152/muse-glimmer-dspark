# Run D (DFlash2) — state at 2026-09-07

**The run completed and regressed on Terminal-Bench.** Acceptance length:
baseline 4.02, mid checkpoint 3.95, final 3.80 — worse, monotonically with
training — while `val/accept_len_epoch` reached 8.093 against run A's 7.559.

**Do not tune anything until the control experiment below has been run.** The
leading hypothesis is that the eval served on stock vLLM, which silently drops
`output_multiplier` and `final_logit_softcapping` from a converted checkpoint.
Full analysis in `docs/train.md`, "Run D happened, and it regressed on
Terminal-Bench".

## Next step, in order

1. On the eval host: `grep -c output_multiplier .../vllm/transformers_utils/configs/speculators/algos.py`.
   `0` means the eval measured a 5.1x logit mismatch, not the fine-tune.
2. Serve the converted-but-**untrained** checkpoint against native z-lab on the
   same harness. This is the control that was never run and it separates
   "conversion/serving is lossy" from "training made it worse".
3. Only then look at the data mix (35% xhigh vs a 10% corpus, never validated
   against Terminal-Bench's actual distribution) and the LR.

The GPU node is gone; its volume with the other three checkpoint snapshots went
with it. Two checkpoints survive on HF — see "Artifacts" in `docs/train.md`.

## The warm-start blocker is resolved

`--from-pretrained z-lab/Muse-Glimmer-30B-DFlash2` fails with
`NotImplementedError: Loading a non-speculator model config is not supported yet.`
because the repo is native z-lab format (no `speculators_model_type`).

Converting is now done, with the two output-shaping knobs preserved on **both**
sides — the "preserve them" option in `docs/train.md`:

- `scripts/convert_dflash2.py` → `/mnt/data/runs/dflash2-speculators`
  (85 tensors, `speculators_model_type: dflash2`, `speculative_tokens: 15`,
  `aux_hidden_state_layer_ids [2,14,26,38,50]`, `output_multiplier` and
  `final_logit_softcapping` on the config).
- `patches/speculators-dflash2-output-shaping.patch`, baked into
  **`specd:dflash2`** (`docker/Dockerfile.dflash2`). Rollback: `specd:latest`.
  torch 2.13.0+cu130 / vllm 0.28.1rc1.dev437+ge962733e0 / transformers 5.16.1
  asserted unchanged in the build.
- `docker/serve_patched.sh` forwards both keys through vLLM's `update_dflash2`,
  sed-and-verified, idempotent. Serving/eval only; training does not need it.

Verified end to end: `--dump-config` resolves, and a 20-step run on GPUs 4-7
against the live extraction server warm-starts to `num_layers=5`,
`draft_arch=qwen3`, `draft_vocab_size=202048` and logs
`accept_rate 0.535 / accept_len 6.859 / position_1_acc 0.911` at the first
step. Patched vs unpatched at step 0: `loss` 0.406 vs 2.155, `selector_loss`
0.287 vs 1.857. Table and reasoning in `docs/train.md`.

Use `specd:dflash2` for this run, not `specd:latest`.

## What is already settled (do not re-derive)

- DFlash2's aux layers are the **same** as DSpark's: z-lab's
  `[1,13,25,37,49]` is `+1`-offset from `[2,14,26,38,50]`. K=6, BLOCK_SIZE 256.
  **The extraction server needs no reconfiguration.** Verified on the wire:
  `(64, 6, 6656) bfloat16`, token ids match, KV cache 60,392 tokens.
- DFlash2 logs `accept_rate` / `accept_len` / per-position accuracy (it
  delegates to DSpark's metrics), plus selector metrics.
- `configs/train_dflash2.yaml` is written and `--dump-config` resolves clean.
- `tensorboard` was missing from `specd:latest`; added and retagged.
  Rollback tag `specd:pre-tb`. Recipe: `docker/Dockerfile.tensorboard`.
  torch 2.13.0+cu130 / vllm 0.28.1rc1.dev437 / transformers 5.16.1 unchanged.
- `--save-best` is store_true; `save_best: false` only settable in YAML.

## Data: re-render at 49,152 was IN PROGRESS

Run A's `/mnt/data/runs/data` (25,000 rows @ 32,768) **cannot** be reused for a
48k run: turns over 32,768 were skipped at render time and are not in the set.

- `scripts/render48k.sh` — resumable, idempotent, one marker per shard.
- Progress at stop: **6 of 32 shards** in `/mnt/data/runs/prepared-48k`
  (9.3 GB), markers in `/mnt/data/runs/state-48k`.
- Resume by re-running the same container command; finished shards are skipped.
- Rate ~2.9 min/shard, ~4,700 rows/shard, so ~93 min total, ~150k rows.

Then merge, balanced by reasoning strength:

```bash
python3 scripts/merge_prepared_balanced.py \
  --shard-dir /mnt/data/runs/prepared-48k \
  --out /mnt/data/runs/data-48k --max-samples 30000
```

At 49,152 there are ~15k xhigh rows against the 7,500 a balanced 30k needs, so
a true four-way split is reachable. (The 32,768 set was 31.3/30.9/27.3/10.4.)

## Resume commands

```bash
# 0. convert the warm start (once; already done, /mnt/data/runs/dflash2-speculators)
sudo docker run --rm --network host -v /mnt/data:/mnt/data -e HF_HOME=/mnt/data/hf \
  specd:dflash2 python3 /Users/sagoyal/Downloads/code/muse-glimmer-dspark/scripts/convert_dflash2.py

# 1. extraction server (only if not already up; it needs NO layer change)
SEQ_LENGTH=49152 TP=4 GPU_MEM_UTIL=0.92 MAX_NUM_SEQS=32 BLOCK_SIZE=256 \
  HIDDEN_STATES_PATH=/dev/shm/hidden_states bash docker/serve_extract.sh

# 2. finish the render (GPUs not needed; render is CPU-only)
sudo docker run -d --name render48k --network host -e HF_HOME=/mnt/data/hf \
  -v /mnt/data:/mnt/data -v /dev/shm:/dev/shm specd:latest \
  bash /mnt/data/runs/render48k.sh

# 3. balanced merge (above)

# 4. train — ONLY after the warm-start decision is made
#    server on 0-3, trainer on 4-7; they cannot share a card.
```

## If the image is gone

`/mnt/data/images/specd-latest.tar` (`docker load -i`), or rebuild from
`docker/Dockerfile.train` then `docker/Dockerfile.tensorboard`.
