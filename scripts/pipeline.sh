#!/bin/bash
# Resumable driver for the whole training run, safe to re-enter after a
# preemption. Every stage is idempotent and guarded by a marker under
# $STATE_DIR, so restarting the unit picks up where the node died rather than
# redoing finished work.
#
#   stage 1  conversations.jsonl   from the HF dataset
#   stage 2  shards                round-robin split, bounds preprocessing loss
#   stage 3  prepared/<shard>      speculators prepare-data, one marker each
#   stage 4  data/                 merged Arrow dataset + token_freq.pt
#   stage 5  train                 resumes from its own checkpoint
#
# Stage 3 is the reason for sharding: prepare-data has no partial progress, so
# without shards a preemption at 90% costs the whole render.
set -uo pipefail

DATA_ROOT="${DATA_ROOT:-/mnt/data}"
REPO="${REPO:-$DATA_ROOT/src/muse-glimmer-dspark}"
RUNS="${RUNS:-$DATA_ROOT/runs}"
STATE_DIR="${STATE_DIR:-$RUNS/state}"
SHARDS="${SHARDS:-32}"
SEQ_LENGTH="${SEQ_LENGTH:-49152}"
MAX_SAMPLES="${MAX_SAMPLES:-50000}"
MIN_VALID_TOKENS="${MIN_VALID_TOKENS:-16}"
RENDER_ENDPOINT="${RENDER_ENDPOINT:-http://127.0.0.1:8000}"
PREP_WORKERS="${PREP_WORKERS:-8}"
MODEL="${MODEL:-meta-models/Muse-Glimmer-30B}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/hf}"

mkdir -p "$STATE_DIR" "$RUNS"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
done_marker() { [ -f "$STATE_DIR/$1.done" ]; }
mark() { touch "$STATE_DIR/$1.done"; }

wait_for_endpoint() {
  # The harness must not start rendering against a half-started engine; the
  # 49k-window server spends minutes in torch.compile before /health answers.
  local deadline=$((SECONDS + ${ENDPOINT_TIMEOUT:-3600}))
  until curl -sf "$RENDER_ENDPOINT/health" >/dev/null 2>&1; do
    if [ $SECONDS -ge $deadline ]; then
      log "endpoint $RENDER_ENDPOINT never became healthy"; return 1
    fi
    sleep 10
  done
  log "endpoint healthy"
}

# ---- stage 1: conversations -------------------------------------------------
if ! done_marker 01-conversations; then
  log "stage 1: building conversations.jsonl"
  "$REPO/.venv/bin/python" "$REPO/scripts/build_train_jsonl.py" \
    --out "$RUNS/conversations.jsonl" || exit 1
  mark 01-conversations
fi

# ---- stage 2: shards --------------------------------------------------------
if ! done_marker 02-shards; then
  log "stage 2: sharding into $SHARDS"
  "$REPO/.venv/bin/python" "$REPO/scripts/shard_jsonl.py" \
    --in "$RUNS/conversations.jsonl" --out-dir "$RUNS/shards" --shards "$SHARDS" || exit 1
  mark 02-shards
fi

# ---- stage 3: prepare-data, one shard at a time -----------------------------
if ! done_marker 03-prepared; then
  wait_for_endpoint || exit 1
  for shard in "$RUNS"/shards/shard-*.jsonl; do
    name=$(basename "$shard" .jsonl)
    done_marker "03-$name" && continue
    log "stage 3: prepare-data $name"
    # No --max-samples here: capping per shard would bias toward whichever
    # shards ran first. The cap is applied once, on the merged set.
    speculators prepare-data \
      --model "$MODEL" \
      --data "$shard" \
      --output "$RUNS/prepared/$name" \
      --seq-length "$SEQ_LENGTH" \
      --minimum-valid-tokens "$MIN_VALID_TOKENS" \
      --num-preprocessing-workers "$PREP_WORKERS" \
      --allow-empty-output \
      --render-endpoint "$RENDER_ENDPOINT"
    rc=$?
    if [ $rc -ne 0 ]; then
      log "prepare-data failed on $name (rc=$rc); will retry on next start"
      exit $rc
    fi
    mark "03-$name"
  done
  mark 03-prepared
fi

# ---- stage 4: merge ---------------------------------------------------------
if ! done_marker 04-merged; then
  log "stage 4: merging shards (cap $MAX_SAMPLES)"
  "$REPO/.venv/bin/python" "$REPO/scripts/merge_prepared.py" \
    --shard-dir "$RUNS/prepared" --out "$RUNS/data" --max-samples "$MAX_SAMPLES" || exit 1
  "$REPO/.venv/bin/python" "$REPO/scripts/verify_prepared.py" \
    --data "$RUNS/data" --model "$MODEL" --sample 400 || exit 1
  mark 04-merged
fi

# ---- stage 5: train ---------------------------------------------------------
# speculators resumes from the newest checkpoint unless told not to, so this
# stage needs no marker of its own -- re-entering it continues the run.
wait_for_endpoint || exit 1
log "stage 5: training"
exec torchrun --standalone --nproc-per-node "${NPROC:-1}" -m speculators.train \
  --config "$REPO/configs/train_dspark.yaml" \
  --data-path "$RUNS/data" \
  --save-path "$RUNS/checkpoints" \
  --vllm-endpoint "$RENDER_ENDPOINT/v1" \
  --total-seq-len "$SEQ_LENGTH" \
  --log-dir "$RUNS/logs"
