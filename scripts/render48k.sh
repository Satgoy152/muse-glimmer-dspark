#!/bin/bash
# Re-render the same 32 shards at a 49,152 window for the DFlash2 run.
# Run A's prepared/ is at 32,768; turns whose prompt alone exceeded that were
# dropped at render time and are not recoverable from it. Fresh state dir so
# run A's stage-3 markers do not short-circuit this.
set -uo pipefail
RUNS=/mnt/data/runs
STATE=$RUNS/state-48k
OUT=$RUNS/prepared-48k
MODEL=meta-models/Muse-Glimmer-30B
export HF_HOME=/mnt/data/hf
mkdir -p "$STATE" "$OUT"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

for shard in "$RUNS"/shards/shard-*.jsonl; do
  name=$(basename "$shard" .jsonl)
  [ -f "$STATE/$name.done" ] && { log "skip $name"; continue; }
  log "prepare-data $name at 49152"
  # No --max-samples: capping per shard biases toward whichever ran first.
  # The row budget is applied once, on the merged set, stratified by level.
  speculators prepare-data \
    --model "$MODEL" \
    --data "$shard" \
    --output "$OUT/$name" \
    --seq-length 49152 \
    --minimum-valid-tokens 16 \
    --num-preprocessing-workers 8 \
    --allow-empty-output \
    --render-endpoint http://127.0.0.1:8000
  rc=$?
  [ $rc -ne 0 ] && { log "FAILED $name rc=$rc"; exit $rc; }
  touch "$STATE/$name.done"
  log "done $name"
done
log "ALL SHARDS RENDERED"
