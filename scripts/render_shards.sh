#!/bin/bash
# Render every shard in $RUNS/shards at $SEQ_LENGTH, resumable and idempotent.
#
# Generalises scripts/render48k.sh, which hardcodes 49,152. Do NOT edit
# render48k.sh while a container is running it -- bash reads a script
# incrementally and an in-place edit corrupts the running job.
#
# Resume is decided by whether the output shard actually holds arrow files, not
# by directory existence: a killed prepare-data leaves an EMPTY output dir
# behind (prepared/shard-007 and shard-010 are both in that state), and a
# directory-existence check silently skips them forever.
set -uo pipefail
RUNS=${RUNS:-/mnt/data/runs}
SEQ_LENGTH=${SEQ_LENGTH:?set SEQ_LENGTH}
OUT=${OUT:?set OUT}
MODEL=${MODEL:-meta-models/Muse-Glimmer-30B}
ENDPOINT=${RENDER_ENDPOINT:-http://127.0.0.1:8000}
export HF_HOME=${HF_HOME:-/mnt/data/hf}
mkdir -p "$OUT"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

log "rendering to $OUT at seq-length $SEQ_LENGTH via $ENDPOINT"
for shard in "$RUNS"/shards/shard-*.jsonl; do
  name=$(basename "$shard" .jsonl)
  if compgen -G "$OUT/$name/*.arrow" > /dev/null; then
    log "skip $name (already has rows)"
    continue
  fi
  rm -rf "${OUT:?}/$name"
  log "prepare-data $name at $SEQ_LENGTH"
  # No --max-samples: capping per shard biases toward whichever ran first.
  # The row budget is applied once, on the merged set, stratified by level.
  speculators prepare-data \
    --model "$MODEL" \
    --data "$shard" \
    --output "$OUT/$name" \
    --seq-length "$SEQ_LENGTH" \
    --minimum-valid-tokens 16 \
    --num-preprocessing-workers 8 \
    --allow-empty-output \
    --render-endpoint "$ENDPOINT"
  rc=$?
  [ $rc -ne 0 ] && { log "FAILED $name rc=$rc"; exit $rc; }
  log "done $name"
done
log "ALL SHARDS RENDERED at $SEQ_LENGTH"
