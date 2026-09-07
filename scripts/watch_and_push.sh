#!/bin/bash
# Watch the DFlash2 run: snapshot every checkpoint, then push to HF on completion.
#
# WHY THE SNAPSHOTTING: speculators writes every checkpoint into the SAME slot
# (`<save_path>/0`) and re-points a `epoch0_step<N>` symlink at it, unlinking the
# previous one. So `checkpoint_freq: 0.125` produces eight OVERWRITES, not eight
# checkpoints -- at any moment exactly one exists. Selecting a checkpoint by
# replay eval is impossible unless something copies each one aside as it lands.
# The symlink is created *after* the write completes, so its appearance is the
# signal that the slot is safe to copy.
set -uo pipefail
CKPT_DIR=/mnt/data/runs/checkpoints-dflash2
KEEP=/mnt/data/runs
CONTAINER=dflash2-train
REPO="${HF_REPO:-Satgoy152/Muse-Glimmer-30B-DFlash2-Coding}"
PATH_IN_REPO="${HF_PATH_IN_REPO:-run-d-32k-mix}"
LOG=/mnt/data/runs/logs/watch_and_push.log
mkdir -p "$(dirname "$LOG")"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

snapshot() {
  for link in "$CKPT_DIR"/epoch0_step* "$CKPT_DIR"/epoch0_end; do
    [ -L "$link" ] || continue
    name=$(basename "$link")
    dest="$KEEP/keep-dflash2-${name#epoch0_}"
    [ -d "$dest" ] && continue
    log "snapshotting $name -> $(basename "$dest")"
    cp -a --dereference "$link" "$dest.partial" && mv "$dest.partial" "$dest" \
      && log "snapshot ok: $(du -sh "$dest" | cut -f1)" \
      || { log "SNAPSHOT FAILED for $name"; rm -rf "$dest.partial"; }
  done
}

log "watching $CONTAINER; repo=$REPO path=$PATH_IN_REPO"
while sudo docker ps --filter "name=^${CONTAINER}$" --format '{{.Names}}' | grep -q "$CONTAINER"; do
  snapshot
  sleep 45
done

rc=$(sudo docker inspect "$CONTAINER" --format '{{.State.ExitCode}}' 2>/dev/null || echo "?")
log "trainer exited rc=$rc"
snapshot   # catch the epoch-end checkpoint

if [ "$rc" != "0" ]; then
  log "NOT PUSHING: trainer exited non-zero. Inspect before uploading."
  exit 1
fi

# Push the final checkpoint first -- that is the one that was asked for -- then
# the preserved mid-run snapshot, so the volume can be torn down safely.
push() {  # $1 = local dir, $2 = subfolder in repo
  [ -d "$1" ] || { log "skip push, missing $1"; return 0; }
  log "pushing $1 -> $REPO/$2"
  sudo docker run --rm --network host -v /mnt/data:/mnt/data \
    --env-file /mnt/data/runs/secrets.env -e HF_HOME=/mnt/data/hf \
    specd:dflash2 python3 /mnt/data/src/muse-glimmer-dspark/scripts/push_checkpoint.py \
      --checkpoint "$1" --repo "$REPO" --path-in-repo "$2" \
      --extra "$CKPT_DIR/run.yaml" "$CKPT_DIR/train_command.txt" >>"$LOG" 2>&1 \
    && log "PUSH OK: $2" || log "PUSH FAILED: $2"
}

final=$(ls -d "$KEEP"/keep-dflash2-end 2>/dev/null || ls -dt "$KEEP"/keep-dflash2-step* 2>/dev/null | head -1)
push "$final" "$PATH_IN_REPO"
push "$KEEP/keep-dflash2-step1976" "$PATH_IN_REPO-step1976"

log "ALL DONE -- safe to stop the GPU"
