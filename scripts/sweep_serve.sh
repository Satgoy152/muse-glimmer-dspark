#!/bin/bash
# Bring ONE speculator server up and leave it running.
#
# `scripts/eval_replay.sh` starts a server, runs one replay and tears it down.
# The bucket x concurrency sweep runs 16+ cells against the same drafter, and a
# 7-minute load per cell would turn it into 128 model loads. So: serve once
# here, run cells with scripts/sweep_cell.sh, tear down with sweep_down.sh.
#
#   LABEL=dflash2 SPEC=/mnt/data/speculators/dflash2 METHOD=dflash \
#   bash scripts/sweep_serve.sh
#
# Every guard eval_replay.sh has is kept, and the one it gets wrong is fixed:
# a missing "V2 Model Runner" line for a DFlash2 checkpoint is FATAL here, not a
# warning. A DFlash2 checkpoint served on the V1 runner silently degrades to
# DFlash1 and produces a clean, plausible, wrong number.
set -euo pipefail

LABEL="${LABEL:?set LABEL}"
SPEC="${SPEC:-none}"
METHOD="${METHOD:-dflash}"
PORT="${PORT:-8001}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-15}"
REPO="${REPO:-/mnt/data/src/muse-glimmer-dspark}"
OUTROOT="${OUTROOT:-/mnt/data/eval/sweep}"
CTR="vllm-sweep-$LABEL"
PY="$REPO/.venv/bin/python"

mkdir -p "$OUTROOT/$LABEL" /mnt/data/logs
echo "=== serve [$LABEL] method=$METHOD spec=$SPEC port=$PORT k=$NUM_SPEC_TOKENS ==="

sudo docker rm -f "$CTR" >/dev/null 2>&1 || true
if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "[$LABEL] ABORT: something is already serving :$PORT" >&2
  sudo docker ps --format '  {{.Names}} {{.Status}}' >&2
  exit 1
fi

sudo docker run -d --name "$CTR" --gpus '"device=0"' --network host --ipc host \
  -v /mnt/data:/mnt/data -e HF_HOME=/mnt/data/hf \
  -e TARGET_MODEL=meta-models/Muse-Glimmer-30B \
  -e SPEC_METHOD="$METHOD" -e SPECULATOR="$SPEC" \
  -e NUM_SPEC_TOKENS="$NUM_SPEC_TOKENS" -e PORT="$PORT" -e ADAPTIVE="${ADAPTIVE:-0}" \
  specd:latest bash "$REPO/docker/serve_patched.sh" >/dev/null

echo "[$LABEL] waiting for health on :$PORT"
for i in $(seq 1 240); do
  if ! sudo docker ps --format '{{.Names}}' | grep -qx "$CTR"; then
    echo "[$LABEL] SERVER FAILED TO START; last 40 lines:" >&2
    sudo docker logs "$CTR" 2>&1 | tail -40 >&2
    exit 1
  fi
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 10
done
sudo docker ps --format '{{.Names}}' | grep -qx "$CTR" || { echo "[$LABEL] container gone" >&2; exit 1; }
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "[$LABEL] TIMED OUT" >&2; exit 1; }

LOGS="$(sudo docker logs "$CTR" 2>&1 || true)"

if [ "$METHOD" = "dspark" ]; then
  case "$LOGS" in
    *"dspark patch OK"*) echo "[$LABEL] dspark patch OK" ;;
    *) echo "[$LABEL] dspark patch NOT confirmed -- refusing to measure" >&2; exit 1 ;;
  esac
fi
# FATAL, unlike eval_replay.sh: a DFlash2 checkpoint on the V1 runner is a
# silent downgrade to DFlash1.
if [ "$SPEC" != "none" ] && grep -q DFlash2DraftModel "$SPEC/config.json" 2>/dev/null; then
  case "$LOGS" in
    *"V2 Model Runner"*|*"V2 model runner"*|*"DFlash2 V2"*)
      echo "[$LABEL] V2 runner confirmed:"
      printf '%s\n' "$LOGS" | grep -iE "v2 model runner|dflash2" | head -5 ;;
    *)
      echo "[$LABEL] FATAL: no 'V2 Model Runner' line; DFlash2 would degrade to DFlash1" >&2
      printf '%s\n' "$LOGS" | grep -iE "model runner|dflash" | head -20 >&2
      exit 1 ;;
  esac
fi
if [ "$METHOD" = "none" ]; then
  case "$LOGS" in
    *"no-spec control: target only"*) echo "[$LABEL] no-spec control confirmed" ;;
    *) echo "[$LABEL] FATAL: no-spec control line missing" >&2; exit 1 ;;
  esac
fi

# Warm up: the first requests after load pay compilation and allocator costs
# that belong to no cell. Cells take their own before-snapshot, so this only has
# to move the engine past cold start.
echo "[$LABEL] warmup"
$PY "$REPO/benchmark/terminal_bench/replay.py" \
  "${WARMUP_SRC:-/mnt/data/eval/manifests/b64_128.jsonl}" \
  --url "http://127.0.0.1:$PORT/v1/chat/completions" \
  --out "/mnt/data/traces/warmup-$LABEL/calls.jsonl" \
  --concurrency 4 --limit "${WARMUP_N:-8}" --stream >/dev/null 2>&1 || true

printf '%s\n' "$LOGS" | grep -iE "v2 model runner|dspark patch|dflash2 knob|no-spec control" \
  > "$OUTROOT/$LABEL/server.info" || true
{ echo "label=$LABEL"; echo "method=$METHOD"; echo "spec=$SPEC"; echo "port=$PORT";
  echo "num_spec_tokens=$NUM_SPEC_TOKENS"; echo "container=$CTR";
  echo "started=$(date -Is)"; } >> "$OUTROOT/$LABEL/server.info"

echo "=== serve [$LABEL] READY ==="
