#!/bin/bash
# Concurrency-1 replay of a fixed 200-call Terminal-Bench subset.
# Purpose: measure per-request wall-clock without queueing contention, and
# include a no-spec (target-only) control so every drafter gets a speedup
# multiple rather than only an acceptance length.
set -euo pipefail
NAME="${NAME:?}" ; SPEC="${SPEC:-meta-models/Muse-Glimmer-30B-assistant}" ; METHOD="${METHOD:?}"
PORT="${PORT:-8011}" ; R=/mnt/data/src/muse-glimmer-dspark ; PY=$R/.venv/bin/python
CTR="vllm-c1-$NAME" ; OUT=/mnt/data/traces/c1-$NAME
mkdir -p "$OUT"
cleanup(){ sudo docker rm -f "$CTR" >/dev/null 2>&1 || true; }
trap cleanup EXIT
sudo docker rm -f "$CTR" >/dev/null 2>&1 || true
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "port busy"; exit 1; }
sudo docker run -d --name "$CTR" --gpus '"device=0"' --network host --ipc host \
  -v /mnt/data:/mnt/data -e HF_HOME=/mnt/data/hf \
  -e TARGET_MODEL=meta-models/Muse-Glimmer-30B -e SPEC_METHOD="$METHOD" -e SPECULATOR="$SPEC" \
  -e NUM_SPEC_TOKENS=15 -e PORT="$PORT" specd:latest bash "$R/docker/serve_patched.sh" >/dev/null
for i in $(seq 1 200); do
  sudo docker ps --format '{{.Names}}' | grep -qx "$CTR" || { echo "SERVER DIED"; sudo docker logs "$CTR" 2>&1|tail -30; exit 1; }
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 10
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo TIMEOUT; exit 1; }
L="$(sudo docker logs "$CTR" 2>&1)"
if [ "$METHOD" = dspark ]; then case "$L" in *"dspark patch OK"*) echo "dspark patch OK";; *) echo "NO DSPARK PATCH"; exit 1;; esac; fi
case "$L" in *"no-spec control"*) echo "NO-SPEC CONTROL confirmed";; esac
echo "[$NAME] replaying c1"; date
rm -f "$OUT/calls.jsonl"
$PY "$R/benchmark/terminal_bench/replay.py" /mnt/data/eval/sub100.jsonl \
  --url "http://127.0.0.1:$PORT/v1/chat/completions" --out "$OUT/calls.jsonl" --concurrency 1 --stream
sudo docker ps --format '{{.Names}}' | grep -qx "$CTR" || { echo "died mid-run"; exit 1; }
curl -s "http://127.0.0.1:$PORT/metrics" -o "/mnt/data/eval/c1-$NAME.prom"
date; echo "=== [$NAME] DONE $(wc -l < "$OUT/calls.jsonl") calls ==="
