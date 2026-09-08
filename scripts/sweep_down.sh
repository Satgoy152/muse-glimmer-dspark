#!/bin/bash
# Tear down a server left up by sweep_serve.sh.
set -uo pipefail
LABEL="${LABEL:?set LABEL}"
sudo docker rm -f "vllm-sweep-$LABEL" >/dev/null 2>&1 || true
for i in $(seq 1 30); do
  curl -sf "http://127.0.0.1:${PORT:-8001}/v1/models" >/dev/null 2>&1 || break
  sleep 2
done
echo "=== serve [$LABEL] DOWN ==="
