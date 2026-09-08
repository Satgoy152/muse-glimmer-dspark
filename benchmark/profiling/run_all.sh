#!/usr/bin/env bash
# Run every short/long x concurrency case against an already-serving profiled
# container, one after another, so the single Nsight report gains one capture
# burst per case in this order. Cases are separated by whole benchmark waves,
# far longer than the splitter's idle threshold.
set -uo pipefail

REPO="${REPO:-/mnt/data/src/muse-glimmer-profile}"
PROFILE_DIR="${PROFILE_DIR:-/mnt/data/profiles/dflash2-20260908}"
PYTHON="${PYTHON:-/mnt/data/src/muse-glimmer-dspark/.venv/bin/python}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8002}"
PROMPTS="${PROMPTS:-$PROFILE_DIR/prompts.json}"
RUNS="$PROFILE_DIR/runs"

# Concurrency 1 walks one prompt per round, so it needs a round per prompt to
# average over acceptance. Concurrency 10 uses all ten prompts every round.
CASES=("short 1 10" "short 10 3" "long 1 10" "long 10 3")

mkdir -p "$RUNS"
status=0
for entry in "${CASES[@]}"; do
  read -r case concurrency rounds <<<"$entry"
  name="$case-c$concurrency"
  echo "=== $name (rounds=$rounds) $(date -u +%H:%M:%S) ==="
  "$PYTHON" "$REPO/benchmark/profiling/run_workload.py" \
    --prompts "$PROMPTS" --case "$case" --concurrency "$concurrency" \
    --rounds "$rounds" --max-tokens 512 --profile \
    --base-url "$BASE_URL" --out "$RUNS/$name" 2>&1 | tail -20
  code=${PIPESTATUS[0]}
  echo "--- $name exit=$code $(date -u +%H:%M:%S)"
  [ "$code" -ne 0 ] && status=1
done
echo "ALL DONE status=$status $(date -u +%H:%M:%S)"
exit "$status"
