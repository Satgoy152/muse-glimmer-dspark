#!/bin/bash
# Measure speculative acceptance on a coding benchmark (HumanEval/MBPP) for one
# drafter, end to end: serve -> generate -> metrics. One drafter per
# invocation, same as eval_replay.sh.
#
#   NAME=he-dflash2 SPEC=/mnt/data/speculators/dflash2 METHOD=dflash \
#   bash scripts/eval_humaneval.sh
#
# This is NOT the Terminal-Bench regime. eval_replay.sh runs temperature 1.0 /
# top_k 64 at concurrency 10 and lands ~3.8-4.0; the published table this
# compares against is greedy at concurrency 1. Acceptance is not
# contention-independent -- we measured it rising 0.04-0.07 from concurrency 10
# to 64 -- so TEMP and CONC are pinned here and recorded in every output row.
# Do not put numbers from the two scripts in one table.
#
# The guards below are eval_replay.sh's, for the same reasons it lists (a
# leftover server on the port yields a clean, plausible, mislabelled result;
# `docker logs | grep -q` under pipefail misreports a present line as absent).
# Two are new:
#   * NAME collision. Reusing a NAME overwrites a previous drafter's trace and
#     eval JSON in place, and the overwrite is invisible afterwards.
#   * The V2 runner check is fatal, not a warning. A DFlash2 checkpoint on the
#     V1 runner degrades to DFlash1 silently and still reports a believable
#     acceptance number, which is the exact failure this whole script exists to
#     make impossible.
set -euo pipefail

NAME="${NAME:?set NAME}"
SPEC="${SPEC:?set SPEC to a local drafter dir}"
METHOD="${METHOD:-dspark}"          # vLLM accepts dflash|dspark only; DFlash2
                                    # checkpoints use dflash and are routed to
                                    # the V2 runner by their architectures field
PORT="${PORT:-8001}"
CONC="${CONC:-1}"
TEMP="${TEMP:-0}"
MAXTOK="${MAXTOK:-2048}"
TASK="${TASK:-HumanEval}"
REPO="${REPO:-/mnt/data/src/muse-glimmer-dspark}"
SRC="${SRC:-/mnt/data/bench/speculator_benchmarks/HumanEval.jsonl}"
PASS1="${PASS1:-1}"
LIMIT="${LIMIT:-0}"   # >0 = smoke test; never use for a reported number
RESUME="${RESUME:-0}"
TRACE_DIR="/mnt/data/traces/he-$NAME"
CTR="vllm-he-$NAME"
PY="$REPO/.venv/bin/python"
EVAL_JSON="/mnt/data/eval/he-$NAME.json"

mkdir -p "$TRACE_DIR" /mnt/data/eval /mnt/data/logs
echo "=== [$NAME] task=$TASK method=$METHOD spec=$SPEC temp=$TEMP conc=$CONC ==="

[ -d "$SPEC" ] || { echo "[$NAME] ABORT: no such drafter dir $SPEC" >&2; exit 1; }
[ -f "$SRC" ]  || { echo "[$NAME] ABORT: no such prompt file $SRC" >&2; exit 1; }

# A reused NAME silently overwrites the previous drafter's results.
if [ "$RESUME" != "1" ] && { [ -s "$EVAL_JSON" ] || [ -s "$TRACE_DIR/gen.jsonl" ]; }; then
  echo "[$NAME] ABORT: results already exist for this NAME:" >&2
  ls -la "$EVAL_JSON" "$TRACE_DIR/gen.jsonl" 2>/dev/null >&2 || true
  echo "  pick a new NAME, or set RESUME=1 to continue this one" >&2
  exit 1
fi

cleanup() { sudo docker rm -f "$CTR" >/dev/null 2>&1 || true; }
trap cleanup EXIT

sudo docker rm -f "$CTR" >/dev/null 2>&1 || true
# Nothing else may own the port, or we would silently measure that instead.
if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "[$NAME] ABORT: something is already serving :$PORT" >&2
  sudo docker ps --format '  {{.Names}} {{.Status}}' >&2
  exit 1
fi

sudo docker run -d --name "$CTR" --gpus '"device=0"' --network host --ipc host \
  -v /mnt/data:/mnt/data -e HF_HOME=/mnt/data/hf \
  -e TARGET_MODEL=meta-models/Muse-Glimmer-30B \
  -e SPEC_METHOD="$METHOD" -e SPECULATOR="$SPEC" \
  -e NUM_SPEC_TOKENS=15 -e PORT="$PORT" \
  specd:latest bash "$REPO/docker/serve_patched.sh" >/dev/null

echo "[$NAME] waiting for health on :$PORT"
for i in $(seq 1 180); do
  # container first: a dead container that never bound the port must not be
  # mistaken for a healthy one just because someone else answers
  if ! sudo docker ps --format '{{.Names}}' | grep -qx "$CTR"; then
    echo "[$NAME] SERVER FAILED TO START; last 40 lines:" >&2
    sudo docker logs "$CTR" 2>&1 | tail -40 >&2
    exit 1
  fi
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 10
done
sudo docker ps --format '{{.Names}}' | grep -qx "$CTR" || { echo "[$NAME] container gone" >&2; exit 1; }
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "[$NAME] TIMED OUT" >&2; exit 1; }

LOGS="$(sudo docker logs "$CTR" 2>&1 || true)"

# The dspark patch is mandatory for a dspark drafter.
if [ "$METHOD" = "dspark" ]; then
  case "$LOGS" in
    *"dspark patch OK"*) echo "[$NAME] dspark patch OK" ;;
    *) echo "[$NAME] dspark patch NOT confirmed -- refusing to measure" >&2; exit 1 ;;
  esac
fi
# A DFlash2 checkpoint on the V1 runner degrades to DFlash1 without saying so.
if grep -q DFlash2DraftModel "$SPEC/config.json" 2>/dev/null; then
  case "$LOGS" in
    *"V2 Model Runner"*|*"V2 model runner"*)
      echo "[$NAME] V2 runner confirmed:"
      printf '%s\n' "$LOGS" | grep -iE "v2 model runner" | head -3 ;;
    *)
      echo "[$NAME] ABORT: DFlash2 checkpoint but no V2 runner line -- would" >&2
      echo "  silently measure DFlash1. Runner/dflash lines in the log:" >&2
      printf '%s\n' "$LOGS" | grep -iE "model runner|dflash" | head -10 >&2
      exit 1 ;;
  esac
fi

# Counter baseline: the cross-check uses the delta across the run, so warmup
# and health traffic cannot leak into it.
curl -s "http://127.0.0.1:$PORT/metrics" -o "/mnt/data/eval/he-$NAME.before.prom"

echo "[$NAME] generating"
$PY "$REPO/benchmark/humaneval/gen.py" "$SRC" \
  --url "http://127.0.0.1:$PORT/v1/chat/completions" \
  --out "$TRACE_DIR/gen.jsonl" --concurrency "$CONC" \
  --temperature "$TEMP" --max-tokens "$MAXTOK" --stream \
  ${LIMIT:+--limit "$LIMIT"}

# still ours? if the container died mid-run the trace is not this drafter's
sudo docker ps --format '{{.Names}}' | grep -qx "$CTR" \
  || { echo "[$NAME] container died during the run -- trace is suspect" >&2; exit 1; }

curl -s "http://127.0.0.1:$PORT/metrics" -o "/mnt/data/eval/he-$NAME.prom"

echo "[$NAME] metrics"
P1=()
[ "$PASS1" = "1" ] && P1=(--pass1)
$PY "$REPO/benchmark/humaneval/metrics.py" "$TRACE_DIR/gen.jsonl" \
  --name "$NAME" --task "$TASK" --json-out "$EVAL_JSON" \
  --prom-before "/mnt/data/eval/he-$NAME.before.prom" \
  --prom-after "/mnt/data/eval/he-$NAME.prom" "${P1[@]}"

echo "=== [$NAME] COMPLETE ==="
