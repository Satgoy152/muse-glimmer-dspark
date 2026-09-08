#!/bin/bash
# Run ONE cell of the sweep against an already-running server.
#
#   CELL=dflash2__b64_128__c8__r1 SRC=/mnt/data/eval/manifests/b64_128.jsonl \
#   CONC=8 LABEL=dflash2 bash scripts/sweep_cell.sh
#
# Takes a /metrics snapshot immediately BEFORE the replay and one after, so
# every server-side number is a delta. eval_replay.sh saved only the after
# snapshot, which meant warmup traffic and every earlier cell on the same server
# were folded into the counters -- fatal here, where 16 cells share one server.
set -euo pipefail

CELL="${CELL:?set CELL}"
SRC="${SRC:?set SRC}"
LABEL="${LABEL:?set LABEL}"
CONC="${CONC:-1}"
PORT="${PORT:-8001}"
REPO="${REPO:-/mnt/data/src/muse-glimmer-dspark}"
OUTROOT="${OUTROOT:-/mnt/data/eval/sweep}"
CTR="vllm-sweep-$LABEL"
PY="$REPO/.venv/bin/python"
OUT="$OUTROOT/$LABEL"
TRACE_DIR="/mnt/data/traces/sweep/$CELL"

mkdir -p "$OUT" "$TRACE_DIR"

# The replay resumes an existing output path, so a reused CELL name would
# silently skip every call and hand back the previous run's numbers.
if [ -s "$OUT/$CELL.json" ]; then
  echo "[$CELL] already complete, skipping"
  exit 0
fi

sudo docker ps --format '{{.Names}}' | grep -qx "$CTR" \
  || { echo "[$CELL] FATAL: server $CTR is not running" >&2; exit 1; }
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null \
  || { echo "[$CELL] FATAL: :$PORT not healthy" >&2; exit 1; }

T0=$(date +%s)
curl -s "http://127.0.0.1:$PORT/metrics" -o "$OUT/$CELL.before.prom"

echo "[$CELL] replay conc=$CONC src=$(basename "$SRC")"
$PY "$REPO/benchmark/terminal_bench/replay.py" "$SRC" \
  --url "http://127.0.0.1:$PORT/v1/chat/completions" \
  --out "$TRACE_DIR/calls.jsonl" --concurrency "$CONC" --stream

curl -s "http://127.0.0.1:$PORT/metrics" -o "$OUT/$CELL.prom"
T1=$(date +%s)

sudo docker ps --format '{{.Names}}' | grep -qx "$CTR" \
  || { echo "[$CELL] FATAL: server died during the cell -- trace is suspect" >&2; exit 1; }

$PY "$REPO/benchmark/terminal_bench/filter_canary.py" \
  "$TRACE_DIR/calls.jsonl" "$TRACE_DIR/calls.clean.jsonl" >/dev/null

$PY "$REPO/benchmark/terminal_bench/replay_metrics.py" "$TRACE_DIR/calls.clean.jsonl" \
  --name "$CELL" --json-out "$OUT/$CELL.json" | tail -8

$PY - "$OUT/$CELL.json" "$CELL" "$LABEL" "$CONC" "$SRC" "$((T1-T0))" <<'PYEOF'
import json, sys
p, cell, label, conc, src, wall = sys.argv[1:7]
d = json.load(open(p))
d["_meta"].update({"cell": cell, "label": label, "concurrency": int(conc),
                   "src": src, "wall_s": int(wall),
                   "bucket": cell.split("__")[1] if "__" in cell else "",
                   "repeat": cell.split("__")[-1]})
json.dump(d, open(p, "w"), indent=1)
PYEOF

echo "=== [$CELL] COMPLETE in $((T1-T0))s ==="
