#!/bin/bash
# Record target-only SWE-bench Multilingual trajectories through scripts/proxy.py,
# then convert them into replay.py's input schema.
#
# Three fixes over scripts/swegym_holdout.sh, which this is modelled on:
#
#  1. The health probe goes to the UPSTREAM port, not through the proxy. The old
#     script curl'd `{"messages":[{"role":"user","content":"hi"}]}` at the proxy,
#     so the very first row of every recording was a 4-token warmup with no
#     tools and no reasoning_strength -- and that row is what ended up in the
#     converted file.
#  2. The converter runs on the trace that exists when the agent runs finish,
#     and drops any row that is not a real agent call (no tools, no
#     reasoning_strength, or an errored response).
#  3. VALIDATE_FIRST=1 records exactly one instance and refuses to continue
#     unless that instance produced replayable calls. A 32-task recording that
#     silently produces nothing costs a night.
#
# Target-only (SPEC_METHOD=none) on purpose: the recording must not be
# conditioned on any drafter that is later measured against it.
set -uo pipefail

R="${REPO:-/mnt/data/src/muse-glimmer-dspark}"
AV="${AV:-/mnt/data/agentvenv}"
SUBSET="${SUBSET:-/mnt/data/bench/swebench_ml}"
SPLIT="${SPLIT:-test}"
TD="${TD:-/mnt/data/traces/swebench-ml-record}"
RUNS="${RUNS:-/mnt/data/runs/swebench-ml}"
OUT="${OUT:-/mnt/data/eval/swebench_ml_calls.jsonl}"
CTR="${CTR:-vllm-mlrec}"
PORT="${PORT:-8051}"
PROXY_PORT="${PROXY_PORT:-8082}"
WORKERS="${WORKERS:-8}"
SPEC_METHOD="${SPEC_METHOD:-none}"
SPECULATOR="${SPECULATOR:-none}"
VALIDATE_FIRST="${VALIDATE_FIRST:-1}"
N=$(wc -l < "$SUBSET/$SPLIT.jsonl")

mkdir -p "$TD" "$RUNS" /mnt/data/logs
echo "=== swebench-ml record: $N instances, spec=$SPEC_METHOD, $WORKERS workers ==="

cleanup() {
  # Never pattern-kill on something that also matches an ssh command line.
  [ -n "${PROXY_PID:-}" ] && kill "$PROXY_PID" 2>/dev/null
  sudo docker rm -f "$CTR" >/dev/null 2>&1
}
trap cleanup EXIT

sudo docker rm -f "$CTR" >/dev/null 2>&1
if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "ABORT: something already serving :$PORT" >&2; exit 1
fi
sudo docker run -d --name "$CTR" --gpus '"device=0"' --network host --ipc host \
  -v /mnt/data:/mnt/data -e HF_HOME=/mnt/data/hf \
  -e TARGET_MODEL=meta-models/Muse-Glimmer-30B \
  -e SPEC_METHOD="$SPEC_METHOD" -e SPECULATOR="$SPECULATOR" \
  -e NUM_SPEC_TOKENS=15 -e PORT="$PORT" specd:latest \
  bash "$R/docker/serve_patched.sh" >/dev/null

for i in $(seq 1 240); do
  sudo docker ps --format '{{.Names}}' | grep -qx "$CTR" \
    || { echo "SERVER DIED"; sudo docker logs "$CTR" 2>&1 | tail -30; exit 1; }
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo TIMEOUT; exit 1; }
LOGS="$(sudo docker logs "$CTR" 2>&1 || true)"
if [ "$SPEC_METHOD" = "none" ]; then
  case "$LOGS" in *"no-spec control: target only"*) echo "target-only confirmed";;
    *) echo "FATAL: target-only line missing" >&2; exit 1;; esac
fi

# Warm the engine WITHOUT going through the proxy, so nothing that is not an
# agent call ever reaches the trace.
curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" -X POST \
  -H 'content-type: application/json' \
  -d '{"model":"meta-models/Muse-Glimmer-30B","messages":[{"role":"user","content":"hi"}],"max_tokens":4}' \
  -o /dev/null && echo "upstream warm"

UPSTREAM_URL="http://127.0.0.1:$PORT/v1/chat/completions" TRACE_DIR="$TD" PORT="$PROXY_PORT" \
  nohup "$AV/bin/python" "$R/scripts/proxy.py" > /mnt/data/logs/mlrec_proxy.log 2>&1 &
PROXY_PID=$!
sleep 5
grep -q "proxy on" /mnt/data/logs/mlrec_proxy.log || { echo "proxy failed to start" >&2; exit 1; }

cd "$R"
export MSWEA_CONFIGURED=true

# Order the subset so each reasoning strength is one contiguous quarter --
# mini-swe-agent takes the effort from --config, not from the row, so the file
# order has to agree with the slices below or every row is mislabelled.
"$AV/bin/python" - "$SUBSET/$SPLIT.jsonl" <<'PYEOF'
import json, sys
p = sys.argv[1]
rows = [json.loads(l) for l in open(p)]
order = {"low": 0, "medium": 1, "high": 2, "xhigh": 3}
rows.sort(key=lambda r: (order[r["reasoning_strength"]], r["instance_id"]))
open(p, "w").write("".join(json.dumps(r) + "\n" for r in rows))
print("subset ordered by strength:", [r["reasoning_strength"] for r in rows])
PYEOF

run_slice() {   # lo hi effort
  "$AV/bin/mini-extra" swebench \
    --subset "$SUBSET" --split "$SPLIT" --slice "$1:$2" \
    -c swebench.yaml -c "$R/configs/swegym.yaml" -c "$R/configs/effort_$3.yaml" \
    -c "model.model_kwargs.api_base=http://127.0.0.1:$PROXY_PORT/v1" \
    -w "$WORKERS" -o "$RUNS" || echo "slice $1:$2 returned $?"
}

before=$(wc -l < "$TD/calls.jsonl" 2>/dev/null || echo 0)
if [ "$VALIDATE_FIRST" = 1 ]; then
  echo "--- validating recording on instance 0 ---"
  run_slice 0 1 low
  after=$(wc -l < "$TD/calls.jsonl" 2>/dev/null || echo 0)
  got=$("$AV/bin/python" "$R/scripts/convert_recorded_calls.py" --src "$TD/calls.jsonl" \
        --out /tmp/mlrec_validate.jsonl --quiet)
  echo "validation: trace rows $before -> $after, replayable $got"
  if [ "${got:-0}" -lt 5 ]; then
    echo "FATAL: instance 0 produced $got replayable calls -- not launching the rest" >&2
    exit 1
  fi
fi

# The subset file is ordered so each reasoning strength is one contiguous
# quarter (build_swebench_ml.py assigns strengths round-robin over the sorted
# instance list, then we sort by strength here).
Q=$(( (N + 3) / 4 ))
i=0
for eff in low medium high xhigh; do
  lo=$i; hi=$(( i + Q )); [ "$hi" -gt "$N" ] && hi=$N; i=$hi
  [ "$lo" -ge "$hi" ] && continue
  echo "--- slice $lo:$hi effort=$eff ---"
  run_slice "$lo" "$hi" "$eff"
done

kill "$PROXY_PID" 2>/dev/null; PROXY_PID=""
sleep 2
"$AV/bin/python" "$R/scripts/convert_recorded_calls.py" --src "$TD/calls.jsonl" --out "$OUT"
echo "=== SWEBENCH-ML RECORDING COMPLETE ==="
