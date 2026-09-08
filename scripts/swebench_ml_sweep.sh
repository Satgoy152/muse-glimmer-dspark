#!/bin/bash
# Task 2 part 2: replay the frozen SWE-bench Multilingual call set through every
# drafter, and additionally run three of them on-policy for resolved rate.
#
# One server per drafter. The three drafters that need an on-policy run get it
# against the same server that just did their replay, so they cost one model
# load, not two.
set -uo pipefail

R="${REPO:-/mnt/data/src/muse-glimmer-dspark}"
AV="${AV:-/mnt/data/agentvenv}"
SRC="${SRC:-/mnt/data/eval/swebench_ml_calls.jsonl}"
SUBSET="${SUBSET:-/mnt/data/bench/swebench_ml}"
OUTROOT="${OUTROOT:-/mnt/data/eval/mlsweep}"
PORT="${PORT:-8001}"
PROXY_PORT="${PROXY_PORT:-8082}"
CONC="${CONC:-10}"
WORKERS="${WORKERS:-8}"
ONPOLICY="${ONPOLICY:-1}"
STATE="$OUTROOT/driver.log"
mkdir -p "$OUTROOT"
say() { echo "[$(date -Is)] $*" | tee -a "$STATE"; }

if [ ! -s "$SRC" ]; then say "FATAL: $SRC is missing or empty"; exit 1; fi
NCALLS=$(wc -l < "$SRC")
say "replaying $NCALLS frozen SWE-bench Multilingual calls per drafter"

onpolicy() {   # label
  local LABEL="$1" OUT="/mnt/data/runs/mlonpolicy/$LABEL"
  mkdir -p "$OUT"
  if [ -s "$OUT/preds.json" ] && \
     [ "$($AV/bin/python -c "import json,sys;print(len(json.load(open('$OUT/preds.json'))))" 2>/dev/null || echo 0)" -ge \
       "$(wc -l < "$SUBSET/test.jsonl")" ]; then
    say "on-policy $LABEL already complete"; return 0
  fi
  say "on-policy $LABEL -> $OUT"
  UPSTREAM_URL="http://127.0.0.1:$PORT/v1/chat/completions" \
    TRACE_DIR="/mnt/data/traces/mlonpolicy-$LABEL" PORT="$PROXY_PORT" \
    nohup "$AV/bin/python" "$R/scripts/proxy.py" \
    > "/mnt/data/logs/mlonpolicy-$LABEL.proxy.log" 2>&1 &
  local PP=$!
  sleep 5
  cd "$R"; export MSWEA_CONFIGURED=true
  local N; N=$(wc -l < "$SUBSET/test.jsonl"); local Q=$(( (N + 3) / 4 )); local i=0
  for eff in low medium high xhigh; do
    local lo=$i hi=$(( i + Q )); [ "$hi" -gt "$N" ] && hi=$N; i=$hi
    [ "$lo" -ge "$hi" ] && continue
    "$AV/bin/mini-extra" swebench --subset "$SUBSET" --split test --slice "$lo:$hi" \
      -c swebench.yaml -c "$R/configs/swegym.yaml" -c "$R/configs/effort_$eff.yaml" \
      -c "model.model_kwargs.api_base=http://127.0.0.1:$PROXY_PORT/v1" \
      -w "$WORKERS" -o "$OUT" >> "/mnt/data/logs/mlonpolicy-$LABEL.log" 2>&1 \
      || say "on-policy $LABEL slice $lo:$hi returned $?"
  done
  kill "$PP" 2>/dev/null
  say "on-policy $LABEL done"
}

run_label() {
  local LABEL="$1" SPEC="$2" METHOD="$3" OP="${4:-0}"
  local CELL="${LABEL}__mlfull__c${CONC}__r1"
  local done_replay=0 done_op=0
  [ -s "$OUTROOT/$LABEL/$CELL.json" ] && done_replay=1
  [ "$OP" = 0 ] && done_op=1
  if [ "$done_replay" = 1 ] && [ "$done_op" = 1 ]; then
    say "### $LABEL already complete"; return 0; fi

  say "### START $LABEL"
  LABEL="$LABEL" SPEC="$SPEC" METHOD="$METHOD" PORT="$PORT" REPO="$R" OUTROOT="$OUTROOT" \
    WARMUP_SRC="$SRC" bash "$R/scripts/sweep_serve.sh" 2>&1 | tee -a "$STATE"
  if [ "${PIPESTATUS[0]}" != 0 ]; then
    say "### $LABEL SERVER FAILED -- skipping"
    LABEL="$LABEL" PORT="$PORT" bash "$R/scripts/sweep_down.sh" >>"$STATE" 2>&1; return 1; fi

  CELL="$CELL" SRC="$SRC" CONC="$CONC" LABEL="$LABEL" PORT="$PORT" REPO="$R" \
    OUTROOT="$OUTROOT" bash "$R/scripts/sweep_cell.sh" 2>&1 | tee -a "$STATE"

  [ "$OP" = 1 ] && [ "$ONPOLICY" = 1 ] && onpolicy "$LABEL"

  LABEL="$LABEL" PORT="$PORT" bash "$R/scripts/sweep_down.sh" 2>&1 | tee -a "$STATE"
  say "### DONE $LABEL"
}

S=/mnt/data/speculators
# The three that also need an on-policy pass go first: if the night runs out,
# the resolved-rate numbers are the ones that cannot be recovered from a replay.
run_label dflash2             "$S/dflash2"                        dflash 1
run_label dflash2-run-d-mid   "$S/dflash2-run-d-32k-mix-step1976" dflash 1
run_label dspark-run-a-32k    "$S/dspark-run-a-32k"               dspark 1
run_label nospec              none                                none   0
run_label dflash2-run-d-final "$S/dflash2-run-d-32k-mix"          dflash 0
run_label dflash-official     "$S/dflash-official"                dflash 0
run_label dspark-community    "$S/dspark-community"               dspark 0
run_label dspark-run-b-49k    "$S/dspark-run-b-49k"               dspark 0
say "########## SWEBENCH-ML SWEEP FINISHED ##########"
