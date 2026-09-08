#!/bin/bash
# Same-drafter repeat of the SWE-bench Multilingual replay.
#
# run-D midpoint beats dflash2 by +0.300 [+0.189, +0.418] step-weighted on this
# benchmark. The Terminal-Bench repeat control at concurrency 10 is +0.010
# [-0.101, +0.120], which suggests +0.300 is real -- but that control was
# measured on a different call set, and borrowing a noise floor across workloads
# is the kind of shortcut this document keeps finding was wrong. So dflash2
# replays the same 2,742 calls a second time and provides its own floor.
set -uo pipefail
R=/mnt/data/src/muse-glimmer-dspark
L=/mnt/data/logs
say() { echo "[$(date -Is)] CHAIN3: $*" | tee -a "$L/night_chain3.log"; }
WAIT_PID="${WAIT_PID:-}"
if [ -n "$WAIT_PID" ]; then
  say "waiting on pid $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
while sudo docker ps --format '{{.Names}}' | grep -q '^vllm-'; do sleep 60; done
say "GPU free; dflash2 repeat on the SWE-bench Multilingual call set"
LABEL=dflash2 SPEC=/mnt/data/speculators/dflash2 METHOD=dflash PORT=8001 REPO="$R" \
  OUTROOT=/mnt/data/eval/mlsweep WARMUP_SRC=/mnt/data/eval/swebench_ml_calls.jsonl \
  bash "$R/scripts/sweep_serve.sh" >> "$L/mlrepeat.log" 2>&1
if [ $? = 0 ]; then
  CELL=dflash2__mlfull__c10__r2 SRC=/mnt/data/eval/swebench_ml_calls.jsonl CONC=10 \
    LABEL=dflash2 PORT=8001 REPO="$R" OUTROOT=/mnt/data/eval/mlsweep \
    bash "$R/scripts/sweep_cell.sh" >> "$L/mlrepeat.log" 2>&1
  say "repeat cell rc=$?"
else
  say "server failed; see $L/mlrepeat.log"
fi
LABEL=dflash2 PORT=8001 bash "$R/scripts/sweep_down.sh" >> "$L/mlrepeat.log" 2>&1
say "########## CHAIN3 FINISHED ##########"
