#!/bin/bash
# Optional last stage: replay the SWE-Gym holdout call set through dflash2 and
# the run-D midpoint.
#
# This is the prior agent's open question and it now has a use. The reasoning
# share of the generated text is 56.8% in training, 62.9% on SWE-bench
# Multilingual and 75.3% on Terminal-Bench, and the fine-tune's step-weighted
# gain over dflash2 runs +0.300 / -0.09 on the latter two. The SWE-Gym holdout
# is unseen instances in seen repositories, so it should sit at the training end
# of that axis and show the largest gain. Two points make a direction; three
# make it checkable.
#
# Runs last and is entirely skippable: the three commissioned tasks do not
# depend on it.
set -uo pipefail
R=/mnt/data/src/muse-glimmer-dspark
L=/mnt/data/logs
SRC=/mnt/data/eval/swegym_holdout_calls.jsonl
say() { echo "[$(date -Is)] CHAIN4: $*" | tee -a "$L/night_chain4.log"; }
WAIT_PID="${WAIT_PID:-}"
if [ -n "$WAIT_PID" ]; then
  say "waiting on pid $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
while sudo docker ps --format '{{.Names}}' | grep -q '^vllm-'; do sleep 60; done
[ -s "$SRC" ] || { say "no holdout call set at $SRC; nothing to do"; exit 0; }
say "GPU free; replaying $(wc -l < "$SRC") SWE-Gym holdout calls"

run() {  # label spec
  LABEL="$1" SPEC="$2" METHOD=dflash PORT=8001 REPO="$R" \
    OUTROOT=/mnt/data/eval/gymsweep WARMUP_SRC="$SRC" \
    bash "$R/scripts/sweep_serve.sh" >> "$L/gymreplay.log" 2>&1 \
    || { say "$1 server failed"; LABEL="$1" PORT=8001 bash "$R/scripts/sweep_down.sh" >>"$L/gymreplay.log" 2>&1; return 1; }
  CELL="$1__gymfull__c10__r1" SRC="$SRC" CONC=10 LABEL="$1" PORT=8001 REPO="$R" \
    OUTROOT=/mnt/data/eval/gymsweep bash "$R/scripts/sweep_cell.sh" >> "$L/gymreplay.log" 2>&1
  say "$1 cell rc=$?"
  LABEL="$1" PORT=8001 bash "$R/scripts/sweep_down.sh" >> "$L/gymreplay.log" 2>&1
}
run dflash2 /mnt/data/speculators/dflash2
run dflash2-run-d-mid /mnt/data/speculators/dflash2-run-d-32k-mix-step1976
say "########## CHAIN4 FINISHED ##########"
