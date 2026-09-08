#!/bin/bash
# Follow-on stage: the target-only control rollout, then grade everything that
# has appeared since. Waits on a recorded PID rather than a pgrep pattern -- a
# pattern also matches the ssh command line that carries it.
set -uo pipefail
R=/mnt/data/src/muse-glimmer-dspark
L=/mnt/data/logs
say() { echo "[$(date -Is)] CHAIN2: $*" | tee -a "$L/night_chain2.log"; }
WAIT_PID="${WAIT_PID:-}"
if [ -n "$WAIT_PID" ]; then
  say "waiting on pid $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
while sudo docker ps --format '{{.Names}}' | grep -q '^vllm-'; do sleep 60; done
say "GPU free; starting target-only control rollout"
bash "$R/scripts/swebench_ml_control.sh" >> "$L/mlcontrol.log" 2>&1
say "control rollout rc=$?"
say "grading every arm"
bash "$R/scripts/grade_swebench_ml.sh" >> "$L/grade.log" 2>&1
say "grading rc=$?"
say "########## CHAIN2 FINISHED ##########"
