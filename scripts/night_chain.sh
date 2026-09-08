#!/bin/bash
# Run the night's GPU work back to back, unattended. Each stage is idempotent,
# so a preempted node resumes by re-running this file.
set -uo pipefail
R=/mnt/data/src/muse-glimmer-dspark
L=/mnt/data/logs
say() { echo "[$(date -Is)] CHAIN: $*" | tee -a "$L/night_chain.log"; }

# Stage 0: wait out whatever is already running, so this can be started at any
# time without fighting the sweep for the GPU.
#
# The wait is on a recorded PID, not on a pgrep pattern. A pattern like
# "sweep_task1.sh" also matches any ssh command line that mentions it -- including
# the monitor that polls this node -- so a pgrep wait can hang on its own watcher.
WAIT_PID="${WAIT_PID:-}"
if [ -n "$WAIT_PID" ]; then
  say "waiting on pid $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
# Belt and braces: do not start while anything still owns the GPU.
while sudo docker ps --format '{{.Names}}' | grep -q '^vllm-'; do sleep 60; done
say "GPU is free"

say "stage 1: SWE-bench Multilingual target-only recording"
bash "$R/scripts/swebench_ml_record.sh" >> "$L/mlrec.log" 2>&1
say "stage 1 rc=$?"

if [ -s /mnt/data/eval/swebench_ml_calls.jsonl ]; then
  say "stage 2: replay + on-policy sweep"
  bash "$R/scripts/swebench_ml_sweep.sh" >> "$L/mlsweep.log" 2>&1
  say "stage 2 rc=$?"
else
  say "stage 2 SKIPPED: no replayable calls were recorded"
fi
say "########## NIGHT CHAIN FINISHED ##########"
