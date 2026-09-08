#!/bin/bash
# 1) scale-fix control: same run-d ckpt, served with output_multiplier/softcap restored
# 2) noise floor: repeat the native dflash2 anchor unchanged
cd /mnt/data/src/muse-glimmer-dspark
echo "######## START scalefix $(date -u +%H:%M:%S)"
NAME=dflash2-run-d-step1976-scalefix \
  SPEC=/mnt/data/speculators/dflash2-run-d-32k-mix-step1976 METHOD=dflash \
  bash /mnt/data/eval_replay_fix.sh 2>&1
echo "######## EXIT scalefix rc=$? $(date -u +%H:%M:%S)"
echo "######## START repeat2 $(date -u +%H:%M:%S)"
NAME=dflash2-repeat2 SPEC=/mnt/data/speculators/dflash2 METHOD=dflash \
  bash scripts/eval_replay.sh 2>&1
echo "######## EXIT repeat2 rc=$? $(date -u +%H:%M:%S)"
echo "######## ALL DONE"
