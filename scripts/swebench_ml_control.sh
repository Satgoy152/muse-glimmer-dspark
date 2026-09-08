#!/bin/bash
# A second target-only agent pass over the same 32 instances.
#
# Why this exists: the first target-only run resolved 13/32 and dflash2
# on-policy resolved 6/32, exact McNemar p=0.065 on the discordant pairs. That
# looks like a drafter effect and may be nothing of the kind -- both are single
# stochastic rollouts at temperature 1.0, so two runs of the *same* model differ
# by construction, and nothing measured so far says by how much.
#
# Task 3 in this same document turned on precisely this mistake: a single pair of
# runs read as a difference. So before the resolved-rate table is written, the
# target model is run again and the two target-only rollouts are compared to
# each other. That is the rollout-variance floor, and no drafter delta smaller
# than it means anything.
#
# ~37 minutes. It also yields a second frozen call set as a by-product.
set -uo pipefail
R=/mnt/data/src/muse-glimmer-dspark
TD=/mnt/data/traces/swebench-ml-record-r2 \
RUNS=/mnt/data/runs/mlonpolicy/target-only-r2 \
OUT=/mnt/data/eval/swebench_ml_calls_r2.jsonl \
CTR=vllm-mlrec-r2 PORT=8052 PROXY_PORT=8084 VALIDATE_FIRST=0 \
  bash "$R/scripts/swebench_ml_record.sh"
echo "=== CONTROL ROLLOUT COMPLETE ==="
