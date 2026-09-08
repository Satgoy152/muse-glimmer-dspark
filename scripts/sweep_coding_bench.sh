#!/bin/bash
# Run one coding benchmark across every drafter, sequentially (one GPU).
#
#   bash scripts/sweep_coding_bench.sh HumanEval
#   bash scripts/sweep_coding_bench.sh MBPP
#
# Order is deliberate: the two drafters with hard serving gates -- dflash2 (V2
# runner) and dspark-community (dspark patch) -- run second and third, so a gate
# failure surfaces ~25 min in rather than after the whole sweep.
#
# A failing drafter does not abort the others; it is logged and the sweep moves
# on, because one bad drafter should not cost the other four their GPU time.
# Check for FAILED in the sweep log before trusting a full set of rows.
#
# NAME carries the task prefix. Reusing a NAME overwrites that row's trace and
# eval JSON in place, which is why eval_humaneval.sh refuses to do it.
set -uo pipefail

TASK="${1:-HumanEval}"
REPO="${REPO:-/mnt/data/src/muse-glimmer-dspark}"
case "$TASK" in
  HumanEval) SRC=/mnt/data/bench/speculator_benchmarks/HumanEval.jsonl; PREFIX="" ;;
  MBPP)      SRC=/mnt/data/bench/mbpp_test.jsonl;                       PREFIX="mbpp-" ;;
  *) echo "unknown task: $TASK (expected HumanEval or MBPP)" >&2; exit 2 ;;
esac
[ -f "$SRC" ] || { echo "missing prompt file $SRC" >&2; exit 1; }

run() {
  local name="${PREFIX}$1" spec="$2" method="$3"
  echo "##### $(date -Is) START $name #####"
  if NAME="$name" SPEC="/mnt/data/speculators/$spec" METHOD="$method" \
     TASK="$TASK" SRC="$SRC" CONC=1 TEMP=0 MAXTOK=2048 PASS1=1 \
     bash "$REPO/scripts/eval_humaneval.sh" > "/mnt/data/logs/he-$name.log" 2>&1; then
    echo "##### $(date -Is) OK $name #####"; tail -14 "/mnt/data/logs/he-$name.log"
  else
    echo "##### $(date -Is) FAILED $name #####"; tail -25 "/mnt/data/logs/he-$name.log"
  fi
  sudo docker rm -f "vllm-he-$name" >/dev/null 2>&1 || true
  sleep 5
}

run dflash-official                  dflash-official                  dflash
run dflash2                          dflash2                          dflash
run dspark-community                 dspark-community                 dspark
run dspark-run-a-32k                 dspark-run-a-32k                 dspark
run dflash2-run-d-32k-mix-step1976   dflash2-run-d-32k-mix-step1976   dflash
echo "##### $(date -Is) $TASK SWEEP DONE #####"
