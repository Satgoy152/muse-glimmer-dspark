#!/bin/bash
# Latency-under-contention sweep at concurrency 64.
# Names are _c64-* so make_summary.py (which skips leading "_") leaves the
# canonical concurrency-10 acceptance table alone, and so we never overwrite
# an existing trace or eval json.
cd /mnt/data/src/muse-glimmer-dspark
run () {  # $1 name  $2 spec dir  $3 method
  echo "######## START $1 $(date -u +%H:%M:%S)"
  NAME="_c64-$1" SPEC="/mnt/data/speculators/$2" METHOD="$3" CONC=64 \
    bash scripts/eval_replay.sh 2>&1
  echo "######## EXIT $1 rc=$? $(date -u +%H:%M:%S)"
}
run dflash2          dflash2                         dflash
run dspark-run-a-32k dspark-run-a-32k                dspark
run dflash-run-d-mid dflash2-run-d-32k-mix-step1976  dflash
run dspark-community dspark-community                dspark
run dflash-official  dflash-official                 dflash
echo "######## ALL DONE"
