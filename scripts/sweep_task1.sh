#!/bin/bash
# Driver for the Terminal-Bench bucket x concurrency sweep (task 1), with the
# task-3 full-set repeats folded into the servers that are up anyway.
#
# One server per drafter, all of that drafter's cells against it, then the next
# drafter. Restarting the server per cell would be 128 model loads at ~7 min.
#
# Everything is idempotent: sweep_cell.sh skips a cell whose result json exists,
# so a preempted node resumes by re-running this script unchanged.
set -uo pipefail

REPO="${REPO:-/mnt/data/src/muse-glimmer-dspark}"
MAN="${MAN:-/mnt/data/eval/manifests}"
OUTROOT="${OUTROOT:-/mnt/data/eval/sweep}"
PORT="${PORT:-8001}"
STATE="$OUTROOT/driver.log"
mkdir -p "$OUTROOT"

BUCKETS="b64_128 b128_256 b256_1K bge1K"
CONCS="${CONCS:-1 2 8 32}"

say() { echo "[$(date -Is)] $*" | tee -a "$STATE"; }

run_label() {
  local LABEL="$1" SPEC="$2" METHOD="$3" EXTRA="${4:-}"
  say "### START $LABEL (method=$METHOD)"

  # already fully done? then do not pay the 7-minute load
  local need=0
  for b in $BUCKETS; do for c in $CONCS; do
    [ -s "$OUTROOT/$LABEL/${LABEL}__${b}__c${c}__r1.json" ] || need=1
  done; done
  case "$EXTRA" in *ctl*) for b in $BUCKETS; do for c in 1 32; do
        [ -s "$OUTROOT/$LABEL/${LABEL}__${b}__c${c}__r2.json" ] || need=1; done; done;; esac
  case "$EXTRA" in *task3*) for r in 2 3; do
        [ -s "$OUTROOT/$LABEL/${LABEL}__full__c10__t1p0__r${r}.json" ] || need=1; done;; esac
  if [ "$need" = 0 ]; then say "### $LABEL already complete, skipping"; return 0; fi

  # REUSE=1 adopts a still-healthy server for this exact label -- what happens
  # when the driver is restarted but the node was not. sweep_serve.sh verifies
  # the container's env before adopting it.
  LABEL="$LABEL" SPEC="$SPEC" METHOD="$METHOD" PORT="$PORT" REPO="$REPO" REUSE=1 \
    OUTROOT="$OUTROOT" bash "$REPO/scripts/sweep_serve.sh" 2>&1 | tee -a "$STATE"
  if [ "${PIPESTATUS[0]}" != 0 ]; then
    say "### $LABEL SERVER FAILED -- skipping this drafter"
    LABEL="$LABEL" PORT="$PORT" bash "$REPO/scripts/sweep_down.sh" >>"$STATE" 2>&1
    return 1
  fi

  # task 3 first while the server is freshly warm: these are the full 1,753-call
  # temperature-1.0 repeats, and they are the cheapest thing here per unit of
  # information (~10 min each).
  case "$EXTRA" in *task3*)
    for r in 2 3; do
      CELL="${LABEL}__full__c10__t1p0__r${r}" SRC=/mnt/data/eval/raw.parquet \
        CONC=10 LABEL="$LABEL" PORT="$PORT" REPO="$REPO" OUTROOT="$OUTROOT" \
        bash "$REPO/scripts/sweep_cell.sh" 2>&1 | tee -a "$STATE"
    done;;
  esac

  for c in $CONCS; do
    for b in $BUCKETS; do
      CELL="${LABEL}__${b}__c${c}__r1" SRC="$MAN/$b.jsonl" CONC="$c" \
        LABEL="$LABEL" PORT="$PORT" REPO="$REPO" OUTROOT="$OUTROOT" \
        bash "$REPO/scripts/sweep_cell.sh" 2>&1 | tee -a "$STATE"
    done
  done

  # repeat control: the same drafter, same cells, second pass. Without it a
  # bucket x concurrency difference has nothing to be compared against.
  case "$EXTRA" in *ctl*)
    for c in 1 32; do
      for b in $BUCKETS; do
        CELL="${LABEL}__${b}__c${c}__r2" SRC="$MAN/$b.jsonl" CONC="$c" \
          LABEL="$LABEL" PORT="$PORT" REPO="$REPO" OUTROOT="$OUTROOT" \
          bash "$REPO/scripts/sweep_cell.sh" 2>&1 | tee -a "$STATE"
      done
    done;;
  esac

  LABEL="$LABEL" PORT="$PORT" bash "$REPO/scripts/sweep_down.sh" 2>&1 | tee -a "$STATE"
  say "### DONE $LABEL"
}

S=/mnt/data/speculators
# Order is by how much depends on it: the no-spec denominator first, then the
# baseline that also carries the repeat control and the task-3 repeats.
run_label nospec              none                            none   ""
run_label dflash2             "$S/dflash2"                    dflash "ctl task3"
run_label dflash2-run-d-mid   "$S/dflash2-run-d-32k-mix-step1976" dflash "task3"
run_label dflash2-run-d-final "$S/dflash2-run-d-32k-mix"      dflash "task3"
run_label dflash-official     "$S/dflash-official"            dflash ""
run_label dspark-run-a-32k    "$S/dspark-run-a-32k"           dspark ""
run_label dspark-community    "$S/dspark-community"           dspark ""
run_label dspark-run-b-49k    "$S/dspark-run-b-49k"           dspark ""

say "########## SWEEP DRIVER FINISHED ##########"
