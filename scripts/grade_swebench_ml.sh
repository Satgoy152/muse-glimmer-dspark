#!/bin/bash
# Grade the on-policy SWE-bench Multilingual runs. CPU and docker only, so this
# can run alongside a GPU job.
#
# The denominator is the full task count, not the number of predictions: an
# instance whose agent crashed is unresolved, not absent. Passing --n-expected to
# benchmark/analysis/wilson.py is what enforces that.
set -uo pipefail
G=/mnt/data/gradevenv/bin/python
SUBSET="${SUBSET:-/mnt/data/bench/swebench_ml}"
RUNS="${RUNS:-/mnt/data/runs/mlonpolicy}"
N=$(wc -l < "$SUBSET/test.jsonl")
cd /mnt/data/runs

# The target-only recording run is graded too, and it is the arm that matters
# most: speculative decoding is supposed to be output-preserving, so a drafter's
# resolved rate should match the target's. docs/RESULTS-eval-sweep.md already
# shows two places where output preservation does not hold in practice, which
# makes this a test rather than a formality.
for d in /mnt/data/runs/swebench-ml "$RUNS"/*/; do
  L=$(basename "$d")
  [ "$L" = "swebench-ml" ] && L="target-only"
  [ -s "$d/preds.json" ] || { echo "skip $L: no preds.json"; continue; }
  echo "=== grading $L ==="
  $G -m swebench.harness.run_evaluation \
    --dataset_name swe-bench/SWE-Bench_Multilingual --split test \
    --predictions_path "$d/preds.json" --run_id "ml-$L" \
    --max_workers 4 --cache_level env --timeout 1800 \
    2>&1 | tail -20
done

echo "=== resolved rate, Wilson 95% ==="
/mnt/data/src/muse-glimmer-dspark/.venv/bin/python \
  /mnt/data/src/muse-glimmer-dspark/benchmark/analysis/wilson.py \
  /mnt/data/runs/*.ml-*.json --n-expected "$N" 2>/dev/null \
  || ls /mnt/data/runs/*.json
