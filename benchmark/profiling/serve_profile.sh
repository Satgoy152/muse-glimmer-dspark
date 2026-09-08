#!/usr/bin/env bash
# Run inside a fresh specd:latest container with /mnt/data and Nsight mounted.
set -euo pipefail
REPO="${REPO:-/mnt/data/src/muse-glimmer-profile}"
PROFILE_DIR="${PROFILE_DIR:-/mnt/data/profiles/dflash2-20260908}"
NSYS="${NSYS:-/opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/nsys}"
export PYTHONPATH="$REPO/benchmark/profiling${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/mnt/data/hf
export HF_HUB_OFFLINE=1
export VLLM_USE_V2_MODEL_RUNNER=1
mkdir -p "$PROFILE_DIR"
python3 "$REPO/benchmark/profiling/fix_profiler_attr.py"
python3 "$REPO/benchmark/profiling/annotate_vllm.py"
exec "$NSYS" profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --trace-fork-before-exec=true \
  --capture-range=cudaProfilerApi --capture-range-end=repeat \
  --force-overwrite=false --output="$PROFILE_DIR/capture" \
  python3 -m vllm.entrypoints.openai.api_server \
  --model meta-models/Muse-Glimmer-30B \
  --host 127.0.0.1 --port 8002 \
  --generation-config auto --tensor-parallel-size 1 \
  --max-num-seqs 128 --enable-prefix-caching --max-num-batched-tokens 8192 \
  --per-request-spec-decode-metrics summary \
  --profiler-config '{"profiler":"cuda","delay_iterations":15,"max_iterations":50}' \
  --speculative-config '{"method":"dflash","model":"/mnt/data/speculators/dflash2","num_speculative_tokens":15}'
