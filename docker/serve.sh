#!/bin/bash
set -euo pipefail

TARGET_MODEL="${TARGET_MODEL:-meta-models/Muse-Glimmer-30B}"
TP_SIZE="${TP_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-15}"
SPEC_METHOD="${SPEC_METHOD:-dspark}"

# Trace generation wants the target alone: a drafter is lossless in
# distribution but adds a moving part, and generation is the run that must not
# be re-done. Set SPECULATOR_PATH="" to serve without it.
SPEC_ARGS=()
if [ -n "${SPECULATOR_PATH:-}" ]; then
  SPEC_ARGS=(--speculative-config \
    "{\"method\": \"${SPEC_METHOD}\", \"model\": \"${SPECULATOR_PATH}\", \"num_speculative_tokens\": ${NUM_SPEC_TOKENS}}")
fi

# Model is positional in the nightly entrypoint, not --model.
exec python3 -m vllm.entrypoints.openai.api_server \
  "${TARGET_MODEL}" \
  --served-model-name meta/muse-glimmer-30b \
  --generation-config auto \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice \
  --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer \
  "${SPEC_ARGS[@]}" \
  --host 0.0.0.0 --port 8000
