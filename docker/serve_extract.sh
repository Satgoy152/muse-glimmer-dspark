#!/bin/bash
# Target-model server for training: renders conversations for `prepare-data` and
# streams hidden states during training. This is NOT the eval server -- that one
# is docker/serve_patched.sh, which loads a drafter and reports per-request
# speculative-decoding metrics. Keep them on separate ports and separate boxes
# if you run both.
#
# No DSpark patch here: nothing loads a drafter, so vllm/v1/worker/gpu/
# spec_decode/dspark/utils.py is never imported.
set -euo pipefail

MODEL="${MODEL:-meta-models/Muse-Glimmer-30B}"
PORT="${PORT:-8000}"
# Must match configs/train_dspark.yaml data.total_seq_len and the --seq-length
# passed to prepare-data. +1 leaves room for the single decoded token that the
# max_tokens=1 hidden-state request emits.
SEQ_LENGTH="${SEQ_LENGTH:-16384}"
HIDDEN_STATES_PATH="${HIDDEN_STATES_PATH:-/data/hidden_states}"
SPECULATORS_DIR="${SPECULATORS_DIR:-/data/speculators}"

# From DaoCloud/Muse-Glimmer-30B-DSpark config.json: aux_hidden_state_layer_ids.
# launch_vllm.py appends the final layer (52) itself under --include-last-layer;
# training gets the auxiliary five only.
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-2 14 26 38 50}"

mkdir -p "${HIDDEN_STATES_PATH}"

exec python3 "${SPECULATORS_DIR}/scripts/launch_vllm.py" "${MODEL}" \
  --hidden-states-path "${HIDDEN_STATES_PATH}" \
  --target-layer-ids ${TARGET_LAYER_IDS} \
  -- \
  --port "${PORT}" \
  --max-model-len "$((SEQ_LENGTH + 1))" \
  --tensor-parallel-size "${TP:-1}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" \
  --max-num-seqs "${MAX_NUM_SEQS:-32}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-32768}" \
  --no-enable-prefix-caching
