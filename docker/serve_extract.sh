#!/bin/bash
# Target-model server for training: renders conversations for `prepare-data` and
# extracts hidden states during training. This is NOT the eval server -- that is
# docker/serve_patched.sh, which loads a drafter and reports per-request
# speculative-decoding metrics.
#
# No DSpark patch here: nothing loads a drafter, so vllm/v1/worker/gpu/
# spec_decode/dspark/utils.py is never imported.
#
# launch_vllm.py turns --target-layer-ids into two vLLM flags:
#   --speculative_config {"method":"extract_hidden_states","num_speculative_tokens":1,
#                         "draft_model_config":{"hf_config":
#                           {"eagle_aux_hidden_state_layer_ids":[...]}}}
#   --kv_transfer_config  {"kv_connector":"ExampleHiddenStatesConnector", ...}
# "extract_hidden_states" is not real speculation -- it writes each layer's
# hidden states into a dedicated KV cache group, and the connector serialises
# them to HIDDEN_STATES_PATH. The client gets the file path back in the
# response's kv_transfer_params.
set -euo pipefail

MODEL="${MODEL:-meta-models/Muse-Glimmer-30B}"
PORT="${PORT:-8000}"
# Must match configs/train_dspark.yaml data.total_seq_len and prepare-data
# --seq-length. +1 covers the single decoded token of the max_tokens=1 request.
SEQ_LENGTH="${SEQ_LENGTH:-49152}"
HIDDEN_STATES_PATH="${HIDDEN_STATES_PATH:-/data/hidden_states}"
SPECULATORS_DIR="${SPECULATORS_DIR:-/data/speculators}"

# From DaoCloud/Muse-Glimmer-30B-DSpark config.json: aux_hidden_state_layer_ids.
# launch_vllm.py appends the final layer (52) under --include-last-layer;
# training takes the auxiliary five only.
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-2 14 26 38 50}"

# TP over DP: hidden states are cached per token at ~78 KB (6 layers x 6656 x
# bf16), so cache capacity, not FLOPs, is the binding resource. Sharding the
# weights frees ~45 GB/GPU over replicating them, and pools the remainder into
# one cache that a whole trajectory's prefix can sit in.
TP="${TP:-4}"

# Chunked prefill is incompatible with extract_hidden_states, so a prefill
# arrives as one batch and the token budget must cover a full sequence.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$((SEQ_LENGTH + 1))}"

# The hidden-state cache is a KV cache group of its own, and vLLM aligns its
# page to the attention page. Muse Glimmer is GQA 32:2 with head_dim 128, so at
# the default block size its attention page is only 2 x 128 x 2 x 2 x 16 =
# 16,384 B, while one token of hidden state is 6 x 6656 x 2 = 79,872 B. The
# alignment in kv_cache_utils._get_kv_cache_groups then clamps the hidden-state
# block size to 1 and still overflows, tripping
#   assert self.page_size_padded >= self.unpadded_page_size_bytes
# in kv_cache_interface.py during cudagraph memory profiling.
#
# block_size 128 lifts the attention page to 131,072 B, the first power of two
# above 79,872, which is the smallest value that lets the alignment succeed. It
# costs 39% padding waste inside the hidden-state cache.
BLOCK_SIZE="${BLOCK_SIZE:-128}"

# Whether a prefix-cache hit still yields hidden states for the cached span is
# unverified -- see docs/train.md. `off` is the safe default and what the
# upstream large-model example uses; `on` is worth measuring, because turns in
# one trajectory share almost all of their context.
PREFIX_CACHING="${PREFIX_CACHING:-off}"
if [ "${PREFIX_CACHING}" = "on" ]; then
  PREFIX_ARGS=(--enable-prefix-caching)
else
  PREFIX_ARGS=(--no-enable-prefix-caching)
fi

mkdir -p "${HIDDEN_STATES_PATH}"

exec python3 "${SPECULATORS_DIR}/scripts/launch_vllm.py" "${MODEL}" \
  --hidden-states-path "${HIDDEN_STATES_PATH}" \
  --target-layer-ids ${TARGET_LAYER_IDS} \
  -- \
  --port "${PORT}" \
  --max-model-len "$((SEQ_LENGTH + 1))" \
  --tensor-parallel-size "${TP}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" \
  --max-num-seqs "${MAX_NUM_SEQS:-32}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --block-size "${BLOCK_SIZE}" \
  --no-enable-chunked-prefill \
  "${PREFIX_ARGS[@]}"
