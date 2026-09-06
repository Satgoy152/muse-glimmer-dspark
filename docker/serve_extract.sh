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
# Bound to loopback below: the trainer and prepare-data both reach this over
# 127.0.0.1, and an unauthenticated vLLM on a public port is an open H200. On
# the bring-up node it was being scanned within two hours of coming up.
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
# The attention page is per rank; the hidden-state page is not. Hidden states
# are the full residual stream and every rank holds all of it, while TP shards
# the attention KV -- and with only 2 KV heads, any TP > 1 gives
# max(1, 2 // TP) = 1 head per rank and halves the attention page:
#
#   TP=1, block 128:  2 x 128 x 2 x 2 x 128 = 131,072 B  >= 79,872   ok
#   TP=4, block 128:  2 x 128 x 1 x 2 x 128 =  65,536 B  <  79,872   assert
#   TP=4, block 256:  2 x 128 x 1 x 2 x 256 = 131,072 B  >= 79,872   ok
#
# So the default here is 256, which is correct for TP 2, 4 and 8; 128 is only
# enough at TP=1. Either way the hidden-state page pads to 131,072 B, wasting
# 51,200 B (39.06%) per block, which is what makes the window ceiling as low as
# it is. Measured on 4xH200 at TP=4: 63,908 tokens of KV cache at
# --gpu-memory-utilization 0.92.
BLOCK_SIZE="${BLOCK_SIZE:-256}"

# Prefix caching is BROKEN for hidden-state extraction. Measured: two runs with
# caching off produce bitwise-identical hidden states (max abs diff 0), while a
# run with caching on differs from that baseline by max 348 / mean 0.134 over
# the cached span. The mechanism is the chunked-prefill bug: hidden states are
# written by ExtractHiddenStatesProposer.propose(), a cache hit does not
# recompute the span, so propose() never runs over it -- yet request_finished
# still reads len(prompt_token_ids) slots out of the block list. Shapes,
# token ids and finiteness all check out; only the values are stale.
#
# Do not re-enable it, and do not reorder rows by trajectory to chase cache
# hits. The upside was capped anyway: hidden-state serialisation scales with
# total tokens regardless of hits, so the measured speed-up was only ~1.5x.
PREFIX_CACHING="${PREFIX_CACHING:-off}"
if [ "${PREFIX_CACHING}" = "on" ]; then
  PREFIX_ARGS=(--enable-prefix-caching)
else
  PREFIX_ARGS=(--no-enable-prefix-caching)
fi

# Every trajectory in this dataset carries `tools`, and the render endpoint
# rejects a request with tools unless the server was started to parse tool
# calls: {"message": "\"auto\" tool choice requires --enable-auto-tool-choice
# and --tool-call-parser to be set", "code": 400}. speculators catches that per
# conversation and returns no rows, so prepare-data reports only "No samples
# remain after preprocessing. Check the dataset schema, assistant masking, and
# --minimum-valid-tokens" -- none of which is the actual cause.
TOOL_ARGS=(--enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER:-muse_glimmer}")

mkdir -p "${HIDDEN_STATES_PATH}"

exec python3 "${SPECULATORS_DIR}/scripts/launch_vllm.py" "${MODEL}" \
  --hidden-states-path "${HIDDEN_STATES_PATH}" \
  --target-layer-ids ${TARGET_LAYER_IDS} \
  -- \
  --port "${PORT}" \
  --host 127.0.0.1 \
  --max-model-len "$((SEQ_LENGTH + 1))" \
  --tensor-parallel-size "${TP}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" \
  --max-num-seqs "${MAX_NUM_SEQS:-32}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --block-size "${BLOCK_SIZE}" \
  "${TOOL_ARGS[@]}" \
  --no-enable-chunked-prefill \
  "${PREFIX_ARGS[@]}"
