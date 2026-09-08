#!/bin/bash
# Serve Muse Glimmer with a speculative drafter, patching vLLM in place first.
#
# Stock vLLM cannot load a DSpark drafter against a Muse Glimmer target:
# dspark/utils.py does an unguarded `target_language_model.model`, but
# MuseGlimmerForCausalLM marks its inner MuseGlimmerModel as the language model,
# so get_language_model() already returns the inner module and it has no .model.
#
#   AttributeError: 'MuseGlimmerModel' object has no attribute 'model'
#
# vLLM PR #51655 ("Add Muse Glimmer model support") fixed exactly this line in
# the *DFlash* loader and never mirrored it onto DSpark -- DSpark was built for
# DeepSeek V4 (#46995) and later Qwen3-Omni (#52560). The DaoCloud model card is
# accurate when it says it was validated against patched builds.
#
# Patching in place beats rebuilding: it is one line, and it keeps the image
# pinned to whatever nightly digest the run was measured against.
#
# SPEC_METHOD/SPECULATOR select the drafter:
#   dflash + meta-models/Muse-Glimmer-30B-assistant   (official, the baseline)
#   dspark + DaoCloud/Muse-Glimmer-30B-DSpark         (community, needs the patch)
set -euo pipefail

TARGET_MODEL="${TARGET_MODEL:-meta-models/Muse-Glimmer-30B}"
SPEC_METHOD="${SPEC_METHOD:-dflash}"
SPECULATOR="${SPECULATOR:-meta-models/Muse-Glimmer-30B-assistant}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-15}"
PORT="${PORT:-8001}"
# Set when the port is internet-facing. Cross-VPC runs cannot use the private
# address, so the proxy reaches the endpoint over its public IP -- an unset key
# there leaves an open H200 on the internet.
VLLM_API_KEY="${VLLM_API_KEY:-}"

F=/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/dspark/utils.py
if [ -f "$F" ]; then
  sed -i 's/^    target_inner = target_language_model\.model$/    target_inner = getattr(target_language_model, "model", target_language_model)/' "$F"
  # Fail loudly now rather than after a 60 GB download.
  grep -q 'getattr(target_language_model' "$F" || { echo "DSPARK PATCH FAILED" >&2; exit 1; }
  echo "dspark patch OK: $(grep -n 'target_inner = ' "$F")"
fi

# Stock vLLM drops DFlash2's two output-shaping knobs when it rebuilds a
# dflash_config from a speculators-format checkpoint. update_dflash2() forwards
# only conv_kernel_size, conv_group_size, selector_rank and selector_top_k --
# and DFlash2Qwen3ForCausalLM reads output_multiplier and
# final_logit_softcapping straight back out of that same dict to build
# LogitsProcessor(scale=..., soft_cap=...). Without them, a converted
# z-lab/Muse-Glimmer-30B-DFlash2 serves at scale=1.0 with no cap while its
# weights (and, with patches/speculators-dflash2-output-shaping.patch, its
# fine-tune) were fit for 0.196 and a cap of 20.0. That is a 5.1x logit
# mismatch and nothing in the stack errors on it.
#
# Native z-lab checkpoints are unaffected -- they keep their own dflash_config
# and never go through update_dflash2. This only matters for converted ones.
G=/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/configs/speculators/algos.py
if [ -f "$G" ]; then
  if ! grep -q '"output_multiplier",' "$G"; then
    sed -i 's/^        "selector_top_k",$/        "selector_top_k",\n        "output_multiplier",\n        "final_logit_softcapping",/' "$G"
  fi
  grep -q '"final_logit_softcapping",' "$G" || {
    echo "DFLASH2 OUTPUT-SHAPING PATCH FAILED" >&2; exit 1; }
  echo "dflash2 knob patch OK: $(grep -c '"output_multiplier",' "$G") site(s)"
fi

# --per-request-spec-decode-metrics puts acceptance length and draft acceptance
# rate in each response body. The aggregate vllm:spec_decode_* counters are
# server-global and blend in any other workload sharing the endpoint; the
# per-request numbers are attributable, and the recording proxy already stores
# whole response bodies.
AUTH_ARGS=()
if [ -n "${VLLM_API_KEY}" ]; then
  AUTH_ARGS=(--api-key "${VLLM_API_KEY}")
fi

# DSpark-only: size the draft budget per request from the checkpoint's
# confidence head instead of always drafting the full block. DFlash/DFlash2 have
# no confidence head, so vLLM refuses this for them -- it is a lever only a
# DSpark drafter has.
ADAPTIVE="${ADAPTIVE:-0}"
SPEC_CFG="{\"method\": \"${SPEC_METHOD}\", \"model\": \"${SPECULATOR}\", \"num_speculative_tokens\": ${NUM_SPEC_TOKENS}}"
if [ "${ADAPTIVE}" = "1" ]; then
  SPEC_CFG="{\"method\": \"${SPEC_METHOD}\", \"model\": \"${SPECULATOR}\", \"num_speculative_tokens\": ${NUM_SPEC_TOKENS}, \"enable_adaptive_verification\": true}"
  echo "adaptive verification ON"
fi
SPEC_ARGS=(--per-request-spec-decode-metrics summary --speculative-config "${SPEC_CFG}")
# SPEC_METHOD=none serves the target alone -- the autoregressive control every
# speedup multiple is quoted against.
if [ "${SPEC_METHOD}" = "none" ]; then SPEC_ARGS=(); echo "no-spec control: target only"; fi

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "${TARGET_MODEL}" \
  "${AUTH_ARGS[@]}" \
  --host 0.0.0.0 --port "${PORT}" \
  --generation-config auto \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer \
  --max-num-seqs 128 \
  --enable-prefix-caching \
  --max-num-batched-tokens 8192 \
  "${SPEC_ARGS[@]}"
