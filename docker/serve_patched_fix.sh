#!/bin/bash
# Serve wrapper for the conversion-confound control.
#
# vLLM rebuilds dflash_config from scratch when loading a speculators-format
# checkpoint (transformers_utils/configs/speculators/algos.py::update_dflash),
# forwarding only mask_token_id/target_layer_ids/sample_from_anchor/causal plus
# four conv+selector keys. output_multiplier and final_logit_softcapping are
# present in the checkpoint config.json but never reach the model, so
# qwen3_dflash2.py falls back to scale=1.0 (a 5.1x logit error) and softcap=0.0.
# The native z-lab checkpoint keeps its own dflash_config and is unaffected --
# which is precisely the asymmetry this control removes.
# Historical: kept as provenance for the scale-fix control. serve_patched.sh
# now applies this patch inline, so new runs do not need this wrapper. The
# patcher it calls is committed at patches/patch_algos.py; on the node it lived
# at /mnt/data/patch_algos.py, which is the path below.
set -euo pipefail
python3 "${PATCH_ALGOS:-/mnt/data/patch_algos.py}"
exec bash "$(dirname "$0")/serve_patched.sh"
