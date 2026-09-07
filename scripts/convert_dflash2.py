#!/usr/bin/env python3
"""Convert a native z-lab DFlash2 checkpoint to speculators format.

`speculators` ships `DFlashConverter` but no DFlash2 equivalent, so
`--from-pretrained z-lab/Muse-Glimmer-30B-DFlash2` dies in
`SpeculatorModelConfig.from_pretrained`:

    NotImplementedError: Loading a non-speculator model config is not supported yet.

The checkpoint has `architectures: ["DFlash2DraftModel"]` and no
`speculators_model_type`. The one auto-conversion path that exists,
`maybe_convert_external_checkpoint`, routes *any* architecture matching
"DFlash" to `DFlashConverter`, which would silently drop the convolutions and
the candidate selector. Hence an explicit DFlash2 converter.

Reused from the parent, unchanged:

  * the `i + 1` target-layer remap. z-lab indexes `hidden_states[layer_id + 1]`
    (index 0 is the embedding output); speculators uses the layer id directly.
    `[1,13,25,37,49] -> [2,14,26,38,50]`, identical to DSpark's capture points,
    so the extraction server needs no reconfiguration between the two runs.
  * `block_size` out of `dflash_config`, and `speculative_tokens = block_size - 1`.
  * filling `embed_tokens` / `lm_head` / `verifier_lm_head` / `verifier_norm`
    from the verifier -- absent from the source by design.
  * `_remap_weights`, a no-op here: q/k/v are already unfused.

Added here:

  * a `DFlash2SpeculatorConfig` carrying `conv_kernel_size`, `conv_group_size`,
    `selector_rank`, `selector_top_k`, and -- with
    patches/speculators-dflash2-output-shaping.patch applied --
    `output_multiplier` and `final_logit_softcapping`. Those last two are why a
    stock conversion is not safe: vLLM applies both when serving
    (qwen3_dflash2.py builds `LogitsProcessor(scale=..., soft_cap=...)`), stock
    speculators implements neither, so a stock conversion trains at scale 1.0
    while serving at 0.196 -- a 5.1x logit mismatch nothing errors on.
  * `DFlash2DraftModel` for save and validation, so the convolution and
    selector tensors are claimed rather than reported as unexpected.

Usage (inside specd:dflash2, which has the patch baked in):

    python3 scripts/convert_dflash2.py \
      --input z-lab/Muse-Glimmer-30B-DFlash2 \
      --verifier meta-models/Muse-Glimmer-30B \
      --output /mnt/data/runs/dflash2-speculators
"""

import argparse
import sys
from pathlib import Path

import torch
from loguru import logger
from transformers import PretrainedConfig

from speculators.convert.dflash.converter import (
    _VERIFIER_FILLED_KEYS,
    DFlashConverter,
)
from speculators.models.dflash2.config import DFlash2SpeculatorConfig
from speculators.models.dflash2.core import DFlash2DraftModel

__all__ = ["DFlash2Converter"]


class DFlash2Converter(DFlashConverter):
    """`DFlashConverter` with DFlash2's config class, model class and knobs."""

    # Keys in the source `dflash_config` that have a matching field on
    # DFlash2SpeculatorConfig. The rest of that dict is consumed by the parent
    # (block_size, mask_token_id, target_layer_ids).
    dflash2_config_keys: tuple[str, ...] = (
        "conv_kernel_size",
        "conv_group_size",
        "selector_rank",
        "selector_top_k",
        "output_multiplier",
        "final_logit_softcapping",
    )

    def __init__(self, allow_missing_knobs: bool = False) -> None:
        self.allow_missing_knobs = allow_missing_knobs

    @staticmethod
    def _verifier_text_config(base_model: str) -> dict:
        """The verifier sub-config holding the language model's dimensions.

        meta-models/Muse-Glimmer-30B is a MuseGlimmerForConditionalGeneration:
        `num_hidden_layers` and `hidden_size` live under `text_config`, and the
        top level carries a *different* `out_hidden_size` (6144, the projector
        output) that must not be mistaken for the drafter's 6656. The parent
        converter reads both from the top level, which is why it raises
        `KeyError: 'num_hidden_layers'` here.
        """
        config_dict, _ = PretrainedConfig.get_config_dict(base_model)
        text = config_dict.get("text_config")
        return text if isinstance(text, dict) else config_dict

    def _build_config(
        self,
        source_config: dict,
        base_model: str,
        aux_hidden_state_layer_ids: list[int] | None,
    ) -> DFlash2SpeculatorConfig:
        text_config = self._verifier_text_config(base_model)

        # The parent's hidden-size guard compares against the *top-level*
        # hidden_size, which a composite verifier config does not have, so it
        # silently no-ops. Do it against the language model instead.
        source_hidden = source_config.get("hidden_size")
        target_hidden = text_config.get("hidden_size")
        if source_hidden != target_hidden:
            raise ValueError(
                f"Architecture mismatch: DFlash2 checkpoint has "
                f"hidden_size={source_hidden}, verifier '{base_model}' text "
                f"config has hidden_size={target_hidden}."
            )

        # The parent derives the layer ids from dflash_config.target_layer_ids,
        # but only on the branch that also reads the top-level
        # num_hidden_layers. Apply its rule verbatim against the resolved text
        # config, then hand the result down so the parent short-circuits.
        if aux_hidden_state_layer_ids is None:
            target_layer_ids = source_config.get("dflash_config", {}).get(
                "target_layer_ids"
            )
            if target_layer_ids is None:
                raise ValueError(
                    "Checkpoint config has no `dflash_config.target_layer_ids`; "
                    "pass `aux_hidden_state_layer_ids` explicitly."
                )
            # z-lab reads hidden_states[layer_id + 1] (index 0 is the embedding
            # output) while speculators uses the layer id directly. The last
            # verifier layer is excluded: it is always captured separately and
            # split off as verifier_last_hidden_states during training.
            num_verifier_layers = text_config["num_hidden_layers"]
            aux_hidden_state_layer_ids = [
                i + 1 for i in target_layer_ids if i + 1 != num_verifier_layers
            ]
            logger.info(
                f"target_layer_ids {target_layer_ids} -> "
                f"aux_hidden_state_layer_ids {aux_hidden_state_layer_ids} "
                f"(verifier has {num_verifier_layers} layers)"
            )

        # The parent still resolves block_size, the transformer config, the
        # verifier entry and the proposal method.
        base = super()._build_config(
            source_config, base_model, aux_hidden_state_layer_ids
        )
        base.speculators_config.algorithm = "dflash2"

        dflash = source_config.get("dflash_config", {})
        present = {
            k: dflash[k] for k in self.dflash2_config_keys if dflash.get(k) is not None
        }
        unsupported = [
            k for k in present if k not in DFlash2SpeculatorConfig.model_fields
        ]
        if unsupported and not self.allow_missing_knobs:
            raise ValueError(
                f"This speculators build has no DFlash2SpeculatorConfig field for "
                f"{unsupported}. They are output-shaping knobs vLLM applies at "
                "serving time, so converting without them trains at a different "
                "logit scale than the checkpoint serves at. Apply "
                "patches/speculators-dflash2-output-shaping.patch (baked into "
                "specd:dflash2), or pass --allow-missing-knobs to drop them "
                "deliberately on both sides."
            )
        for key in unsupported:
            logger.warning(f"Dropping unsupported output-shaping knob: {key}")
            del present[key]

        config = DFlash2SpeculatorConfig(
            transformer_layer_config=base.transformer_layer_config,
            draft_vocab_size=base.draft_vocab_size,
            block_size=base.block_size,
            aux_hidden_state_layer_ids=base.aux_hidden_state_layer_ids,
            mask_token_id=base.mask_token_id,
            sample_from_anchor=base.sample_from_anchor,
            speculators_config=base.speculators_config,
            **present,
        )
        logger.info(
            f"DFlash2 config: block_size={config.block_size} "
            f"aux_layers={config.aux_hidden_state_layer_ids} "
            f"sample_from_anchor={config.sample_from_anchor} "
            f"conv={config.conv_kernel_size}/{config.conv_group_size} "
            f"selector=rank{config.selector_rank}/top{config.selector_top_k} "
            f"output_multiplier={getattr(config, 'output_multiplier', 'UNSUPPORTED')} "
            f"softcap={getattr(config, 'final_logit_softcapping', 'UNSUPPORTED')}"
        )
        return config

    def _save(
        self,
        config: DFlash2SpeculatorConfig,
        weights: dict[str, torch.Tensor],
        output_path: str | Path,
    ) -> Path:
        # Mirrors DFlashConverter._save; it constructs DFlashDraftModel inline,
        # so the model class cannot be swapped without reimplementing it.
        model = DFlash2DraftModel(config=config)

        body = {k: v for k, v in weights.items() if k not in ("t2d", "d2t")}
        body = self._remap_weights(body, config, model)
        missing, unexpected = model.load_state_dict(body, strict=False)
        if unexpected:
            raise ValueError(
                "Unexpected keys in checkpoint -- the structure does not match "
                f"DFlash2DraftModel. Unexpected keys: {unexpected}"
            )
        critical_missing = [k for k in missing if k not in _VERIFIER_FILLED_KEYS]
        if critical_missing:
            raise ValueError(f"Draft weights missing after load: {critical_missing}")
        logger.info(f"Keys filled from the verifier at save time: {sorted(missing)}")

        model.load_verifier_weights()
        model.to(dtype=next(iter(body.values())).dtype)
        model.save_pretrained(str(output_path))
        return Path(output_path)

    def _validate(self, output_path: Path) -> None:
        logger.info("Validating converted DFlash2 checkpoint...")
        model = DFlash2DraftModel.from_pretrained(str(output_path))
        state = model.state_dict()
        # The parent checks three tensors; add the DFlash2-only ones, which are
        # exactly what a DFlash (not DFlash2) conversion would have silently lost.
        checked = (
            "fc.weight",
            "lm_head.weight",
            "embed_tokens.weight",
            "candidate_selector.predecessor_codebook",
            "candidate_selector.successor_codebook",
            "candidate_selector.hidden_projection.weight",
            "layers.0.attention_conv.base_kernel",
            "layers.0.attention_conv.kernel_projection.weight",
            "layers.0.mlp_conv.base_kernel",
            "layers.0.mlp_conv.kernel_projection.weight",
        )
        for name in checked:
            if name not in state:
                raise ValueError(f"Converted checkpoint is missing {name}")
            if torch.isnan(state[name]).any():
                raise ValueError(f"Converted checkpoint has NaN in {name}")
        logger.success(f"Validation succeeded ({len(state)} tensors)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a z-lab DFlash2 checkpoint to speculators format."
    )
    parser.add_argument("--input", default="z-lab/Muse-Glimmer-30B-DFlash2")
    parser.add_argument("--verifier", default="meta-models/Muse-Glimmer-30B")
    parser.add_argument("--output", default="/mnt/data/runs/dflash2-speculators")
    parser.add_argument(
        "--aux-hidden-state-layer-ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Override the layer ids. Leave unset: they are derived from "
            "dflash_config.target_layer_ids with the same i+1 remap the trainer "
            "and the extraction server assume."
        ),
    )
    parser.add_argument(
        "--allow-missing-knobs",
        action="store_true",
        help=(
            "Convert even if this speculators build has no field for "
            "output_multiplier / final_logit_softcapping. The result then trains "
            "AND serves at scale 1.0 with no soft cap -- self-consistent, but it "
            "discards what the warm start was fit for."
        ),
    )
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    DFlash2Converter(allow_missing_knobs=args.allow_missing_knobs).convert(
        input_path=args.input,
        output_path=args.output,
        base_model=args.verifier,
        validate=not args.no_validate,
        aux_hidden_state_layer_ids=args.aux_hidden_state_layer_ids,
        cache_dir=args.cache_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
