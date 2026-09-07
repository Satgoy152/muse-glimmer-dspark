#!/usr/bin/env python3
"""Teach speculators (and optionally vLLM) DFlash2's two output-shaping knobs.

z-lab/Muse-Glimmer-30B-DFlash2 carries `output_multiplier` (0.19611613513818404,
a 5.1x logit scale) and `final_logit_softcapping` (20.0) in its `dflash_config`.
vLLM applies both when it serves the drafter; stock speculators implements
neither. Converting the checkpoint without them means training at scale 1.0 and
serving at 0.196 -- a 5.1x logit mismatch that nothing in either stack errors on.

Measured on this stack at step 0, same seed and batch:

    train/loss           0.406 patched   vs   2.155 unpatched
    train/selector_loss  0.287 patched   vs   1.857 unpatched

The selector is what breaks: it scores `unary_scores + transition_scores`, and
at scale 1.0 the unary term is 5.1x larger than the one its codebooks were fit
against, so a converged selector reads as badly miscalibrated.

Applied as string-anchored edits rather than a context diff so it survives
speculators point releases and re-running (idempotent -- a second run is a
no-op). `--check` reports status without writing.

    python3 patches/apply_dflash2_output_shaping.py            # speculators only
    python3 patches/apply_dflash2_output_shaping.py --vllm     # + serving side
    python3 patches/apply_dflash2_output_shaping.py --check

The vLLM half is only needed to SERVE or EVAL a converted checkpoint; training
does not touch it. docker/serve_patched.sh applies the same change with sed at
container start, which is the usual path -- use `--vllm` when baking an image.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CONFIG_ADDITION = '''    output_multiplier: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Multiplier applied to the draft LM head logits before the candidate "
            "selector and the loss. z-lab DFlash2 checkpoints carry this in "
            "`dflash_config.output_multiplier` and vLLM applies it at serving "
            "time (LogitsProcessor(scale=...)), so training must apply the same "
            "factor or the fine-tune re-learns the output scale. Must be "
            "positive: vLLM's local top-k reduction rejects non-positive scales."
        ),
    )
    final_logit_softcapping: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "If set, the scaled logits are squashed as `tanh(x / cap) * cap` "
            "before the candidate selector and the loss, matching vLLM's "
            "LogitsProcessor(soft_cap=...). Applied *after* output_multiplier, "
            "which is the order LogitsProcessor.get_top_k_tokens uses -- the "
            "path DFlash2 serving actually takes. None disables it."
        ),
    )
'''

CORE_HELPER = '''    def _shape_output_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the checkpoint's output scale and soft cap to the unary logits.

        z-lab DFlash2 checkpoints are trained with a logit multiplier and a tanh
        soft cap (``dflash_config.output_multiplier`` /
        ``final_logit_softcapping``). vLLM applies both when serving:
        ``DFlash2Qwen3ForCausalLM`` builds ``LogitsProcessor(scale=..., soft_cap=...)``
        and ``compute_candidates`` calls ``get_top_k_tokens``, which does

            values = values * scale
            values = tanh(values / soft_cap) * soft_cap

        i.e. **scale first, then cap**. (The generic ``forward`` and
        ``get_top_tokens`` paths on that class apply them the other way round;
        DFlash2 does not take either.) Training has to match, or a warm-started
        drafter spends its capacity re-learning the output scale and the served
        checkpoint disagrees with the trained one by 5.1x on the logits.

        Both maps are strictly increasing, so the unary top-k -- the serving
        candidate set -- is unchanged. What does change is the *value* the
        candidate selector adds its transition score to
        (``CandidateSelector.score_candidates``: ``unary_scores + transition_scores``)
        and the scale the unary loss sees.

        Applied in the logits' own dtype rather than vLLM's float32: the tensor
        is ``[1, num_anchors * block_size, vocab_size]`` and an fp32 copy at
        max_anchors 1024 / block_size 16 / vocab 202,048 is 13 GB.
        """
        scale = getattr(self.config, "output_multiplier", 1.0)
        if scale != 1.0:
            logits = logits * scale
        soft_cap = getattr(self.config, "final_logit_softcapping", None)
        if soft_cap:
            logits = torch.tanh(logits / soft_cap) * soft_cap
        return logits

'''

CORE_ANCHOR = """        candidate_ids = unary_logits.topk(self.candidate_selector.top_k, dim=-1).indices
        # shape: [1, num_anchors*block_size, top_k]
"""

CORE_CALL = """        # Select on the raw head output, then shape -- the order
        # LogitsProcessor.get_top_k_tokens uses at serving time. Scale and tanh
        # cap are both strictly increasing, so the two orders agree in exact
        # arithmetic, but tanh saturation ties neighbouring logits in bfloat16
        # and would reorder the tail of the candidate set. Everything downstream
        # (the selector's additive transition scores, the unary loss) sees the
        # shaped values, which is what serving scores.
        unary_logits = self._shape_output_logits(unary_logits)
"""

VLLM_ANCHOR = """    for key in (
        "conv_kernel_size",
        "conv_group_size",
        "selector_rank",
        "selector_top_k",
    ):
"""

VLLM_REPLACEMENT = """    # output_multiplier / final_logit_softcapping are read straight back out of
    # dflash_config by DFlash2Qwen3ForCausalLM, which builds
    # LogitsProcessor(scale=..., soft_cap=...) from them. Without them here a
    # converted checkpoint serves at scale=1.0 with no cap, while the z-lab
    # weights it came from were fit for a 0.196 scale -- a 5.1x logit mismatch
    # that nothing in either stack errors on.
    for key in (
        "conv_kernel_size",
        "conv_group_size",
        "selector_rank",
        "selector_top_k",
        "output_multiplier",
        "final_logit_softcapping",
    ):
"""


def site_packages() -> Path:
    import speculators  # noqa: PLC0415

    return Path(speculators.__file__).resolve().parent.parent


def edit(path: Path, marker: str, apply, check_only: bool) -> str:
    if not path.is_file():
        return f"MISSING  {path}"
    text = path.read_text()
    if marker in text:
        return f"already  {path.name}"
    if check_only:
        return f"NEEDED   {path.name}"
    new = apply(text)
    if new == text:
        return f"FAILED   {path.name}: anchor not found (upstream changed?)"
    path.write_text(new)
    return f"patched  {path.name}"


def patch_speculators(root: Path, check_only: bool) -> list[str]:
    out = []
    cfg = root / "speculators/models/dflash2/config.py"

    def do_cfg(text: str) -> str:
        # The knobs go at the end of DFlash2SpeculatorConfig's field block.
        anchor = '        description="Number of unary candidates reranked during inference.",\n    )\n'
        if anchor not in text:
            return text
        return text.replace(anchor, anchor + CONFIG_ADDITION, 1)

    out.append(edit(cfg, "output_multiplier", do_cfg, check_only))

    core = root / "speculators/models/dflash2/core.py"

    def do_core(text: str) -> str:
        if "    def _predecessor_ids(\n" not in text or CORE_ANCHOR not in text:
            return text
        text = text.replace(
            "    def _predecessor_ids(\n", CORE_HELPER + "    def _predecessor_ids(\n", 1
        )
        return text.replace(CORE_ANCHOR, CORE_ANCHOR + CORE_CALL, 1)

    out.append(edit(core, "_shape_output_logits", do_core, check_only))
    return out


def patch_vllm(check_only: bool) -> list[str]:
    try:
        import vllm  # noqa: PLC0415
    except ImportError:
        return ["skipped  vllm not importable"]
    algos = (
        Path(vllm.__file__).resolve().parent
        / "transformers_utils/configs/speculators/algos.py"
    )
    return [
        edit(
            algos,
            '"output_multiplier",',
            lambda t: t.replace(VLLM_ANCHOR, VLLM_REPLACEMENT, 1),
            check_only,
        )
    ]


def verify() -> None:
    from speculators.models.dflash2.config import DFlash2SpeculatorConfig  # noqa: PLC0415
    from speculators.models.dflash2.core import DFlash2DraftModel  # noqa: PLC0415

    fields = DFlash2SpeculatorConfig.model_fields
    for name in ("output_multiplier", "final_logit_softcapping"):
        assert name in fields, f"{name} missing from DFlash2SpeculatorConfig"
    assert hasattr(DFlash2DraftModel, "_shape_output_logits")
    import inspect  # noqa: PLC0415

    src = inspect.getsource(DFlash2DraftModel.forward)
    assert "_shape_output_logits(unary_logits)" in src, src
    # The shaping must come AFTER the top-k, matching get_top_k_tokens.
    assert src.index(".topk(") < src.index("_shape_output_logits(unary_logits)"), (
        "shaping must follow candidate selection; vLLM tops-k the raw logits"
    )
    print("verify: dflash2 output-shaping patch OK")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vllm", action="store_true", help="also patch vLLM's algos.py")
    ap.add_argument("--check", action="store_true", help="report status, write nothing")
    ap.add_argument("--root", default=None, help="site-packages root (default: auto)")
    args = ap.parse_args()

    root = Path(args.root) if args.root else site_packages()
    results = patch_speculators(root, args.check)
    if args.vllm:
        results += patch_vllm(args.check)
    for line in results:
        print(" ", line)
    if any(r.startswith(("FAILED", "MISSING")) for r in results):
        return 1
    # verify() imports speculators, so it only says anything about the tree that
    # `import speculators` actually resolves to. With an explicit --root (e.g. a
    # second checkout kept in sync) that is a different tree, so skip it there
    # rather than reporting on the wrong files.
    if not args.check and args.root is None:
        verify()
    return 0


if __name__ == "__main__":
    sys.exit(main())
