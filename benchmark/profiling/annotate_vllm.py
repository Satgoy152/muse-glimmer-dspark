#!/usr/bin/env python3
"""Add launch annotations to disposable container source, never model weights.

Fail on a changed source layout. NVTX inside captured graphs only runs during
capture; GPU graph replay is attributed by the enclosing graph-launch range.
"""
import ast
import importlib.util
from pathlib import Path

TARGETS = {
    "v1/worker/gpu/model_runner.py": {
        "GPUModelRunner": {
            "execute_model": "target_execute",
            "sample_tokens": "sample_and_draft",
            "sample": "target_logits_and_sampling",
            "prepare_inputs": "prepare_inputs",
            "prepare_attn": "prepare_attn",
            "postprocess_sampled": "postprocess_sampled",
        },
    },
    "v1/worker/gpu/spec_decode/dflash/speculator.py": {
        "DFlashSpeculator": {
            "propose": "draft_propose",
            "_run_model": "draft_backbone",
            "_build_draft_attn_metadata": "draft_attn_metadata",
        },
    },
    "v1/worker/gpu/cudagraph_utils.py": {
        "CudaGraphManager": {"run_fullgraph": "graph"},
        "ModelCudaGraphManager": {"run_fullgraph": "target_graph"},
    },
    "model_executor/models/qwen3_dflash.py": {
        "DFlashQwen3ForCausalLM": {
            "precompute_and_store_context_kv": "draft_context_kv",
            "combine_hidden_states": "draft_feature_projection",
        },
    },
    "model_executor/models/qwen3_dflash2.py": {
        "DFlash2Qwen3ForCausalLM": {"compute_candidates": "draft_candidates"},
    },
    "model_executor/models/muse_glimmer.py": {
        "MuseGlimmerForCausalLM": {"compute_logits": "target_lm_head"},
    },
    "v1/worker/gpu/spec_decode/rejection_sampler.py": {
        "RejectionSampler": {"__call__": "target_rejection"},
    },
}


def main():
    root = Path(importlib.util.find_spec("vllm").origin).parent
    plans = []
    for relative, classes in TARGETS.items():
        path = root / relative
        text = path.read_text()
        if "from profile_nvtx import range_call" in text:
            raise RuntimeError(f"Already annotated: {path}")
        tree = ast.parse(text)
        edits = []
        for class_name, methods in classes.items():
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                       and n.name == class_name)
            for method, label in methods.items():
                fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                          and n.name == method)
                line = min([fn.lineno] + [d.lineno for d in fn.decorator_list])
                edits.append((line - 1, f'    @range_call("profile.{label}")\n'))
        # Keep module docstring and all future imports before the new import.
        insertion = 0
        for node in tree.body:
            if (isinstance(node, ast.ImportFrom) and node.module == "__future__") or (
                isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str) and node is tree.body[0]
            ):
                insertion = node.end_lineno
            else:
                break
        edits.append((insertion, "from profile_nvtx import range_call\n"))
        lines = text.splitlines(keepends=True)
        for position, addition in sorted(edits, reverse=True):
            lines.insert(position, addition)
        new_text = "".join(lines)
        compile(new_text, str(path), "exec")
        plans.append((path, new_text))
    for path, text in plans:
        path.write_text(text)
        print(f"NVTX annotated: {path.relative_to(root)}", flush=True)


if __name__ == "__main__":
    main()
