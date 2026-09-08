import re, pathlib
F = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/configs/speculators/algos.py")
s = F.read_text()
if "output_multiplier" in s:
    print("algos patch: already applied"); raise SystemExit(0)
old = """    for key in (
        "conv_kernel_size",
        "conv_group_size",
        "selector_rank",
        "selector_top_k",
    ):"""
new = """    for key in (
        "conv_kernel_size",
        "conv_group_size",
        "selector_rank",
        "selector_top_k",
        "output_multiplier",
        "final_logit_softcapping",
        "block_size",
    ):"""
assert old in s, "anchor not found"
F.write_text(s.replace(old, new, 1))
print("algos patch OK: forwarding output_multiplier/final_logit_softcapping/block_size")
