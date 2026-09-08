#!/usr/bin/env python3
"""Repair AsyncLLM.profiler in the disposable container's vLLM source.

Upstream assigns `self.profiler` only inside the `profiler == "torch"` branch of
`AsyncLLM.__init__`, yet `start_profile`/`stop_profile` guard on
`self.profiler is not None` and `__init__` already accepts a `profiler`
argument. Any other backend -- `cuda`, which is what an Nsight
`--capture-range=cudaProfilerApi` session needs -- therefore fails with
`'AsyncLLM' object has no attribute 'profiler'` before the request ever reaches
the engine core, where CudaProfilerWrapper would call cudaProfilerStart.

Assigning the constructor argument first restores the documented behaviour and
leaves the torch branch free to overwrite it. Idempotent; fails loudly if the
source no longer looks the way this fix assumes.
"""

import importlib.util
from pathlib import Path

RELATIVE = "v1/engine/async_llm.py"
ANCHOR = """        if (
            vllm_config.profiler_config.profiler == "torch"
            and not vllm_config.profiler_config.ignore_frontend
        ):
"""
FIX = """        # Upstream sets self.profiler only for the torch backend, so every other
        # backend raises AttributeError in start_profile. Honour the argument.
        self.profiler = profiler
"""


def main() -> None:
    root = Path(importlib.util.find_spec("vllm").origin).parent
    path = root / RELATIVE
    text = path.read_text()
    if FIX in text:
        print(f"Profiler attribute fix already present: {RELATIVE}", flush=True)
        return
    if "self.profiler = TorchProfilerWrapper(" not in text:
        raise RuntimeError(f"{path}: torch profiler assignment is gone; re-check this fix.")
    if text.count(ANCHOR) != 1:
        raise RuntimeError(f"{path}: expected exactly one frontend-profiler guard; "
                           f"found {text.count(ANCHOR)}.")
    new_text = text.replace(ANCHOR, FIX + ANCHOR)
    compile(new_text, str(path), "exec")
    path.write_text(new_text)
    print(f"Profiler attribute fixed: {RELATIVE}", flush=True)


if __name__ == "__main__":
    main()
