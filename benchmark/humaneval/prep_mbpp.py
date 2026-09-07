#!/usr/bin/env python3
"""Build an MBPP prompt file in the same shape gen.py/metrics.py already read.

RedHatAI/speculator_benchmarks ships HumanEval but not MBPP, so MBPP comes from
the canonical dataset and is normalised here rather than special-cased in the
driver. We use the 500-problem test split (task_id 11-510), which is what the
published MBPP numbers are computed over -- not the full 974.

Prompt shape is the standard MBPP one: the description followed by the asserts
the solution has to satisfy. Without the asserts the function name and
signature are unspecified and pass@1 measures guesswork rather than coding.

entry_point is recovered from the first assert, because MBPP does not record
it. Builtin names are skipped when there is an alternative, but kept as a
fallback -- MBPP/126 genuinely defines a function called `sum`. `test` is emitted as a self-contained check() so the scorer treats MBPP and
HumanEval identically.
"""
import json, re, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/bench/mbpp_test.jsonl"

from datasets import load_dataset
ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
print("rows:", len(ds), "cols:", ds.column_names)

CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")


def entry_point(tests):
    """First identifier that is called inside an assert, skipping wrappers."""
    skip = {"assert", "abs", "len", "set", "sorted", "round", "str", "int",
            "float", "list", "tuple", "dict", "bool", "math", "isclose",
            "all", "any", "sum", "max", "min", "type", "repr"}
    first = None
    for t in tests:
        body = t.split("assert", 1)[-1]
        for m in CALL.finditer(body):
            first = first or m.group(1)
            if m.group(1) not in skip:
                return m.group(1)
    # MBPP/126 really does name its function `sum`; a shadowed builtin is a
    # worse answer than no answer only if we never look past the skip list
    return first


n_skip = 0
with open(OUT, "w") as fh:
    for r in ds:
        tests = list(r["test_list"] or [])
        ep = entry_point(tests)
        if not ep or not tests:
            n_skip += 1
            continue
        setup = (r.get("test_setup_code") or "").strip()
        # wrapped as check() so the scorer's HumanEval path works unchanged;
        # candidate is bound but unused -- MBPP asserts call the name directly
        body = "\n".join("    " + l for l in
                         ([setup] if setup else []) + tests)
        test = "def check(candidate):\n" + body + "\n"
        prompt = (r["text"].strip() + "\nYour code should pass these tests:\n\n"
                  + "\n".join(tests) + "\n")
        fh.write(json.dumps({"task_id": f"MBPP/{r['task_id']}", "prompt": prompt,
                             "entry_point": ep, "test": test,
                             "canonical_solution": r.get("code", "")}) + "\n")
print(f"wrote {OUT}, skipped {n_skip}")
