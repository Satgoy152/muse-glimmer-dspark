"""Check a `speculators prepare-data` output before spending GPU hours on it.

Both of this project's silent failures show up here and nowhere else.

seq_length: a turn whose context alone fills the window is skipped, not
truncated, so an under-sized window yields a small dataset rather than an
error. Compare the row count against `scripts/plan_seq_length.py` for the
--seq-length that was actually used.

Reasoning strength: it survives only as a sentence in the system message. If
preprocessing rewrote or dropped system messages, the rendered rows come back
at the chat template's default of "high" instead of the recorded mix, and
nothing downstream complains. This decodes the rows and counts the levels.

    uv run python scripts/verify_prepared.py --data ./output \
        --model meta-models/Muse-Glimmer-30B --sample 400
"""

import argparse
import re
from collections import Counter

STRENGTH_RE = re.compile(r"reasoning strength:\s*([a-z]+)", re.IGNORECASE)
# The mix in Satgoy152/Muse-Glimmer-SWE-Gym-2k, by trajectory.
EXPECTED = {"high": 594, "low": 593, "medium": 596, "xhigh": 198}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="prepare-data --output directory")
    ap.add_argument("--model", required=True, help="target model, for the tokenizer")
    ap.add_argument("--sample", type=int, default=400, help="rows to decode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    import numpy as np
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    ds = load_from_disk(args.data)
    ds = ds.with_format(None)
    print(f"{len(ds):,} rows, columns {ds.column_names}")

    seq_len = np.asarray(ds["seq_len"])
    print(f"seq_len: mean {seq_len.mean():,.0f}  median {np.median(seq_len):,.0f}  "
          f"min {seq_len.min():,}  max {seq_len.max():,}")

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), size=min(args.sample, len(ds)), replace=False)

    tok = AutoTokenizer.from_pretrained(args.model,
                                        trust_remote_code=args.trust_remote_code)
    levels: Counter[str] = Counter()
    occurrences: Counter[int] = Counter()
    unsupervised = 0
    supervised = []

    for i in idx:
        row = ds[int(i)]
        mask = np.asarray(row["loss_mask"])
        n_sup = int(mask.sum())
        supervised.append(n_sup)
        unsupervised += n_sup == 0
        found = STRENGTH_RE.findall(tok.decode(row["input_ids"]))
        occurrences[len(found)] += 1
        for level in found:
            levels[level.lower()] += 1

    sup = np.asarray(supervised)
    print(f"supervised tokens per row (n={len(sup)}): mean {sup.mean():,.0f}  "
          f"median {np.median(sup):,.0f}  min {sup.min():,}  max {sup.max():,}")
    if unsupervised:
        print(f"FAIL {unsupervised} sampled rows have an all-zero loss mask")

    print(f"reasoning-strength sentences per row: {dict(sorted(occurrences.items()))}")
    total = sum(levels.values()) or 1
    print("rendered strength mix: " + "  ".join(
        f"{k} {v} ({100*v/total:.0f}%)" for k, v in sorted(levels.items())))
    expected_total = sum(EXPECTED.values())
    print("source mix by trajectory: " + "  ".join(
        f"{k} ({100*v/expected_total:.0f}%)" for k, v in sorted(EXPECTED.items())))

    problems = []
    if set(occurrences) != {1}:
        problems.append("some rows do not carry the sentence exactly once -- "
                        "preprocessing altered the system message")
    if len(levels) < 2:
        problems.append(f"only {sorted(levels)} present -- the recorded mix was "
                        "replaced by the chat template default")
    if unsupervised:
        problems.append("rows with no supervised tokens survived filtering")
    for problem in problems:
        print(f"FAIL {problem}")
    if problems:
        raise SystemExit(1)
    print("OK strength sentence intact, mix preserved, every sampled row supervised")


if __name__ == "__main__":
    main()
