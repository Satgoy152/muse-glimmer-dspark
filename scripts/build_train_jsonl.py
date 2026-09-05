"""Turn the published traces into the JSONL `speculators prepare-data` reads.
    uv run python scripts/build_train_jsonl.py --out data/train/conversations.jsonl
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

DATASET = "Satgoy152/Muse-Glimmer-SWE-Gym-2k"
# Matches export_traces.STRENGTH_SENTENCE, and the template's case-insensitive test.
STRENGTH_RE = re.compile(r"reasoning strength:", re.IGNORECASE)


def strip_nulls(message: dict) -> dict:
    """Drop the null struct fields parquet materialises on every message."""
    return {k: v for k, v in message.items() if v is not None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--config", default="train")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-trajectories", type=int,
                    help="sample this many trajectories (stratified by strength)")
    ap.add_argument("--seed", type=int, default=20260830)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, name=args.config, split=args.split)
    print(f"loaded {len(ds):,} rows, columns {ds.column_names}")
    if "messages" not in ds.column_names:
        raise SystemExit(f"expected a `messages` column, got {ds.column_names}")

    if args.max_trajectories and args.max_trajectories < len(ds):
        # Stratify by strength: the strengths are unbalanced already and an
        # unstratified sample would move the mix as well as the size.
        import random

        rng = random.Random(args.seed)
        by_strength: dict[str, list[int]] = {}
        for i, s in enumerate(ds["reasoning_strength"]):
            by_strength.setdefault(s, []).append(i)
        keep: list[int] = []
        for strength, idxs in sorted(by_strength.items()):
            n = round(args.max_trajectories * len(idxs) / len(ds))
            keep.extend(rng.sample(idxs, min(n, len(idxs))))
        ds = ds.select(sorted(keep))
        print(f"sampled {len(ds):,} trajectories (seed {args.seed})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    strengths: Counter[str] = Counter()
    turns: Counter[str] = Counter()
    missing, duplicated, no_system = [], [], []

    with out.open("w") as fh:
        for row in ds:
            messages = [strip_nulls(m) for m in row["messages"]]
            system = [m for m in messages if m.get("role") == "system"]
            hits = sum(len(STRENGTH_RE.findall(m.get("content") or "")) for m in system)
            if not system:
                no_system.append(row["id"])
            elif hits == 0:
                missing.append(row["id"])
            elif hits > 1:
                duplicated.append(row["id"])

            strengths[row["reasoning_strength"]] += 1
            for m in messages:
                turns[m.get("role", "?")] += 1
            # `tools` stays a JSON string: _parse_conv_tools json.loads it per row.
            fh.write(json.dumps({
                "id": row["id"],
                "conversations": messages,
                "tools": row["tools"],
                "reasoning_strength": row["reasoning_strength"],
            }) + "\n")

    print(f"wrote {out} ({out.stat().st_size/1e6:.0f} MB)")
    print(f"strengths: {dict(sorted(strengths.items()))}")
    print(f"turns by role: {dict(sorted(turns.items()))}")

    problems = [("no system message", no_system),
                ("no reasoning-strength sentence", missing),
                ("reasoning-strength sentence more than once", duplicated)]
    for label, ids in problems:
        if ids:
            print(f"FAIL {len(ids)} trajectories with {label}: {ids[:5]}")
    if any(ids for _, ids in problems):
        raise SystemExit(1)
    print(f"OK reasoning-strength sentence appears exactly once in all {len(ds):,}")


if __name__ == "__main__":
    main()
