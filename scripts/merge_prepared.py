"""Concatenate per-shard `prepare-data` outputs into one training dataset.

`speculators train --data-path` wants a single directory holding the Arrow
dataset and `token_freq.pt`. Sharding preprocessing for preemption resilience
produces one directory per shard, so they are merged here rather than by
re-running preprocessing over the whole corpus.

Vocabulary mapping files (`d2t.npy` / `t2d.npy`) are deliberately absent: the
draft uses the full 202,048-token verifier vocabulary, so `draft_vocab_size`
equals the verifier's and speculators skips the mapping.

    uv run python scripts/merge_prepared.py \
        --shard-dir /mnt/data/runs/prepared --out /mnt/data/runs/data
"""

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-samples", type=int,
                    help="cap rows after shuffling, matching prepare-data")
    ap.add_argument("--seed", type=int, default=20260830)
    args = ap.parse_args()

    from datasets import concatenate_datasets, load_from_disk
    from speculators.train.vocab_mapping import save_token_frequency_distribution

    shard_dirs = sorted(
        d for d in Path(args.shard_dir).iterdir()
        if d.is_dir() and any(d.glob("*.arrow"))
    )
    if not shard_dirs:
        raise SystemExit(f"no prepared shards under {args.shard_dir}")

    parts = []
    for d in shard_dirs:
        ds = load_from_disk(str(d))
        print(f"{d.name}: {len(ds):,} rows")
        parts.append(ds)

    merged = concatenate_datasets(parts)
    print(f"merged: {len(merged):,} rows from {len(parts)} shards")
    merged = merged.shuffle(seed=args.seed)
    if args.max_samples and len(merged) > args.max_samples:
        merged = merged.select(range(args.max_samples))
        print(f"capped to {len(merged):,} rows")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_to_disk(str(out))
    # The trainer reads this from data_path; it is per-dataset, so it has to be
    # recomputed on the merged set rather than taken from any one shard.
    save_token_frequency_distribution(dataset=merged, output_path=out / "token_freq.pt")
    print(f"wrote {out} and {out / 'token_freq.pt'}")


if __name__ == "__main__":
    main()
