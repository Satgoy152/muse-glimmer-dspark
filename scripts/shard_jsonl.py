"""Split the conversations JSONL into shards so preprocessing is resumable.

`speculators prepare-data` is all-or-nothing: it skips a run whose output
directory already holds `*.arrow`, but it has no partial progress. On a
preemptible node a multi-hour render that dies at 90% restarts from zero.
Running it per shard bounds that loss to one shard.

    uv run python scripts/shard_jsonl.py --in conversations.jsonl \
        --out-dir /mnt/data/runs/shards --shards 16
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shards", type=int, default=16)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    handles = [
        (out_dir / f"shard-{i:03d}.jsonl").open("w") for i in range(args.shards)
    ]
    counts = [0] * args.shards
    try:
        # Round-robin rather than contiguous blocks: trajectory length and
        # reasoning strength both correlate with position in the file, and a
        # shard that is short or all-xhigh would skew whatever partial run
        # survives a preemption.
        for i, line in enumerate(Path(args.src).open()):
            shard = i % args.shards
            handles[shard].write(line)
            counts[shard] += 1
    finally:
        for handle in handles:
            handle.close()

    total = sum(counts)
    print(f"{total:,} rows -> {args.shards} shards in {out_dir}")
    print(f"rows per shard: min {min(counts)}, max {max(counts)}")
    manifest = out_dir / "shards.json"
    manifest.write_text(json.dumps(
        {f"shard-{i:03d}": counts[i] for i in range(args.shards)}, indent=2))
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
