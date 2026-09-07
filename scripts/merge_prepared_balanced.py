"""Concatenate per-shard `prepare-data` outputs, balanced by reasoning strength.

`merge_prepared.py` shuffles and caps, which preserves whatever level mix the
corpus happens to have -- for SWE-Gym that is roughly 31% medium / 31% high /
27% low / 10% xhigh, because only 198 of the 1,981 trajectories are xhigh.
This variant instead draws an equal quota per level, so the drafter sees each
reasoning strength about equally often.

Reasoning strength is read back out of the rendered system message rather than
from a column: `export_traces.py` bakes `Reasoning strength: <level>.` into the
system text (see docs/train.md, "Reasoning strength lives in the system
message"), and that string is what the chat template actually honoured. Reading
it here means the balance is measured on what was rendered, not on metadata
that could have drifted from it.

A level short of its quota is taken in full and its shortfall redistributed
across the levels that still have rows, so the total row budget is always met.

`--weights` replaces the equal quota with an explicit target mix. There the
redistribution behaviour is deliberately inverted: if a level cannot meet its
share, the *budget shrinks* to hold the ratio rather than the ratio drifting to
hold the budget. Asking for 35% xhigh and silently getting 15% is the failure
mode this script exists to prevent; `--allow-mix-drift` opts back into
redistribution when the row count matters more than the mix.

    # equal quota across levels
    python3 scripts/merge_prepared_balanced.py \
        --shard-dir /mnt/data/runs/prepared-48k \
        --out /mnt/data/runs/data-48k --max-samples 30000

    # explicit mix
    python3 scripts/merge_prepared_balanced.py \
        --shard-dir /mnt/data/runs/prepared-48k \
        --out /mnt/data/runs/data-48k --max-samples 25000 \
        --weights low=0.10,medium=0.25,high=0.30,xhigh=0.35
"""

import argparse
import collections
import re
from pathlib import Path

LEVEL_RE = re.compile(r"reasoning strength:\s*(\w+)", re.I)


def row_level(messages) -> str:
    for msg in messages:
        if msg.get("role") == "system":
            match = LEVEL_RE.search(str(msg.get("content", "")))
            return match.group(1).lower() if match else "unknown"
    return "unknown"


def allocate(available: dict[str, int], budget: int) -> dict[str, int]:
    """Equal quota per level, redistributing any level's shortfall to the rest.

    Repeats because satisfying one shortfall can push another level over its
    own supply; it converges since each pass either fills the budget or
    permanently exhausts at least one level.
    """
    quota = {level: 0 for level in available}
    remaining_levels = set(available)
    remaining_budget = budget
    while remaining_levels and remaining_budget > 0:
        share = remaining_budget // len(remaining_levels)
        if share == 0:
            # Fewer rows left than levels: hand them out one at a time.
            for level in sorted(remaining_levels):
                if remaining_budget == 0:
                    break
                if quota[level] < available[level]:
                    quota[level] += 1
                    remaining_budget -= 1
            break
        progressed = False
        for level in sorted(remaining_levels):
            take = min(share, available[level] - quota[level])
            quota[level] += take
            remaining_budget -= take
            if take:
                progressed = True
            if quota[level] >= available[level]:
                remaining_levels.discard(level)
        if not progressed:
            break
    return quota


def parse_weights(spec: str) -> dict[str, float]:
    """`low=0.10,medium=0.25,high=0.30,xhigh=0.35` -> dict, normalised."""
    weights: dict[str, float] = {}
    for part in spec.split(","):
        if not part.strip():
            continue
        level, _, value = part.partition("=")
        if not _:
            raise SystemExit(f"--weights: expected level=value, got {part!r}")
        level = level.strip().lower()
        if level in weights:
            raise SystemExit(f"--weights: {level} given twice")
        try:
            weights[level] = float(value)
        except ValueError:
            raise SystemExit(f"--weights: {value!r} is not a number") from None
        if weights[level] <= 0:
            raise SystemExit(f"--weights: {level} must be positive")
    if not weights:
        raise SystemExit("--weights: no levels given")
    total = sum(weights.values())
    # Accept 10,25,30,35 as readily as 0.10,0.25,0.30,0.35.
    return {level: w / total for level, w in weights.items()}


def allocate_weighted(
    available: dict[str, int],
    budget: int,
    weights: dict[str, float],
    allow_drift: bool,
) -> dict[str, int]:
    """Draw `budget` rows at the requested mix.

    Levels absent from `weights` are excluded entirely -- an explicit mix is a
    statement about the whole set, not a floor on some of it.

    When a level cannot supply its share, holding both the ratio and the budget
    is impossible. The default holds the ratio and shrinks the budget, because a
    mix that silently drifts is worse than a smaller set: the caller asked for
    35% xhigh for a reason. `allow_drift` restores the equal-quota script's
    behaviour of redistributing the shortfall to keep the row count.
    """
    missing = [level for level in weights if level not in available]
    if missing:
        raise SystemExit(
            f"--weights names levels with no rows in the shards: {missing}. "
            f"Available: {sorted(available)}"
        )
    dropped = {level: n for level, n in available.items() if level not in weights}
    if dropped:
        print(f"\nexcluded by --weights: {dropped}")

    # The largest budget every level can still cover at its requested share.
    feasible = int(min(available[level] / w for level, w in weights.items()))
    if budget <= feasible:
        quota = {level: int(budget * w) for level, w in weights.items()}
        # Integer truncation loses a few rows; hand them to the level furthest
        # below its exact share so the realised mix stays closest to target.
        while sum(quota.values()) < budget:
            level = min(
                weights, key=lambda x: quota[x] / weights[x] if weights[x] else 0
            )
            if quota[level] >= available[level]:
                break
            quota[level] += 1
        return quota

    binder = min(weights, key=lambda level: available[level] / weights[level])
    print(
        f"\n!! {budget:,} rows at this mix needs "
        f"{int(budget * weights[binder]):,} {binder} rows; only "
        f"{available[binder]:,} exist."
    )
    if not allow_drift:
        print(
            f"   Holding the mix and shrinking the budget to {feasible:,} rows.\n"
            f"   Pass --allow-mix-drift to keep {budget:,} rows at a drifted mix,\n"
            f"   or render more shards to lift the {binder} supply."
        )
        return allocate_weighted(available, feasible, weights, allow_drift=False)

    print("   --allow-mix-drift set: keeping the row count, mix will drift.")
    quota = {level: min(int(budget * w), available[level])
             for level, w in weights.items()}
    remaining = budget - sum(quota.values())
    while remaining > 0:
        room = {level: available[level] - quota[level] for level in weights}
        if not any(room.values()):
            break
        share_total = sum(weights[level] for level in weights if room[level])
        progressed = False
        for level in sorted(weights):
            if not room[level] or remaining <= 0:
                continue
            take = min(room[level], max(1, int(remaining * weights[level] / share_total)))
            quota[level] += take
            remaining -= take
            progressed = True
        if not progressed:
            break
    return quota


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-samples", type=int, required=True,
                    help="total row budget; split evenly across levels unless "
                         "--weights is given")
    ap.add_argument("--weights", default=None,
                    help="explicit mix, e.g. "
                         "'low=0.10,medium=0.25,high=0.30,xhigh=0.35'. "
                         "Values are normalised, so percentages work too. "
                         "Levels not named are excluded.")
    ap.add_argument("--allow-mix-drift", action="store_true",
                    help="with --weights, keep the row budget and let the mix "
                         "drift when a level runs short, instead of shrinking "
                         "the budget to hold the mix.")
    ap.add_argument("--seed", type=int, default=20260830)
    args = ap.parse_args()

    weights = parse_weights(args.weights) if args.weights else None

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

    # Shuffle before slicing so each level's quota is a uniform sample over the
    # whole context range, not the leading rows of the earliest shards.
    merged = merged.shuffle(seed=args.seed)

    levels = [row_level(m) for m in merged["messages"]]
    by_level: dict[str, list[int]] = collections.defaultdict(list)
    for idx, level in enumerate(levels):
        by_level[level].append(idx)

    available = {level: len(idx) for level, idx in by_level.items()}
    print("\navailable by level:")
    for level, count in sorted(available.items(), key=lambda kv: -kv[1]):
        print(f"  {level:8s} {count:7,d}  {100 * count / len(merged):5.1f}%")

    budget = min(args.max_samples, len(merged))
    if weights is not None:
        print("\nrequested mix:")
        for level, w in sorted(weights.items(), key=lambda kv: -kv[1]):
            print(f"  {level:8s} {100 * w:5.1f}%  -> {int(budget * w):7,d} rows")
        quota = allocate_weighted(available, budget, weights, args.allow_mix_drift)
    else:
        quota = allocate(available, budget)
    keep: list[int] = []
    for level, count in quota.items():
        keep.extend(by_level[level][:count])
    keep.sort()

    balanced = merged.select(keep).shuffle(seed=args.seed)
    print(f"\nselected {len(balanced):,} rows:")
    for level, count in sorted(quota.items(), key=lambda kv: -kv[1]):
        if count:
            print(f"  {level:8s} {count:7,d}  {100 * count / len(balanced):5.1f}%")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    balanced.save_to_disk(str(out))
    # Per-dataset, so it must be recomputed on the balanced set rather than
    # taken from any one shard.
    save_token_frequency_distribution(dataset=balanced,
                                      output_path=out / "token_freq.pt")
    print(f"wrote {out} and {out / 'token_freq.pt'}")


if __name__ == "__main__":
    main()
