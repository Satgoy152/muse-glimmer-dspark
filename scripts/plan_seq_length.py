"""Pick --seq-length from the recorded prompt lengths, not from the default.

speculators drops an assistant turn whose context alone fills the window
(`_render_boundary_rows`: `if len(prompt_ids) >= max_length: continue`), so a
too-small window does not truncate traces -- it deletes turns, and the run
trains on almost nothing without failing.

Every recorded call is one assistant turn, and its `prompt_tokens` is exactly
the context that turn would be rendered with, so the trade-off can be measured
from the dataset's `raw` config without a tokenizer or a GPU.

    uv run python scripts/plan_seq_length.py
    uv run python scripts/plan_seq_length.py --tok-per-s 5200 --budget-hours 12
"""

import argparse

import numpy as np
import pandas as pd

DATASET = "Satgoy152/Muse-Glimmer-SWE-Gym-2k"
COLS = ["id", "reasoning_strength", "is_error", "prompt_tokens", "completion_tokens"]
LENGTHS = [2048, 4096, 8192, 16384, 24576, 32768, 49152, 65536]


def load_call_metadata(dataset: str) -> pd.DataFrame:
    """Read only the length columns of the `raw` config, over HTTP range reads."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    paths = fs.glob(f"datasets/{dataset}/raw/*.parquet")
    if not paths:
        raise SystemExit(f"no raw parquet under {dataset}")
    frames = []
    for path in sorted(paths):
        with fs.open(path, "rb") as handle:
            frames.append(pq.ParquetFile(handle).read(columns=COLS).to_pandas())
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--cache", help="parquet to read/write the length columns from")
    # Prefill throughput of the target on the extraction node. The default is a
    # rough 30B/H200 estimate; measure it on the box and pass the real number.
    ap.add_argument("--tok-per-s", type=float, default=6600)
    ap.add_argument("--budget-hours", type=float, default=12)
    args = ap.parse_args()

    if args.cache:
        from pathlib import Path

        cache = Path(args.cache)
        if cache.exists():
            df = pd.read_parquet(cache)
        else:
            df = load_call_metadata(args.dataset)
            df.to_parquet(cache)
    else:
        df = load_call_metadata(args.dataset)

    calls = df[(~df.is_error) & df.prompt_tokens.notna()].copy()
    ctx = calls.prompt_tokens.astype(int).to_numpy()
    gen = calls.completion_tokens.fillna(0).astype(int).to_numpy()

    print(f"{len(df):,} calls, {int(df.is_error.sum()):,} errored, "
          f"{calls.id.nunique():,} trajectories")
    print(f"context tokens: mean {ctx.mean():,.0f}  median {np.median(ctx):,.0f}  "
          f"max {ctx.max():,}")
    print(f"assistant tokens: mean {gen.mean():,.0f}  total {gen.sum():,}\n")

    budget = args.tok_per_s * args.budget_hours * 3600
    print(f"one extraction pass at {args.tok_per_s:,.0f} tok/s; "
          f"budget {args.budget_hours:g} h = {budget/1e6:,.0f}M prefill tokens\n")
    header = (f"{'seq_len':>8} {'turns':>9} {'%turns':>7} {'sup tok':>11} {'%sup':>6} "
              f"{'prefill tok':>13} {'GPU-h':>7} {'rows/budget':>12} {'sup/budget':>11}")
    print(header)
    print("-" * len(header))
    for length in LENGTHS:
        kept = ctx < length
        full = np.minimum(ctx + gen, length)
        sup = np.where(kept, np.clip(full - ctx, 0, None), 0)
        cost = np.where(kept, full, 0)
        n_kept = int(kept.sum())
        if n_kept == 0:
            continue
        # Per-row averages over the kept rows, so a budget maps to a row count.
        mean_cost = cost.sum() / n_kept
        mean_sup = sup.sum() / n_kept
        rows_in_budget = min(n_kept, budget / mean_cost)
        print(f"{length:>8} {n_kept:>9,} {100*kept.mean():>6.1f}% {sup.sum():>11,} "
              f"{100*sup.sum()/gen.sum():>5.1f}% {cost.sum():>13,} "
              f"{cost.sum()/args.tok_per_s/3600:>7.0f} {rows_in_budget:>12,.0f} "
              f"{rows_in_budget*mean_sup/1e6:>10.1f}M")

    print("\n%sup is the share of all recorded assistant tokens that survive as "
          "supervision.\nrows/budget and sup/budget are what one extraction pass "
          "buys inside --budget-hours;\npass --max-samples that row count to "
          "`speculators prepare-data`.")


if __name__ == "__main__":
    main()
