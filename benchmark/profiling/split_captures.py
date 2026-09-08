#!/usr/bin/env python3
"""Split one Nsight report into the separate capture bursts it contains.

`nsys --capture-range=cudaProfilerApi --capture-range-end=repeat` records every
start/stop pair into a single report, so a server that profiled several
workloads yields one file holding several disjoint bursts of activity. Analysing
the whole file would count the idle time between bursts as GPU gap.

Bursts are cut where traced GPU activity stops for longer than --gap-s. Within a
capture the largest true inter-kernel gap is milliseconds at most, so a
multi-second threshold cannot split a single wave. Emitted windows are start/end
nanosecond bounds for `analyze_nsys.py --start-ns/--end-ns`, in capture order.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ACTIVITY_TABLES = (
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_MEMCPY",
    "CUPTI_ACTIVITY_KIND_MEMSET",
)


def activity_intervals(connection: sqlite3.Connection) -> list[tuple[int, int]]:
    present = {row[0] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    intervals: list[tuple[int, int]] = []
    for table in ACTIVITY_TABLES:
        if table not in present:
            continue
        for start, end in connection.execute(
                f'SELECT start, end FROM "{table}" WHERE start IS NOT NULL AND end IS NOT NULL'):
            if end > start:
                intervals.append((int(start), int(end)))
    if not intervals:
        raise ValueError("No traced GPU activity; verify CUPTI captured this report.")
    intervals.sort()
    return intervals


def find_bursts(intervals: list[tuple[int, int]], gap_ns: int) -> list[dict[str, int]]:
    bursts = []
    start, end, count = intervals[0][0], intervals[0][1], 1
    for lower, upper in intervals[1:]:
        if lower - end > gap_ns:
            bursts.append({"start_ns": start, "end_ns": end, "activity_records": count})
            start, end, count = lower, upper, 1
        else:
            end = max(end, upper)
            count += 1
    bursts.append({"start_ns": start, "end_ns": end, "activity_records": count})
    for index, burst in enumerate(bursts):
        burst["index"] = index
        burst["wall_ms"] = (burst["end_ns"] - burst["start_ns"]) / 1e6
    return bursts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sqlite", type=Path, help="Nsight Systems SQLite export")
    parser.add_argument("--gap-s", type=float, default=2.0,
                        help="idle seconds that separate two captures (default 2)")
    parser.add_argument("--expect", type=int,
                        help="fail unless exactly this many bursts are found")
    parser.add_argument("--min-wall-ms", type=float, default=1.0,
                        help="drop bursts shorter than this; they are stray activity, not a wave")
    parser.add_argument("--output", type=Path, help="write JSON here; otherwise print it")
    args = parser.parse_args()
    if args.gap_s <= 0 or args.min_wall_ms < 0:
        parser.error("--gap-s must be positive and --min-wall-ms nonnegative")
    connection = sqlite3.connect(f"file:{args.sqlite}?mode=ro", uri=True)
    try:
        bursts = find_bursts(activity_intervals(connection), int(args.gap_s * 1e9))
    finally:
        connection.close()
    kept = [burst for burst in bursts if burst["wall_ms"] >= args.min_wall_ms]
    for index, burst in enumerate(kept):
        burst["index"] = index
    result = {
        "sqlite": str(args.sqlite), "gap_s": args.gap_s,
        "bursts_found": len(bursts), "bursts_kept": len(kept),
        "dropped_short_bursts": len(bursts) - len(kept),
        "bursts": kept,
        "note": "Windows bound traced GPU activity only; they exclude request head/tail host time.",
    }
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")
    if args.expect is not None and len(kept) != args.expect:
        print(f"error: expected {args.expect} captures, found {len(kept)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
