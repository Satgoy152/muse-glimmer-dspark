#!/usr/bin/env python3
"""Attribute an Nsight Systems SQLite export to NVTX launch phases.

Example:
  python benchmark/profiling/analyze_nsys.py trace.sqlite \
    --active-range '^profile.request$' --phase-pattern '^profile.phase.' \
    --output summary.json

Only CUDA API calls fully contained in a phase on their launching CPU thread
are attributed to that phase. The innermost matching phase wins. CUDA graph
nodes sharing a cudaGraphLaunch correlation ID inherit its phase. GPU activity
is clipped to the union of the selected active ranges, including partial kernels.
Gap is active wall time minus the UNION of traced kernel/memory intervals; it
does not assume kernels or streams execute serially. This is a trace-activity
gap, not evidence that the CPU caused a stall.

Use --start-ns/--end-ns for an explicit active window when the request marker is
outside the profiled process. By default the window spans first through last
GPU activity: it excludes capture idle periods AND unobserved request head/tail
latency, so it cannot measure full request latency or CPU scheduling overhead.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sqlite3
import sys


PID_MASK = 0xFFFFFFFFFF000000


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def duration(intervals):
    return sum(end - start for start, end in merge_intervals(intervals))


class Windows:
    def __init__(self, intervals):
        self.intervals = merge_intervals(intervals)
        self.ends = [end for _, end in self.intervals]

    def clip(self, start, end):
        if start is None or end is None or end <= start:
            return []
        result = []
        index = bisect_right(self.ends, start)
        for lower, upper in self.intervals[index:]:
            if lower >= end:
                break
            result.append((max(lower, start), min(upper, end)))
        return result


@dataclass(frozen=True)
class NvtxRange:
    start: int
    end: int
    tid: int | None
    name: str


class RangeIndex:
    """Per-thread interval lookup, retaining only user-selected phase ranges."""

    def __init__(self, ranges):
        by_tid = defaultdict(list)
        for item in ranges:
            if item.tid is not None:
                by_tid[item.tid].append(item)
        self.by_tid = {}
        for tid, items in by_tid.items():
            items.sort(key=lambda item: (item.start, -item.end))
            max_ends = []
            for item in items:
                max_ends.append(max(item.end, max_ends[-1] if max_ends else 0))
            self.by_tid[tid] = (items, [item.start for item in items], max_ends)

    def find(self, tid, start, end):
        if tid not in self.by_tid:
            return None
        items, starts, max_ends = self.by_tid[tid]
        index = bisect_right(starts, start) - 1
        matches = []
        while index >= 0 and max_ends[index] >= end:
            item = items[index]
            if item.end >= end:
                matches.append(item)
            index -= 1
        if not matches:
            return None
        # For properly nested ranges, shortest duration is the deepest range.
        return min(matches, key=lambda item: (item.end - item.start, -item.start)).name


def table_schema(connection):
    result = {}
    for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        name = row[0]
        columns = [item[1] for item in connection.execute(f'PRAGMA table_info("{name}")')]
        result[name] = {column.lower(): column for column in columns}
    return result


def get(row, name, default=None):
    # Rows are normalized once on read so common names tolerate case differences.
    return row.get(name.lower(), default)


def read_rows(connection, table):
    cursor = connection.execute(f'SELECT * FROM "{table}"')
    names = [item[0].lower() for item in cursor.description]
    for row in cursor:
        yield dict(zip(names, row))


def require_columns(schema, table, columns):
    if table not in schema:
        raise ValueError(f"Missing {table}: export a trace captured with --trace=cuda,nvtx.")
    missing = [name for name in columns if name.lower() not in schema[table]]
    if missing:
        raise ValueError(f"{table} is missing required columns: {', '.join(missing)}")


def analyze(args):
    path = Path(args.sqlite).resolve()
    if not path.is_file():
        raise ValueError(f"SQLite file does not exist: {path}")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        return analyze_connection(connection, path, args)
    finally:
        connection.close()


def analyze_connection(connection, path, args):
    schema = table_schema(connection)
    warnings = []
    require_columns(schema, "CUPTI_ACTIVITY_KIND_KERNEL", ["start", "end", "correlationId"])
    require_columns(schema, "CUPTI_ACTIVITY_KIND_RUNTIME", ["start", "end", "correlationId"])
    require_columns(schema, "NVTX_EVENTS", ["start", "end"])
    strings = {}
    if "StringIds" in schema:
        strings = {get(row, "id"): get(row, "value") for row in read_rows(connection, "StringIds")}
    else:
        warnings.append("StringIds is absent; unresolved names are emitted as string IDs.")

    def resolve(value):
        if value is None:
            return "<unknown>"
        return str(strings.get(value, value))

    ranges = []
    for row in read_rows(connection, "NVTX_EVENTS"):
        start, end = get(row, "start"), get(row, "end")
        if start is None or end is None or end <= start:
            continue
        name = get(row, "text") or resolve(get(row, "textId"))
        ranges.append(NvtxRange(start, end, get(row, "globalTid"), str(name)))

    active_pattern = re.compile(args.active_range) if args.active_range else None
    if active_pattern:
        active = [item for item in ranges if active_pattern.search(item.name)]
        if not active:
            sample = sorted({item.name for item in ranges})[:30]
            raise ValueError(f"No NVTX ranges match active-range {args.active_range!r}. Names: {sample}")
        windows = Windows((item.start, item.end) for item in active)
        boundary_source = f"NVTX regex: {args.active_range}"
    elif args.start_ns is not None and args.end_ns is not None:
        if args.end_ns <= args.start_ns:
            raise ValueError("--end-ns must be greater than --start-ns")
        windows = Windows([(args.start_ns, args.end_ns)])
        boundary_source = "explicit nanosecond boundaries"
    elif args.start_ns is not None or args.end_ns is not None:
        raise ValueError("Supply both --start-ns and --end-ns")
    else:
        activity_bounds = []
        for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY",
                      "CUPTI_ACTIVITY_KIND_MEMSET"):
            if table in schema and {"start", "end"}.issubset(schema[table]):
                lower, upper = connection.execute(
                    f'SELECT MIN(start), MAX(end) FROM "{table}" WHERE end > start').fetchone()
                if lower is not None:
                    activity_bounds.append((lower, upper))
        if not activity_bounds:
            raise ValueError("No GPU activity exists to derive an active envelope")
        windows = Windows([(min(lower for lower, _ in activity_bounds),
                            max(upper for _, upper in activity_bounds))])
        boundary_source = "first-to-last GPU activity envelope"
        warnings.append("Active window excludes request head/tail latency; GPU gap is measured only between first and last traced GPU activity.")

    phase_pattern = re.compile(args.phase_pattern)
    phase_ranges = [item for item in ranges if phase_pattern.search(item.name)
                    and not (active_pattern and active_pattern.search(item.name))]
    if not phase_ranges:
        warnings.append("No NVTX phase ranges match; all GPU work will remain unclassified.")
    phases = RangeIndex(phase_ranges)
    phase_cpu = defaultdict(list)
    phase_cpu_count = Counter()
    for item in phase_ranges:
        clips = windows.clip(item.start, item.end)
        if clips:
            phase_cpu[item.name].extend(clips)
            phase_cpu_count[item.name] += 1

    runtime = defaultdict(list)
    runtime_count = 0
    runtime_in_active = Counter()
    runtime_duration = Counter()
    for row in read_rows(connection, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        start, end = get(row, "start"), get(row, "end")
        if start is None or end is None:
            continue
        runtime_count += 1
        tid = get(row, "globalTid")
        pid = (tid & PID_MASK) if tid is not None else None
        name = resolve(get(row, "nameId"))
        phase = phases.find(tid, start, end)
        runtime[(pid, get(row, "correlationId"))].append((start, end, phase, name))
        clips = windows.clip(start, end)
        if clips:
            runtime_in_active[name] += 1
            runtime_duration[name] += sum(upper - lower for lower, upper in clips)
    if runtime_count == 0:
        raise ValueError("CUDA runtime table is empty; this export cannot attribute launches.")
    for items in runtime.values():
        items.sort(key=lambda item: (item[0], item[1]))

    have_pid = ("globalpid" in schema["CUPTI_ACTIVITY_KIND_KERNEL"] and
                "globaltid" in schema["CUPTI_ACTIVITY_KIND_RUNTIME"])
    if not have_pid:
        warnings.append("Process identity is missing from CUPTI; ambiguous correlations are unclassified.")
        by_correlation = defaultdict(list)
        for (_, correlation), items in runtime.items():
            by_correlation[correlation].extend(items)
    else:
        by_correlation = None

    def launch_for(row):
        correlation = get(row, "correlationId")
        if have_pid and get(row, "globalPid") is not None:
            candidates = runtime.get((get(row, "globalPid"), correlation), [])
        else:
            candidates = by_correlation.get(correlation, []) if by_correlation is not None else []
        if not candidates:
            return None
        # A launch may return after a kernel starts; compare API start, not end.
        candidates = [item for item in candidates if item[0] <= get(row, "start")]
        if not candidates:
            return None
        if not have_pid and len(candidates) != 1:
            return None
        return max(candidates, key=lambda item: item[0])

    gpu_intervals = []
    kernel_intervals = []
    device_intervals = defaultdict(list)
    phase_gpu = defaultdict(list)
    phase_kernel_sum = Counter()
    phase_count = Counter()
    kernel_stats = defaultdict(lambda: {"count": 0, "duration_ns": 0, "phases": Counter()})
    unclassified = Counter()
    graph_kernel_count = 0
    active_kernel_count = 0
    seen_kernel_count = 0
    for row in read_rows(connection, "CUPTI_ACTIVITY_KIND_KERNEL"):
        seen_kernel_count += 1
        clips = windows.clip(get(row, "start"), get(row, "end"))
        if not clips:
            continue
        active_kernel_count += 1
        launch = launch_for(row)
        phase = launch[2] if launch and launch[2] else "unclassified"
        if phase == "unclassified":
            unclassified["no_runtime_match" if launch is None else "no_matching_phase"] += 1
        if launch and "GraphLaunch" in launch[3]:
            graph_kernel_count += 1
        elapsed = sum(end - start for start, end in clips)
        gpu_intervals.extend(clips)
        kernel_intervals.extend(clips)
        device_intervals[str(get(row, "deviceId", "unknown"))].extend(clips)
        phase_gpu[phase].extend(clips)
        phase_kernel_sum[phase] += elapsed
        phase_count[phase] += 1
        name_id = next((get(row, key) for key in ("demangledName", "shortName", "mangledName")
                        if get(row, key) is not None), None)
        name = resolve(name_id)
        stats = kernel_stats[name]
        stats["count"] += 1
        stats["duration_ns"] += elapsed
        stats["phases"][phase] += elapsed
    if seen_kernel_count == 0:
        raise ValueError("CUDA kernel table is empty; verify CUPTI captured GPU execution.")
    if active_kernel_count == 0:
        raise ValueError("No CUDA kernels overlap the selected active window.")

    memory = {}
    for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if table not in schema:
            continue
        require_columns(schema, table, ["start", "end"])
        count, elapsed = 0, 0
        intervals = []
        for row in read_rows(connection, table):
            clips = windows.clip(get(row, "start"), get(row, "end"))
            if clips:
                count += 1
                elapsed += sum(end - start for start, end in clips)
                intervals.extend(clips)
                device_intervals[str(get(row, "deviceId", "unknown"))].extend(clips)
        gpu_intervals.extend(intervals)
        memory[table.removeprefix("CUPTI_ACTIVITY_KIND_").lower()] = {
            "count": count, "duration_sum_ms": elapsed / 1e6,
            "busy_union_ms": duration(intervals) / 1e6,
        }

    wall_ns = duration(windows.intervals)
    busy_ns = duration(gpu_intervals)
    kernel_sum_ns = sum(phase_kernel_sum.values())
    phase_output = []
    for phase in set(phase_cpu) | set(phase_gpu):
        phase_output.append({
            "phase": phase,
            "kernel_count": phase_count[phase],
            "kernel_duration_sum_ms": phase_kernel_sum[phase] / 1e6,
            "kernel_duration_share_pct": 100 * phase_kernel_sum[phase] / kernel_sum_ns,
            "kernel_busy_union_ms": duration(phase_gpu[phase]) / 1e6,
            "cpu_range_count": phase_cpu_count[phase],
            "cpu_range_duration_sum_ms": sum(end - start for start, end in phase_cpu[phase]) / 1e6,
            "cpu_range_busy_union_ms": duration(phase_cpu[phase]) / 1e6,
        })
    phase_output.sort(key=lambda item: item["kernel_duration_sum_ms"], reverse=True)
    if unclassified:
        warnings.append("Some kernels were not attributed; inspect unclassified_kernel_reasons.")
    if len(device_intervals) > 1:
        warnings.append("Multiple devices: aggregate GPU busy is time ANY traced device was active.")

    return {
        "sqlite": str(path),
        "boundary_source": boundary_source,
        "active_windows_ns": windows.intervals,
        "active_wall_ms": wall_ns / 1e6,
        "gpu_busy_union_ms": busy_ns / 1e6,
        "gpu_gap_ms": (wall_ns - busy_ns) / 1e6,
        "gpu_busy_pct": 100 * busy_ns / wall_ns,
        "kernel_busy_union_ms": duration(kernel_intervals) / 1e6,
        "kernel_duration_sum_ms": kernel_sum_ns / 1e6,
        "kernel_count": active_kernel_count,
        "cuda_graph_kernel_count": graph_kernel_count,
        "unclassified_kernel_reasons": dict(unclassified),
        "nvtx_range_names": dict(Counter(item.name for item in ranges)),
        "devices": {device: {"gpu_busy_union_ms": duration(items) / 1e6,
                             "gpu_gap_ms": (wall_ns - duration(items)) / 1e6}
                    for device, items in device_intervals.items()},
        "phases": phase_output,
        "memory_operations": memory,
        "top_kernels": [
            {"name": name, "count": stats["count"],
             "duration_sum_ms": stats["duration_ns"] / 1e6,
             "duration_share_pct": 100 * stats["duration_ns"] / kernel_sum_ns,
             "phase_duration_sum_ms": {phase: value / 1e6
                                       for phase, value in stats["phases"].items()}}
            for name, stats in sorted(kernel_stats.items(), key=lambda item: item[1]["duration_ns"],
                                      reverse=True)[:args.top]],
        "top_cuda_apis": [
            {"name": name, "count": runtime_in_active[name], "cpu_duration_sum_ms": value / 1e6}
            for name, value in runtime_duration.most_common(args.top)],
        "notes": [
            "GPU phase attribution uses CUDA launch correlation ID and process ID, with the innermost containing NVTX phase on the launch thread.",
            "Durations are clipped to active windows. GPU busy/gap uses interval unions, including traced memory operations.",
            "Kernel-duration sums and phase busy unions may overlap across streams; they are not additive wall-time components.",
            "CPU NVTX duration is elapsed host time, including waits; nested phase durations overlap.",
            "GPU gap means no traced kernel/memory activity in the active windows; it does not establish the cause of inactivity.",
        ],
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sqlite", help="Nsight Systems SQLite export")
    parser.add_argument("--active-range", help="Regex selecting complete NVTX active-request ranges")
    parser.add_argument("--start-ns", type=int, help="Explicit active interval start in trace nanoseconds")
    parser.add_argument("--end-ns", type=int, help="Explicit active interval end in trace nanoseconds")
    parser.add_argument("--phase-pattern", default=r"^(profile[.:/]|target[.:/]|draft[.:/]|bookkeeping)",
                        help="Regex selecting phase NVTX names (default: profile.*, target.*, draft.*, bookkeeping)")
    parser.add_argument("--top", type=int, default=20, help="Number of top kernels and CUDA APIs")
    parser.add_argument("--output", type=Path, help="Write JSON here; otherwise print it")
    args = parser.parse_args()
    if args.active_range and (args.start_ns is not None or args.end_ns is not None):
        parser.error("Use --active-range or explicit boundaries, not both")
    if args.top < 1:
        parser.error("--top must be positive")
    try:
        result = analyze(args)
    except (ValueError, sqlite3.Error, re.error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
