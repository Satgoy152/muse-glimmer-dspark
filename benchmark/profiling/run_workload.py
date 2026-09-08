#!/usr/bin/env python3
"""Benchmark pre-rendered prompt token IDs, then optionally profile a separate wave.

Input is {"short": [{"id": "example", "prompt_token_ids": [...]}], ...}.
An enclosing "cases" key is also accepted. Each round is one synchronized wave
of exactly --concurrency requests; prompts cycle across slots and rounds. The
fixed-length, greedy workload is a runtime diagnostic, not a quality evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def http(base_url: str, path: str, timeout: float, body: Any = None,
         *, method: str | None = None) -> bytes:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        base_url + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:4096]
        raise RuntimeError(f"{request.method} {path}: HTTP {exc.code}: {detail}") from exc


def load_prompts(path: Path, case: str) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    document = json.loads(raw)
    cases = document.get("cases", document) if isinstance(document, dict) else None
    if not isinstance(cases, dict) or not isinstance(cases.get(case), list):
        raise ValueError(f"Prompt file must contain a list for case {case!r}")
    prompts = []
    seen_ids = set()
    for index, item in enumerate(cases[case]):
        if not isinstance(item, dict):
            raise ValueError(f"Prompt {index} must be an object")
        prompt_id = str(item.get("id", index))
        token_ids = item.get("prompt_token_ids")
        if (not isinstance(token_ids, list) or not token_ids
                or any(type(token) is not int or token < 0 for token in token_ids)):
            raise ValueError(f"Prompt {prompt_id!r} needs nonempty nonnegative integer token IDs")
        if prompt_id in seen_ids:
            raise ValueError(f"Duplicate prompt id {prompt_id!r}")
        seen_ids.add(prompt_id)
        canonical = json.dumps(token_ids, separators=(",", ":")).encode()
        prompts.append({"id": prompt_id, "prompt_token_ids": token_ids,
                        "token_count": len(token_ids),
                        "sha256": hashlib.sha256(canonical).hexdigest()})
    if not prompts:
        raise ValueError(f"Case {case!r} has no prompts")
    return prompts, hashlib.sha256(raw).hexdigest()


def prompt_metadata(prompt: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in prompt.items() if key != "prompt_token_ids"}


def snapshot(args: argparse.Namespace, path: Path) -> None:
    path.write_bytes(http(args.base_url, "/metrics", args.timeout))


def run_batch(args: argparse.Namespace, model: str, prompts: list[dict[str, Any]],
              phase: str, round_index: int, max_tokens: int, output) -> dict[str, Any]:
    clock: dict[str, float] = {}

    def mark_start() -> None:
        clock["started_at_unix"] = time.time()
        clock["started_monotonic"] = time.perf_counter()

    barrier = threading.Barrier(len(prompts) + 1, action=mark_start, timeout=60)

    def request_one(slot: int, prompt: dict[str, Any]) -> dict[str, Any]:
        body = {
            "model": model,
            "prompt": prompt["prompt_token_ids"],
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "ignore_eos": True,
            "max_tokens": max_tokens,
            "stream": False,
            "n": 1,
        }
        record: dict[str, Any] = {
            "phase": phase, "round": round_index, "slot": slot,
            "concurrency": len(prompts), "prompt": prompt_metadata(prompt),
            "requested_output_tokens": max_tokens,
        }
        barrier.wait()
        started = time.perf_counter()
        record["started_at_unix"] = time.time()
        try:
            response = json.loads(http(args.base_url, "/v1/completions", args.timeout, body))
            record["response"] = response
            if not isinstance(response, dict):
                raise RuntimeError("Completion response is not an object")
            if response.get("error"):
                raise RuntimeError(f"Completion returned error: {response['error']}")
            usage = response.get("usage") or {}
            record["usage"] = usage
            record["speculative_decoding"] = (
                (response.get("metrics") or {}).get("speculative_decoding")
            )
            actual = usage.get("completion_tokens")
            if actual != max_tokens:
                raise RuntimeError(f"Expected {max_tokens} output tokens; received {actual!r}")
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        finished = time.perf_counter()
        record["latency_s"] = finished - started
        record["finished_at_unix"] = time.time()
        record["finish_offset_s"] = finished - clock["started_monotonic"]
        return record

    records = []
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = [pool.submit(request_one, slot, prompt)
                   for slot, prompt in enumerate(prompts)]
        barrier.wait()
        for future in as_completed(futures):
            record = future.result()
            output.write(json.dumps(record, allow_nan=False) + "\n")
            output.flush()
            records.append(record)
    makespan = max(record["finish_offset_s"] for record in records)
    errors = [record for record in records if record.get("error")]
    result = {
        "phase": phase, "round": round_index, "concurrency": len(prompts),
        "started_at_unix": clock["started_at_unix"], "makespan_s": makespan,
        "requests": len(records), "errors": len(errors),
        "output_tokens": sum((record.get("usage") or {}).get("completion_tokens", 0) or 0
                             for record in records),
        "request_latencies_s": [record["latency_s"] for record in records],
    }
    result["tokens_per_s"] = result["output_tokens"] / makespan
    if errors:
        details = "; ".join(f"slot {record['slot']}: {record['error']}" for record in errors)
        raise RuntimeError(f"{phase} round {round_index} failed: {details}")
    return result


def summarize(rounds: list[dict[str, Any]], response_path: Path) -> dict[str, Any]:
    latencies = sorted(latency for item in rounds for latency in item["request_latencies_s"])
    total_tokens = sum(item["output_tokens"] for item in rounds)
    active_time = sum(item["makespan_s"] for item in rounds)
    spec_steps = accepted = drafted = covered = 0
    with response_path.open() as source:
        for line in source:
            record = json.loads(line)
            spec = record.get("speculative_decoding")
            if isinstance(spec, dict) and spec.get("num_spec_steps") is not None:
                covered += 1
                spec_steps += spec.get("num_spec_steps", 0) or 0
                accepted += spec.get("num_accepted_draft_tokens", 0) or 0
                drafted += spec.get("num_draft_tokens", 0) or 0
    return {
        "rounds": rounds,
        "requests": len(latencies), "output_tokens": total_tokens,
        "total_wave_makespan_s": active_time,
        "tokens_per_s": total_tokens / active_time,
        "throughput_definition": "sum(output_tokens) / sum(wave_makespan_s)",
        "request_latency_p50_s": statistics.median(latencies),
        "request_latency_p95_s": latencies[max(0, math.ceil(0.95 * len(latencies)) - 1)],
        "spec_metrics_requests": covered,
        "num_spec_steps": spec_steps, "num_accepted_draft_tokens": accepted,
        "num_draft_tokens": drafted,
        "accept_len": 1 + accepted / spec_steps if spec_steps else None,
        "draft_acceptance_rate": accepted / drafted if drafted else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--concurrency", type=int, choices=(1, 10), default=1)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", action="store_true",
                        help="capture one additional wave after the unprofiled benchmark")
    args = parser.parse_args()
    if args.rounds < 1 or args.max_tokens < 1 or args.timeout <= 0:
        parser.error("rounds, max-tokens, and timeout must be positive")
    args.base_url = args.base_url.rstrip("/")
    prompts, source_hash = load_prompts(args.prompts, args.case)
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        parser.error(f"Output path must be absent or an empty directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
        "case": args.case, "concurrency": args.concurrency, "rounds": args.rounds,
        "max_tokens": args.max_tokens, "warmup_tokens": 32, "profile": args.profile,
        "base_url": args.base_url, "timeout_s": args.timeout,
        "prompt_source": str(args.prompts.resolve()), "prompt_source_sha256": source_hash,
        "prompts": [prompt_metadata(prompt) for prompt in prompts],
        "sampling": {"temperature": 0, "top_p": 1, "top_k": -1,
                     "ignore_eos": True, "stream": False},
        "workload": "One synchronized wave per round; prompt index = (round * concurrency + slot) % prompt_count",
        "cache_policy": "All prompts primed with 32 output tokens at the measured concurrency; final warmup wave wraps if needed; server cache configuration unchanged",
        "purpose": "Fixed-length runtime diagnostic; not a quality evaluation",
    }
    metadata_path = args.out / "metadata.json"
    write_json(metadata_path, metadata)
    try:
        models = json.loads(http(args.base_url, "/v1/models", args.timeout))
        model = models["data"][0]["id"]
        metadata["model"] = model
        metadata["server_models"] = models
        write_json(metadata_path, metadata)
        with (args.out / "warmup.responses.jsonl").open("x") as output:
            for index, start in enumerate(range(0, len(prompts), args.concurrency)):
                wave = [prompts[(start + slot) % len(prompts)]
                        for slot in range(args.concurrency)]
                run_batch(args, model, wave, "warmup", index, 32, output)
        snapshot(args, args.out / "benchmark.before.prom")
        response_path = args.out / "benchmark.responses.jsonl"
        rounds = []
        with response_path.open("x") as output:
            for index in range(args.rounds):
                wave = [prompts[(index * args.concurrency + slot) % len(prompts)]
                        for slot in range(args.concurrency)]
                result = run_batch(args, model, wave, "benchmark", index, args.max_tokens, output)
                rounds.append(result)
                write_json(args.out / "benchmark.rounds.json", rounds)
                print(f"round {index + 1}/{args.rounds}: {result['tokens_per_s']:.2f} tok/s, "
                      f"{result['makespan_s']:.3f} s", flush=True)
        snapshot(args, args.out / "benchmark.after.prom")
        summary = summarize(rounds, response_path)
        write_json(args.out / "benchmark.summary.json", summary)

        if args.profile:
            snapshot(args, args.out / "profile.before.prom")
            profile_rounds = []
            try:
                # Stop is attempted even if start's HTTP response is lost.
                http(args.base_url, "/start_profile", args.timeout, method="POST")
                wave = [prompts[slot % len(prompts)] for slot in range(args.concurrency)]
                with (args.out / "profile.responses.jsonl").open("x") as output:
                    profile_rounds.append(run_batch(args, model, wave, "profile", 0,
                                                    args.max_tokens, output))
                write_json(args.out / "profile.rounds.json", profile_rounds)
            finally:
                original_error = sys.exc_info()[1]
                try:
                    http(args.base_url, "/stop_profile", args.timeout, method="POST")
                except Exception as stop_error:
                    write_json(args.out / "profile.stop_error.json", {"error": str(stop_error)})
                    if original_error is None:
                        raise
                    print(f"Additionally failed to stop profiler: {stop_error}", file=sys.stderr)
            snapshot(args, args.out / "profile.after.prom")
            write_json(args.out / "profile.summary.json",
                       summarize(profile_rounds, args.out / "profile.responses.jsonl"))

        metadata["status"] = "complete"
        print(f"benchmark: {summary['tokens_per_s']:.2f} tok/s; "
              f"latency p50 {summary['request_latency_p50_s']:.3f} s; "
              f"accept_len {summary['accept_len']}; results {args.out}", flush=True)
        return 0
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(metadata_path, metadata)


if __name__ == "__main__":
    raise SystemExit(main())
