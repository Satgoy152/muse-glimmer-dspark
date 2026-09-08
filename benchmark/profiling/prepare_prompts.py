#!/usr/bin/env python3
"""Select real short/long coding requests and render exact prompt IDs with vLLM.

Requires pyarrow in the interpreter used to run this script. The output contains
private prompt token IDs; its metadata companion contains only identifiers,
hashes, rendering settings, and token-count comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Rendering drops prior-turn reasoning_content, so a request is shorter here than
# the length recorded when it was originally served. Candidates are therefore
# prefiltered on recorded length but accepted on rendered length, which is what
# this server actually processes.
CASES = {
    "short": {"select_min": 1024, "select_max": 6144, "target": 2400,
              "render_min": 1024, "render_max": 4096},
    "long": {"select_min": 20000, "select_max": 1 << 30, "target": 30000,
             "render_min": 16384, "render_max": 32768},
}
TEMPLATE_FIELDS = {
    "messages", "tools", "chat_template", "chat_template_kwargs",
    "add_generation_prompt", "continue_final_message", "add_special_tokens",
    "chat_template_content_format", "documents", "mm_processor_kwargs",
    "truncate_prompt_tokens",
}


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def request_json(base: str, endpoint: str, timeout: float, body: Any = None) -> Any:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(base + endpoint, data=data,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:4096]
        raise RuntimeError(f"{request.method} {endpoint}: HTTP {exc.code}: {detail}") from exc


def schema_properties(document: dict[str, Any], schema: dict[str, Any],
                      seen: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Resolve request-object fields, including OpenAPI request unions."""
    result = {}
    reference = schema.get("$ref")
    if reference:
        if not reference.startswith("#/"):
            raise ValueError(f"External OpenAPI schema reference is unsupported: {reference}")
        if reference not in seen:
            target: Any = document
            for component in reference[2:].split("/"):
                target = target[component.replace("~1", "/").replace("~0", "~")]
            result.update(schema_properties(document, target, seen | {reference}))
    for keyword in ("allOf", "anyOf", "oneOf"):
        for child in schema.get(keyword, []):
            result.update(schema_properties(document, child, seen))
    result.update(schema.get("properties", {}))
    return result


def endpoint_properties(document: dict[str, Any], endpoint: str) -> dict[str, Any]:
    try:
        schema = document["paths"][endpoint]["post"]["requestBody"]["content"]["application/json"]["schema"]
    except KeyError as exc:
        raise ValueError(f"Cannot inspect the {endpoint} request schema in /openapi.json") from exc
    return schema_properties(document, schema)


def select_candidates(source: Path, strength: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required; run this script with the benchmark .venv interpreter") from exc
    parquet = pq.ParquetFile(source)
    required = {"id", "request", "prompt_tokens", "reasoning_strength"}
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Missing raw-parquet columns: {sorted(missing)}")
    columns = sorted(required | ({"ts", "is_error"} & set(parquet.schema_arrow.names)))
    candidates: dict[str, list[dict[str, Any]]] = {name: [] for name in CASES}
    seen: dict[str, set[str]] = {name: set() for name in CASES}
    stats: dict[str, Any] = {"total_rows": 0, "error_rows": 0,
                             "matching_strength_rows": 0, "strength_counts": {},
                             "cases": {name: {"matching_rows": 0, "unique_requests": 0}
                                       for name in CASES}}
    strengths: Counter[str] = Counter()
    for batch in parquet.iter_batches(batch_size=128, columns=columns):
        for row in batch.to_pylist():
            index = stats["total_rows"]
            stats["total_rows"] += 1
            strengths[str(row.get("reasoning_strength"))] += 1
            if row.get("is_error"):
                stats["error_rows"] += 1
                continue
            if row.get("reasoning_strength") != strength:
                continue
            stats["matching_strength_rows"] += 1
            count = row.get("prompt_tokens")
            if type(count) is not int:
                continue
            for case, bounds in CASES.items():
                if not bounds["select_min"] <= count <= bounds["select_max"]:
                    continue
                stats["cases"][case]["matching_rows"] += 1
                original = row["request"]
                original = json.loads(original) if isinstance(original, str) else original
                if not isinstance(original, dict) or not isinstance(original.get("messages"), list):
                    raise ValueError(f"Invalid request/messages in source row {index}")
                request_strength = (original.get("chat_template_kwargs") or {}).get("reasoning_strength")
                if request_strength != strength:
                    raise ValueError(f"Source row {index}: column strength {strength!r} disagrees with "
                                     f"request chat_template_kwargs strength {request_strength!r}")
                render_inputs = {key: original[key] for key in TEMPLATE_FIELDS if key in original}
                unique_hash = digest(render_inputs)
                if unique_hash in seen[case]:
                    continue
                seen[case].add(unique_hash)
                candidates[case].append({
                    "source_row_index": index, "source_id": str(row["id"]),
                    "source_timestamp": row.get("ts"), "request": original,
                    "original_request_sha256": digest(original),
                    "render_input_sha256": unique_hash,
                    "original_prompt_tokens": count, "distance_from_target": abs(count - bounds["target"]),
                })
    stats["strength_counts"] = dict(sorted(strengths.items()))
    for case, rows in candidates.items():
        rows.sort(key=lambda row: (row["distance_from_target"], row["source_id"],
                                   row["source_row_index"]))
        stats["cases"][case]["unique_requests"] = len(rows)
    return candidates, stats


def render_body(original: dict[str, Any], model: str, supported: dict[str, Any],
                chat_fields: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {"model": model}
    for field in TEMPLATE_FIELDS:
        if field not in original or original[field] is None:
            continue
        if field not in supported:
            raise ValueError(f"/tokenize does not support supplied rendering field {field!r}; "
                             "refusing to drop it")
        body[field] = original[field]
    for field in ("model", "messages"):
        if field not in supported:
            raise ValueError(f"/tokenize schema is missing expected chat field {field!r}")
    # Match chat-completion defaults, rather than trusting /tokenize defaults.
    for field, fallback in (("continue_final_message", False),
                            ("add_generation_prompt", True), ("add_special_tokens", False)):
        if field not in body:
            if field not in supported:
                raise ValueError(f"/tokenize cannot explicitly set {field!r}")
            body[field] = chat_fields.get(field, {}).get("default", fallback)
    if body["continue_final_message"] and "add_generation_prompt" not in original:
        body["add_generation_prompt"] = False
    if original.get("tool_choice") not in (None, "auto"):
        raise ValueError("An explicit tool_choice other than 'auto' needs additional chat-renderer "
                         "handling; refusing to assume /tokenize reproduces it")
    return body


def rendering_metadata(body: dict[str, Any], original: dict[str, Any]) -> dict[str, Any]:
    settings = {field: body[field] for field in (
        "model", "add_generation_prompt", "continue_final_message", "add_special_tokens",
        "chat_template_content_format", "truncate_prompt_tokens") if field in body}
    settings["forwarded_fields"] = sorted(body)
    settings["omitted_request_fields"] = sorted(set(original) - set(body))
    settings["rendered_request_sha256"] = digest(body)
    settings["message_count"] = len(body["messages"])
    settings["tool_count"] = len(body.get("tools") or [])
    for field in ("tools", "chat_template", "chat_template_kwargs", "documents", "mm_processor_kwargs"):
        if field in body:
            settings[field + "_sha256"] = digest(body[field])
    kwargs = body.get("chat_template_kwargs") or {}
    settings["chat_template_kwargs_keys"] = sorted(kwargs)
    settings["chat_template_kwargs_scalar_values"] = {
        key: value for key, value in kwargs.items()
        if value is None or type(value) in (bool, int, float)
        or (isinstance(value, str) and len(value) <= 128)
    }
    return settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--reasoning-strength", default="high")
    parser.add_argument("--count", type=int, default=10, help="unique prompts per case, 1 to 10")
    parser.add_argument("--max-token-delta", type=int, default=0,
                        help="maximum allowed absolute rendered-vs-recorded token difference; default exact")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--out", type=Path, required=True, help="new prompt JSON path")
    args = parser.parse_args()
    if not 1 <= args.count <= 10 or args.max_token_delta < 0 or args.timeout <= 0:
        parser.error("count must be 1..10; max-token-delta nonnegative; timeout positive")
    args.base_url = args.base_url.rstrip("/")
    metadata_path = args.out.with_name(args.out.stem + ".metadata.json")
    if args.out.exists() or metadata_path.exists():
        parser.error(f"Refusing existing output: {args.out} or {metadata_path}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source.resolve()), "base_url": args.base_url,
        "reasoning_strength": args.reasoning_strength, "count_per_case": args.count,
        "max_token_delta": args.max_token_delta,
        "cases": {case: dict(bounds) for case, bounds in CASES.items()},
        "selection": "Unique rendering inputs, nearest recorded prompt length to each target; one reasoning strength",
        "selected": {case: [] for case in CASES},
    }
    # Reserve the metadata path so simultaneous invocations cannot share it.
    with metadata_path.open("x") as output:
        json.dump(metadata, output, indent=2, allow_nan=False)
        output.write("\n")
    try:
        candidates, stats = select_candidates(args.source, args.reasoning_strength)
        metadata["source_counts"] = stats
        insufficient = [case for case, rows in candidates.items() if len(rows) < args.count]
        if insufficient:
            counts = "; ".join(
                f"{case}: {len(candidates[case])} unique "
                f"({CASES[case]['select_min']}..{CASES[case]['select_max']} recorded tokens)"
                for case in CASES)
            raise ValueError(f"Need {args.count} unique rows per case at strength {args.reasoning_strength!r}; "
                             f"found {counts}. Source strength counts: {stats['strength_counts']}. "
                             "Choose a smaller --count or another --reasoning-strength explicitly.")
        models = request_json(args.base_url, "/v1/models", args.timeout)
        model = models["data"][0]["id"]
        document = request_json(args.base_url, "/openapi.json", args.timeout)
        supported = endpoint_properties(document, "/tokenize")
        chat_fields = endpoint_properties(document, "/v1/chat/completions")
        metadata["model"] = model
        metadata["server_models"] = models
        metadata["openapi_sha256"] = digest(document)
        metadata["tokenize_fields"] = sorted(supported)
        prepared: dict[str, list[dict[str, Any]]] = {case: [] for case in CASES}
        for case, rows in candidates.items():
            token_hashes = set()
            for row in rows:
                body = render_body(row["request"], model, supported, chat_fields)
                response = request_json(args.base_url, "/tokenize", args.timeout, body)
                tokens = response.get("tokens") if isinstance(response, dict) else None
                if (not isinstance(tokens, list) or not tokens
                        or any(type(token) is not int or token < 0 for token in tokens)):
                    raise ValueError("/tokenize must return a nonempty integer 'tokens' array")
                if response.get("count") is not None and response["count"] != len(tokens):
                    raise ValueError(f"/tokenize count {response['count']} disagrees with {len(tokens)} token IDs")
                token_hash = digest(tokens)
                if token_hash in token_hashes:
                    metadata["cases"][case]["duplicate_rendered_prompts_skipped"] = (
                        metadata["cases"][case].get("duplicate_rendered_prompts_skipped", 0) + 1)
                    continue
                token_hashes.add(token_hash)
                delta = len(tokens) - row["original_prompt_tokens"]
                prompt_id = f"{case}-{len(prepared[case]):02d}-{token_hash[:12]}"
                audit = {key: value for key, value in row.items() if key != "request"}
                audit.update({"id": prompt_id, "accepted": False,
                              "rendered_prompt_tokens": len(tokens),
                              "token_count_delta": delta, "prompt_token_ids_sha256": token_hash,
                              "rendering": rendering_metadata(body, row["request"])})
                metadata["selected"][case].append(audit)
                write_json(metadata_path, metadata)
                print(f"{prompt_id}: recorded={row['original_prompt_tokens']} rendered={len(tokens)} "
                      f"delta={delta:+d}", flush=True)
                if abs(delta) > args.max_token_delta:
                    raise ValueError(f"{prompt_id}: token count differs by {delta:+d}; allowed absolute delta "
                                     f"is {args.max_token_delta}. Inspect template/tokenizer settings in "
                                     f"{metadata_path}; no prompt file was written.")
                if delta:
                    print(f"WARNING: {prompt_id} has a {delta:+d} token rendering mismatch "
                          "within the explicitly configured tolerance", file=sys.stderr)
                bounds = CASES[case]
                if not bounds["render_min"] <= len(tokens) <= bounds["render_max"]:
                    # Expected: rendering shrinks a recorded request by a variable amount.
                    metadata["cases"][case]["out_of_band_rendered_skipped"] = (
                        metadata["cases"][case].get("out_of_band_rendered_skipped", 0) + 1)
                    print(f"skip {prompt_id}: rendered {len(tokens)} outside {case} band "
                          f"{bounds['render_min']}..{bounds['render_max']}", flush=True)
                    write_json(metadata_path, metadata)
                    continue
                audit["accepted"] = True
                prepared[case].append({
                    "id": prompt_id, "prompt_token_ids": tokens,
                    "original_prompt_tokens": row["original_prompt_tokens"],
                    "rendered_prompt_tokens": len(tokens), "reasoning_strength": args.reasoning_strength,
                })
                if len(prepared[case]) == args.count:
                    break
            write_json(metadata_path, metadata)
            if len(prepared[case]) < args.count:
                raise ValueError(f"{case}: only {len(prepared[case])} in-band rendered prompts among "
                                 f"{len(rows)} unique source requests; need {args.count}")
        with args.out.open("x") as output:
            json.dump(prepared, output, separators=(",", ":"), allow_nan=False)
            output.write("\n")
        metadata["status"] = "complete"
        metadata["output_sha256"] = hashlib.sha256(args.out.read_bytes()).hexdigest()
        print(f"Saved {args.count} prompts per case to {args.out}; audit: {metadata_path}", flush=True)
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
