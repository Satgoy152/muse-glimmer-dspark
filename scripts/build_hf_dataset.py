#!/usr/bin/env python3
"""Build the two HuggingFace configs -- `train` and `raw` -- as parquet.

`train` carries the exported `messages`, which is the field speculators'
build_speculator_training_dataset() reads. It renders them through vLLM and
derives the loss mask itself, so nothing here is tokenized: the tokenizer and
chat template stay a decision made at training time rather than one frozen
into the dataset.

`raw` carries the request/response pair for every recorded call, so the
leakage audit and any re-render can be redone from source. `request` and
`response` stay JSON strings: a chat-completions body nests tool JSON-Schema
to arbitrary depth, and pinning that into a parquet struct would encode this
one agent's tool definition into the dataset schema. The scalars worth
filtering on are promoted to real columns alongside them.
"""
import json, sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SHARD_BYTES = 2 << 30          # roll at ~2 GB, comfortably under the 5 GB limit
BATCH = 2000

MESSAGE = pa.struct([
    ("role", pa.string()),
    ("content", pa.string()),
    ("reasoning_content", pa.string()),
    ("tool_call_id", pa.string()),
    ("tool_calls", pa.list_(pa.struct([
        ("id", pa.string()),
        ("type", pa.string()),
        ("function", pa.struct([("name", pa.string()), ("arguments", pa.string())])),
    ]))),
])

TRAIN_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("reasoning_strength", pa.string()),
    ("messages", pa.list_(MESSAGE)),
    ("tools", pa.string()),                # constant across every row; JSON
])

RAW_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("ts", pa.float64()),
    ("latency", pa.float64()),
    ("model", pa.string()),
    ("reasoning_strength", pa.string()),
    ("temperature", pa.float64()),
    ("top_p", pa.float64()),
    ("top_k", pa.int64()),
    ("n_messages", pa.int64()),
    ("is_error", pa.bool_()),
    ("finish_reason", pa.string()),
    ("prompt_tokens", pa.int64()),
    ("completion_tokens", pa.int64()),
    ("request", pa.string()),
    ("response", pa.string()),
])


def norm_message(m):
    """Flatten the three ways a reasoning trace was recorded into one field.

    litellm echoed the drafter's reasoning back in up to three places:
    `reasoning_content`, `reasoning`, and provider_specific_fields.reasoning.
    Checked across the export, they never disagree -- provider_specific_fields
    also carried a `refusal` that was null on every single message -- so this
    keeps one field and drops the duplicates rather than publishing the same
    text three times.
    """
    psf = m.get("provider_specific_fields") or {}
    reasoning = m.get("reasoning_content") or m.get("reasoning") or psf.get("reasoning")
    return {
        "role": m.get("role"),
        "content": m.get("content"),
        "reasoning_content": reasoning,
        "tool_call_id": m.get("tool_call_id"),
        "tool_calls": m.get("tool_calls") or [],
    }


class ShardWriter:
    """Parquet writer that rolls to a new file once one gets large."""

    def __init__(self, outdir, schema, stem):
        self.outdir = Path(outdir); self.outdir.mkdir(parents=True, exist_ok=True)
        self.schema, self.stem = schema, stem
        self.w = None; self.n = 0; self.rows = 0

    def _roll(self):
        if self.w:
            self.w.close()
        path = self.outdir / f"{self.stem}-{self.n:05d}.parquet"
        self.w = pq.ParquetWriter(path, self.schema, compression="zstd")
        self.path = path; self.n += 1

    def write(self, batch):
        if not batch:
            return
        if self.w is None:
            self._roll()
        self.w.write_table(pa.Table.from_pylist(batch, schema=self.schema))
        self.rows += len(batch)
        if self.path.stat().st_size >= SHARD_BYTES:
            self._roll()

    def close(self):
        if self.w:
            self.w.close()


def build_train(src, outdir):
    w = ShardWriter(outdir, TRAIN_SCHEMA, "train")
    batch = []
    for line in open(src):
        r = json.loads(line)
        batch.append({
            "id": r["id"],
            "reasoning_strength": r.get("reasoning_strength"),
            "messages": [norm_message(m) for m in r["messages"]],
            "tools": json.dumps(r["tools"], sort_keys=True),
        })
        if len(batch) >= BATCH:
            w.write(batch); batch = []
    w.write(batch); w.close()
    print(f"train: {w.rows} rows -> {w.n} shard(s)")


def build_raw(src, outdir):
    w = ShardWriter(outdir, RAW_SCHEMA, "raw")
    batch = []
    for line in open(src):
        r = json.loads(line)
        req, resp = r["request"], r["response"]
        err = "error" in resp
        choice = (resp.get("choices") or [{}])[0] if not err else {}
        usage = resp.get("usage") or {}
        batch.append({
            "id": r["traj"],
            "ts": r.get("ts"),
            "latency": r.get("latency"),
            "model": req.get("model"),
            "reasoning_strength": (req.get("chat_template_kwargs") or {}).get("reasoning_strength"),
            "temperature": req.get("temperature"),
            "top_p": req.get("top_p"),
            "top_k": req.get("top_k"),
            "n_messages": len(req.get("messages") or []),
            "is_error": err,
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "request": json.dumps(req),
            "response": json.dumps(resp),
        })
        if len(batch) >= BATCH:
            w.write(batch); batch = []
    w.write(batch); w.close()
    print(f"raw: {w.rows} rows -> {w.n} shard(s)")


if __name__ == "__main__":
    which, src, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    (build_train if which == "train" else build_raw)(src, outdir)
