# Training the DSpark drafter

Fine-tunes `DaoCloud/Muse-Glimmer-30B-DSpark` on the 1,981 agentic coding
trajectories in `Satgoy152/Muse-Glimmer-SWE-Gym-2k`, to beat the official DFlash
drafter on Terminal-Bench.

**Target to beat: pooled acceptance length 3.913** at 15 speculative tokens on
the frozen 40 ([benchmark/terminal_bench/README.md](../benchmark/terminal_bench/README.md)).
The 4.925 in `speculator_history.csv` is the same drafter measured on the
SWE-Gym generation workload, not on the eval set; it is the training-domain
number and is not the bar.

`block_size` is pinned to 15 in `configs/train_dspark.yaml` so acceptance length
is measured at the same draft length as the baseline. Change it and the
comparison against 3.913 no longer holds.

## Two silent failures

### 1. The sequence-length window deletes turns

`speculators` builds one training row per assistant turn, and skips a turn whose
context alone fills the window:

```python
# speculators/data_generation/preprocessing.py, _render_boundary_rows
if len(prompt_ids) >= max_length:
    continue
```

Nothing is truncated and nothing errors — the dataset just comes out small.
Prompts here average 22,132 tokens and reach 85,124, so the defaults are far
below the workload: the Python API `build_speculator_training_dataset` defaults
`max_length=2048`, and the `prepare-data` CLI defaults `--seq-length 8192`.

Measured from the `raw` config's recorded `prompt_tokens`, one row per call
(`scripts/plan_seq_length.py`):

| seq_length | turns kept | % turns | supervised tokens | % of all | prefill tokens | H200-h¹ |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 2,244 | 1.4% | 242,227 | 0.9% | 4.0M | 0 |
| 8,192 | 25,454 | 15.9% | 2,722,734 | 9.6% | 128M | 5 |
| 16,384 | 61,518 | 38.5% | 8,483,732 | 29.9% | 577M | 24 |
| 24,576 | 95,642 | 59.9% | 14,985,332 | 52.8% | 1.28B | 54 |
| 32,768 | 124,835 | 78.2% | 20,817,222 | 73.3% | 2.12B | 89 |
| 49,152 | 155,172 | 97.2% | 27,239,319 | 95.9% | 3.31B | 139 |

¹ one full extraction pass at an assumed 6,600 prefill tok/s. **Measure the real
rate on the box and re-run `scripts/plan_seq_length.py --tok-per-s`** before
committing to a row budget; the whole right-hand side scales with it.

The config uses **16,384**. Rationale: within a fixed GPU-hour budget it yields
the most supervision, because context grows much faster than the assistant turn
it supervises (mean 22,132 tokens of context per 178 supervised tokens), so a
larger window buys coverage at a worse rate. 8,192 is cheaper per row still, but
holds only the first ~16% of turns and cuts exactly the long-context regime this
project exists to improve. Raise it if the budget allows; log the value used.

Three separate knobs must agree: `prepare-data --seq-length`,
`data.total_seq_len` in the train config, and `--max-model-len` on the
extraction server.

### 2. Reasoning strength lives in the system message

`chat_template_kwargs` is a request field, invisible to anything rendering from
`messages` — and `_render_conversation_rows` never forwards it to the render
endpoint, even though `render_conversation` accepts it. `export_traces.py`
therefore baked `Reasoning strength: <level>.` into the system text, which the
chat template honours (`{%- if 'reasoning strength' not in (sys_text|lower) -%}`).

**Do not strip, rewrite, or regenerate system messages.** Doing so re-renders
all 1,981 trajectories at the template default of `high`, against a real mix of
596 medium / 594 high / 593 low / 198 xhigh. Nothing errors.

`scripts/build_train_jsonl.py` asserts the sentence appears exactly once per
trajectory before writing, and `scripts/verify_prepared.py` decodes prepared
rows and checks the rendered mix afterwards. Run both.

## Machine

One H200 is enough for a single-GPU run, but vLLM and the trainer then share it:
the 30B target is ~60 GB in BF16 and `--gpu-memory-utilization 0.90` leaves
little for the draft. Prefer 2×H200 (or 2 nodes) with the server on one and
`torchrun` on the other, and set `generation.vllm_endpoint` accordingly.

Hidden states are ~78 KB per token (5 auxiliary layers + the final layer, hidden
size 6,656, bf16), so a 16,384-token row is ~1.3 GB. They are **not** cacheable
at this scale — the config runs online with `on_generate: delete`, and
`data.num_workers`/`prefetch_factor` are cut to 2 for the same reason.

## Pipeline

```bash
# 0. seq-length budget, once, on the real throughput number
uv run python scripts/plan_seq_length.py --tok-per-s <measured> --budget-hours 12

# 1. traces -> the `conversations` column prepare-data reads
uv run python scripts/build_train_jsonl.py --out /data/train/conversations.jsonl
```

`build_train_jsonl.py` renames `messages` to `conversations`. That rename is
load-bearing: `_preprocess_batch` reads `examples.get("conversations", [])` and
a `messages` column produces zero rows with only a log line. `tools` is passed
through as the JSON string `_parse_conv_tools` expects.

```bash
# 2. extraction server (tmux; see benchmark/terminal_bench/README.md on why)
SEQ_LENGTH=16384 HIDDEN_STATES_PATH=/data/hidden_states \
  bash docker/serve_extract.sh

# 3. render + loss masks. --max-samples is the row budget from step 0.
speculators prepare-data \
  --model meta-models/Muse-Glimmer-30B \
  --data /data/train/conversations.jsonl \
  --output runs/dspark/data \
  --seq-length 16384 \
  --max-samples 30000 \
  --minimum-valid-tokens 16 \
  --render-endpoint http://127.0.0.1:8000

# 4. verify before spending GPU hours
uv run python scripts/verify_prepared.py \
  --data runs/dspark/data --model meta-models/Muse-Glimmer-30B

# 5. train
torchrun --standalone --nproc-per-node 1 -m speculators.train \
  --config configs/train_dspark.yaml --print-config
```

`--max-samples` caps rows *after* fan-out and shuffling, so it is a direct row
budget — but rendering still runs over every trajectory first, so it bounds GPU
cost, not preprocessing time.

Keep the resolved `--print-config` output next to the checkpoints. Checkpoints
go to the persistent volume at `checkpoint_freq: 0.1`; mirror to HF at the end
only.

## What training tells you, and what it does not

The trainer logs `loss` and `accuracy`. That accuracy is per-position
teacher-forced argmax agreement, conditioned on all earlier positions in the
block being correct (`speculators/models/metrics.py`). It is a useful convergence
signal and an upper bound, but it is **not** the served acceptance length: the
baseline was measured under sampling (temperature 1.0, top_p 0.95, top_k 64) with
rejection sampling, prefix caching, and real batching. Do not report a training
curve as if it were comparable to 3.913 or 4.925.

The only comparable number comes from re-running the Terminal-Bench harness with
the new drafter:

```bash
SPEC_METHOD=dspark SPECULATOR=<trained-checkpoint> NUM_SPEC_TOKENS=15 \
  bash docker/serve_patched.sh
```

Same frozen 40, same `--concurrent 8`, same one-segment-at-a-time schedule, its
own `TRACE_DIR` and proxy port. The DSpark patch in `serve_patched.sh` is still
required: `target_language_model.model` is unguarded on vLLM `main` as of
2026-09, not only on the 0.28.1rc1 the generation run used.
