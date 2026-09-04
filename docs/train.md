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

| seq_length | turns kept | % turns | supervised positions | prefill tokens |
|---:|---:|---:|---:|---:|
| 2,048 | 2,244 | 1.4% | 242,227 | 4.0M |
| 8,192 | 25,454 | 15.9% | 2,722,734 | 128M |
| 16,384 | 61,518 | 38.5% | 8,483,732 | 577M |
| 32,768 | 124,835 | 78.2% | 20,817,222 | 2.12B |
| 49,152 | 155,172 | 97.2% | 27,239,319 | 3.31B |

### Choosing it against the eval distribution, not the training set

The window has to be judged against where Terminal-Bench actually decodes, and
the two sets are close enough that the training data is representative:

| | SWE-Gym (train) | Terminal-Bench (eval) |
|---|---:|---:|
| calls | 159,724 | 1,755 |
| context, median | 20,663 | 20,282 |
| context, mean | 22,132 | 24,108 |
| context, p99 | 55,420 | 82,321 |
| output tokens, mean | 178 | 296 |

Share of **decoded** tokens emitted at a context inside the window — the number
that matters, since acceptance is measured per generated token:

| seq_length | SWE-Gym decode covered | TB decode covered |
|---:|---:|---:|
| 8,192 | 9.9% | 22.8% |
| 16,384 | 30.4% | 44.4% |
| 32,768 | 73.8% | 75.5% |
| 49,152 | 96.1% | 92.4% |

Window and row budget are separate levers. `--max-samples` trims after fan-out
and shuffling, so a large window with a row cap yields a uniform sample over the
*whole* context range, rather than every row from its bottom slice:

| plan | rows | supervised positions | prefill | TB decode covered |
|---|---:|---:|---:|---:|
| 16,384, no cap | 61,518 | 8.5M | 577M | 44.4% |
| 32,768, cap 50,000 | 50,000 | 8.3M | 848M | 75.5% |
| 49,152, cap 50,000 | 50,000 | 8.8M | 1.07B | 92.4% |

The configured plan is **32,768 with `--max-samples 50000`**: the same
supervision volume as filling a 16,384 window, at 1.5x the extraction cost, for
70% more of the eval's decoding in distribution. 49,152 buys another 17 points
for 26% more cost and is the better run if the smoke test shows the memory
headroom — a row's hidden states are ~3.8 GB there against ~2.6 GB at 32,768.

What a short window does **not** break is the drafter's own context. All five
DSpark decoder layers are `sliding_attention` with `sliding_window: 2048`, so
the draft never attends beyond 2,048 tokens whatever the row length, and
`position_ids` are a plain `arange` per row, compared only within that window.
The exposure is distribution shift in the *target's* hidden states at deep
context, not positional extrapolation.

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

## How the pieces fit

Three processes, one node, one shared filesystem.

```text
  conversations.jsonl
        |
        v
  speculators prepare-data ----HTTP /v1/chat/completions/render----> vLLM
        |                                                             ^
        v  Arrow: input_ids, loss_mask, seq_len                       |
  runs/dspark/data                                                    |
        |                                                             |
        v                                                             |
  torchrun -m speculators.train --on-missing generate ---------------->|
        ^                          (POST /v1/completions, max_tokens=1)
        |                                                             |
        +-- hs_*.safetensors <--- /data/hidden_states <--- hs_connectors
```

`launch_vllm.py` is a thin wrapper. It converts `--target-layer-ids` into two
ordinary vLLM flags and then `exec`s `vllm serve`:

```json
--speculative_config {"method": "extract_hidden_states", "num_speculative_tokens": 1,
    "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [2,14,26,38,50,52]}}}
--kv_transfer_config  {"kv_connector": "ExampleHiddenStatesConnector",
    "kv_connector_extra_config": {"hidden_states_path": "/data/hidden_states"}}
```

`extract_hidden_states` is not real speculation. It writes each requested
layer's hidden states into a dedicated KV cache group (`HiddenStateCacheSpec`)
and returns the sampled token as its own "draft" so verification always passes.
The connector serialises that cache group to a safetensors file and hands the
path back in the response's `kv_transfer_params["hidden_states_path"]`. The
trainer reads the file off the shared filesystem and deletes it.

So the trainer is given two things: `generation.vllm_endpoint` (where to ask)
and `data.data_path` (the prepared rows). It does not stream hidden states over
HTTP -- only the path travels over HTTP.

`hs_connectors` is a separate distribution that `pip install speculators` pulls
in as a dependency. The vLLM container does **not** get it that way: install it
there explicitly, or the `kv_connector` name will not resolve.

Two vLLM constraints follow from the above:

- **Chunked prefill must be off.** Upstream states it is incompatible with the
  feature, so a prefill arrives as one batch and `--max-num-batched-tokens` must
  be at least `--max-model-len`.
- **Prefix caching is unverified here.** Hidden states live in a KV cache group,
  so in principle a cache hit retains them, and turns within one trajectory
  share nearly all their context -- the upside is roughly an order of magnitude
  on extraction cost. The upstream large-model example disables it and gives no
  reason. Measure it (see below) rather than assuming either way. Note that
  `prepare-data` shuffles rows before writing, so even a working prefix cache
  buys nothing unless extraction visits a trajectory's turns together.

## Smoke test on one GPU, before the 8x node

Everything except throughput can be falsified on a single 180 GB card. The 30B
is ~60 GB in BF16, so vLLM and the trainer fit side by side at a small window.

```bash
head -40 /data/train/conversations.jsonl > /data/train/smoke.jsonl

SEQ_LENGTH=8192 TP=1 GPU_MEM_UTIL=0.55 MAX_NUM_SEQS=8   HIDDEN_STATES_PATH=/data/hidden_states bash docker/serve_extract.sh

speculators prepare-data --model meta-models/Muse-Glimmer-30B \
  --data /data/train/smoke.jsonl --output /data/runs/smoke/data \
  --seq-length 8192 --minimum-valid-tokens 16 \
  --render-endpoint http://127.0.0.1:8000

uv run python scripts/verify_prepared.py --data /data/runs/smoke/data \
  --model meta-models/Muse-Glimmer-30B --sample 50

torchrun --standalone --nproc-per-node 1 -m speculators.train \
  --config configs/train_dspark.yaml --print-config \
  --data-path /data/runs/smoke/data --total-seq-len 8192 --max-steps 20
```

What each step is actually testing, in the order it fails:

1. **The extraction server starts against a `ForConditionalGeneration` target.**
   Muse Glimmer is multimodal; the aux-layer capture has to find the inner
   language model. This is the same class of defect as the DSpark serving patch.
2. **Render works and the mask is sane.** `prepare-data` visualises row 0 with
   trainable tokens highlighted; `verify_prepared.py` checks the strength mix.
3. **Hidden states match the stored `input_ids`.** `check_hidden_states` raises
   `Token ids don't match expected token ids` on any re-render mismatch. The
   target ships `processor_config.json`, so `AutoProcessor` returns a
   `ProcessorMixin` and speculators takes its multimodal path -- prepared rows
   carry an extra `messages` column. `train/data.py` only forwards `messages`
   to the hidden-state request when a message's content is a *list*, which ours
   never is, so this should stay on the token-id path. Confirm it does.
4. **The warm start loads.** `DaoCloud/Muse-Glimmer-30B-DSpark` reports
   `speculators_version: "0+source"`. If `--draft.from-pretrained` rejects it,
   fall back to a scratch init with that config's shape and raise `lr` to 1e-4.
5. **Steps run and loss moves.**

Then repeat step 3 with `PREFIX_CACHING=on` and compare wall-clock. That answers
the TP-vs-DP question for the real run.

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

## Scale

Three different quantities get called "tokens" here; keep them apart.

| | |
|---|---:|
| assistant tokens the model actually generated, whole dataset | 28.4M |
| ... of those, inside a 32,768 window | 20.8M |
| ... in a 50,000-row sample of that window | 8.3M |
| **distinct supervised positions trained on** | **8.3M** |
| loss terms, at `block_size` 15 | ~119M |

The last row is not more data. Each anchor predicts the next 15 tokens, so every
token is a target 15 times over, once from each of the 15 anchors that precede
it. The count of independent things being learned from is the 8.3M positions;
the 119M is the number of loss terms computed over them.

`max_anchors` is set to 3072 because DSpark's own default is 3072 while
`DataArgs` defaults to 512, and 512 would clip 8.2% of positions at this
window.

## Replay

The community DSpark was trained on general chat, and this fine-tune can wash
that out. `DaoCloud/Muse-Glimmer-OPB-100K` (148,900 rows) is the replay set.

It is **pre-tokenized** -- `input_ids` and `loss_mask` -- so it takes
`_passthrough_pretokenized` and needs no render endpoint. `--data` is
repeatable and the pretokenized check is per input path, so one command handles
both:

```bash
speculators prepare-data --model meta-models/Muse-Glimmer-30B \
  --data /data/train/conversations.jsonl \
  --data /data/train/opb_replay.jsonl \
  --seq-length 16384 --minimum-valid-tokens 16 \
  --render-endpoint http://127.0.0.1:8000 --output runs/dspark/data
```

Two things to get right:

- **Ratio is set by what goes in, not by a flag.** The two sets are concatenated
  and shuffled, and `--max-samples` then trims the combined pool, so it
  preserves whatever ratio it was handed. Slice the OPB side yourself. Start at
  roughly 3:1 SWE-Gym:OPB by *supervised positions*, not by rows -- OPB rows are
  short chat turns and ours are long agentic ones, so equal row counts are not
  equal supervision.
- **Reconcile on `reasoning_strength`.** OPB carries it as a column, but the
  rows are already tokenized, so it cannot be re-rendered at another level -- it
  can only be filtered. Match its mix to ours (596 medium / 594 high / 593 low /
  198 xhigh by trajectory) by subsampling OPB per level, or the replay set will
  shift the strength distribution the drafter sees.

Replay is not free: pre-tokenized rows still need a full target forward pass for
their hidden states, so they consume the same GPU budget per token as ours.

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
