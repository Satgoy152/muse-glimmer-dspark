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

Widening past 32,768 is unusually cheap here, because contexts that deep are
rare: almost all of the extra prefill is already paid by 49,152, and 65,536
costs only 4% more than that while closing nearly the whole gap.

| plan | rows | sup positions | prefill | h @20k tok/s | TB decode covered | hidden states/row |
|---|---:|---:|---:|---:|---:|---:|
| 32,768, cap 50,000 | 50,000 | 8.3M | 848M | 11.8 | 75.5% | 2.6 GB |
| 49,152, cap 50,000 | 50,000 | 8.8M | 1.07B | 14.8 | 92.4% | 3.9 GB |
| 65,536, cap 50,000 | 50,000 | 8.9M | 1.11B | 15.4 | 98.0% | 5.2 GB |

The config is set to **49,152 with `--max-samples 50000`**. Try 65,536 first in
the smoke test and drop down this ladder only if it does not fit: per-row hidden
states reach 5.2 GB there, and the logit tensor over a 202,048 vocab at 3,072
anchors x 15 positions is large independent of the window (see
`loss.loss_implementation` for chunking).

At 49,152 a uniform row sample already lands close to the eval's decode-weighted
context profile, so no stratified sampling is needed:

| context | train rows | eval decode |
|---|---:|---:|
| <4k | 6.2% | 11.5% |
| 4–8k | 10.2% | 11.2% |
| 8–16k | 23.2% | 21.6% |
| 16–24k | 22.0% | 16.5% |
| 24–32k | 18.8% | 14.6% |
| 32–48k | 19.6% | 16.9% |
| 48k+ | 0.0% | 7.6% |

The one real gap is the last row, and it is the reason to prefer 65,536.

### Why the drafter's own sliding window does not make this safe

All five DSpark decoder layers are `sliding_attention` at `sliding_window: 2048`,
and `position_ids` are a plain `arange` per row, compared only inside that
window. That rules out *positional* extrapolation — the draft never sees an
unfamiliar relative offset at 85k. It does not rule out the window mattering,
because the draft's input is not raw tokens. It is the target's hidden states,
and the target is a hybrid:

```
layer_types  = [sliding, sliding, sliding, full] x 13   (52 layers)
full attention at 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51
layer_rope_theta == 0 at exactly those indices -> the global layers are NoPE
```

Against the five aux capture points, plus the final hidden state:

| aux layer | own type | full-attention layers below it |
|---:|---|---:|
| 2 | sliding | 0 |
| 14 | sliding | 3 |
| 26 | sliding | 6 |
| 38 | sliding | 9 |
| 50 | sliding | 12 |
| 52 (final) | — | 13 |

Only layer 2 is context-local. Everything from layer 14 up has already mixed
through several unbounded-attention layers, so the hidden states the drafter
consumes encode the entire prompt, not the last 2,048 tokens. A state taken at
position 60,000 of an 85,000-token trajectory is genuinely a different input
distribution from anything a 16,384-token row contains — and because the global
layers are NoPE, their attention is pure content matching whose mass spreads as
the prompt grows, so the shift is a function of prompt length directly.

The token sequence inside the draft's own 2,048-token window is also not
independent of this: those tokens were themselves generated conditioned on the
full context, so the local distribution the draft models carries long-range
structure even where it cannot attend to it.

Train at a window that covers where the eval actually decodes. The drafter's
sliding window bounds its positional exposure and nothing else.

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

- **Chunked prefill must be off, and vLLM will not do it for you.**
  `enable_chunked_prefill` defaults to `True` and nothing in the config layer
  ties it to `extract_hidden_states` — the only automatic disable is for
  encoder-decoder models. The flag is ours to set.

  The reason: hidden states are written into the cache by the *proposer*
  (`ExtractHiddenStatesProposer.propose`), which runs on the post-sampling
  drafting path. A request mid-way through a chunked prefill has not sampled
  anything that step, so its chunk never reaches `propose()` and those tokens'
  slots are never written. At `request_finished` the connector reads
  `len(prompt_token_ids)` slots straight out of the block list regardless, so
  the extracted tensor still has the right shape and the right `token_ids` —
  only the values for every chunk but the last are stale cache memory.
  `check_hidden_states` compares shapes and token ids and scans for non-finite
  values, none of which catches uninitialised memory. This one fails silently.

  Because chunked prefill is off, `--max-num-batched-tokens` must be at least
  `--max-model-len`; `SchedulerConfig.verify_max_model_len` raises otherwise.
  That in turn sizes a permanent GPU buffer in the proposer of
  `(max_num_batched_tokens + max_num_seqs) x 6 layers x 6656 x bf16` — about
  3.9 GB at a 49,152 window.
- **Prefix caching is broken here, and this is now measured.** Two runs with
  caching off produce bitwise-identical hidden states (max abs diff 0); a run
  with caching on differs from that baseline by max 348 / mean 0.134 over the
  cached span. It is the chunked-prefill bug in another dress: a cache hit does
  not recompute the span, so `propose()` never writes it, while
  `request_finished` still reads `len(prompt_token_ids)` slots out of the block
  list. Shapes, token ids and finiteness all pass; only the values are stale.

  The upside was smaller than hoped anyway. Hidden-state serialisation scales
  with total tokens regardless of cache hits -- every request writes
  `N x 79,872 B` and allocates N tokens of hidden-state cache -- so only the
  attention prefill is skipped. Measured speed-up on an 8-turn synthetic
  trajectory was 1.5x against an ideal of 7.3x. `PREFIX_CACHING=off` stays, and
  there is no reason to reorder rows by trajectory to chase hits.

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
  --config configs/train_dspark.yaml --dump-config \
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
4. **`get_hidden_size()` returns 6656, not 6144.** The proposer sizes its
   buffer from `model_config.get_hidden_size()`. Muse Glimmer's config carries
   `out_hidden_size: 6144` at the top level and `text_config.hidden_size: 6656`;
   if the former wins, the extracted states are the wrong width for a draft
   built on 6656. Check the shape in the first safetensors file.
5. **The warm start loads.** `DaoCloud/Muse-Glimmer-30B-DSpark` reports
   `speculators_version: "0+source"`. If `--draft.from-pretrained` rejects it,
   fall back to a scratch init with that config's shape and raise `lr` to 1e-4.
6. **Steps run and loss moves.**

Then repeat step 3 with `PREFIX_CACHING=on` and compare wall-clock. That answers
the TP-vs-DP question for the real run.

## Verified on hardware (1xH200, vLLM 0.28.1rc1.dev388+g8a728663c)

Everything below was run on a preemptible Nebius node: 1x H200 (143 GB),
16 vCPU, 196 GB RAM, plus a 279 GB persistent disk mounted at `/mnt/data`.

### `hs_connectors` is not shipped with vLLM

The stock `vllm/vllm-openai:nightly` image has
`vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py`
and `vllm/v1/spec_decode/extract_hidden_states.py`, but `import hs_connectors`
fails. It is a separate package that lives in the speculators repo and has to be
installed into the vLLM environment. `docker/Dockerfile.train` does that, and
the order matters:

```
pip install setuptools-git-versioning        # speculators' build backend
pip install <speculators>/hs_connectors      # before speculators itself
pip install --no-build-isolation <speculators>
```

Without `--no-build-isolation` the speculators build resolves its own torch and
shadows vLLM's. With it, torch 2.13.0+cu130, transformers and vllm are all left
untouched -- checked after install.

### `--block-size 128` is mandatory, or the engine will not start

Muse Glimmer is GQA 32:2 with `head_dim` 128, so one attention page at the
default block size is `2 x 128 x 2 x 2 x 16 = 16,384 B`. One token of hidden
state is `6 x 6656 x 2 = 79,872 B`. `kv_cache_utils._get_kv_cache_groups` aligns
the hidden-state group's page to the attention page, clamps its block size to 1,
and still overflows:

```
File ".../vllm/v1/kv_cache_interface.py", line 428, in page_size_bytes
    assert self.page_size_padded >= self.unpadded_page_size_bytes
AssertionError
```

`--block-size 128` lifts the attention page to 131,072 B, the first power of two
above 79,872. The engine then logs

```
Using block size 1 for hidden-state cache layer cache_only_layers.52;
page alignment wastes 51200 bytes (39.06%) per block
```

and starts. That 39% is unavoidable padding in the hidden-state cache, and it is
what makes the window ceiling below as low as it is.

### Extraction works, and the width is 6656

A `max_tokens=1` completion over 64 token ids returned, via the connector:

```
hidden_states (64, 6, 6656) torch.bfloat16
token_ids (64,) match: True     finite: True
per-layer norms: [764, 1600, 1720, 2000, 3952, 8960]
```

So `model_config.get_hidden_size()` resolves to `text_config.hidden_size`, not
the top-level `out_hidden_size: 6144`. That smoke-test risk is closed.

### One H200 caps the window at ~34,600 tokens

At `--max-model-len 49153` and `--gpu-memory-utilization 0.92` the engine
refuses to start:

```
To serve at least one request with the model's max seq len (49153),
80.45 GiB KV cache is needed, which is larger than the available KV cache
memory (56.69 GiB). Based on the available memory, the estimated maximum
model length is 34639.
```

The hidden-state cache dominates that budget. **49,152 is not reachable on a
single H200**, and 34,639 is the ceiling before any trainer shares the card, so
32,768 is the working window here. Reaching 49,152 needs the weights sharded
across more than one GPU to free cache -- which is the argument for TP that
`serve_extract.sh` already carries, now with a measured number behind it.

### The trainer and the extraction server cannot share one H200

The pipeline runs end to end up to the training step. Training itself then OOMs,
and the arithmetic is not close:

| | |
|---|---:|
| extraction server, weights + minimal cache | ~61 GB |
| trainer, before its first forward | ~57 GB |
| the forward's next allocation | 5.8 GB |
| H200 | 139.8 GB |

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 5.78 GiB.
GPU 0 has 139.80 GiB total of which 4.69 GiB is free.
```

The trainer does **not** hold a copy of the verifier -- `load_model_layers`
pulls only `model.embed_tokens.weight` and `lm_head.weight`. At a 202,048-token
vocabulary those two alone are 5.4 GB in bf16, and the draft carries its own
pair; with fp32 master weights and Muon/AdamW state on top, the vocabulary is
most of the trainer's footprint before any activations.

So on a multi-GPU node, **put the server and the trainer on disjoint GPUs**.
Sharing a card is not a tuning problem.

### `max_anchors` is the first thing that OOMs

Each anchor predicts `block_size` tokens over the full vocabulary, so the logit
tensor is `anchors x 15 x 202,048`. At `max_anchors` 3072 that is tens of GB in
fp32 and it is what fails first, before activations or optimizer state.

Measured over the 1,119 prepared rows: supervised positions per row are
mean 172, median 91, p99 1053, max 2506.

| max_anchors | supervised positions clipped | rows affected |
|---:|---:|---:|
| 512 | 10.75% | 5.90% |
| 1024 | 3.69% | 1.07% |
| 3072 | 0% | 0% |

The config uses **1024**: it costs 3.7% of the supervision and removes the
largest single memory term. `loss.loss_implementation` is `fused`, which already
helps; lowering anchors is the lever with a measurable data cost attached.

### Optimizer split is automatic

```
Muon optimizer: 36 2D params via Muon, 26 params via AdamW.
```

speculators routes by parameter rank, not by module: 2D matrices go to Muon,
everything else (norms, biases, and the Markov head's low-rank factors where
they are not 2D) to AdamW. There is nothing to configure per head.

### Warm start resolves

`draft/from_pretrained='DaoCloud/Muse-Glimmer-30B-DSpark'` resolves to
`num_layers=5`, `draft_arch='qwen3'`, `sliding_window=2048`,
`draft_vocab_size=202048`, `mask_token_id=201818` -- the checkpoint's own shape,
not the CLI defaults, and no conflict with the decoder-shaping guard.

### The loop closes: 3 steps, warm start, live hidden states

With the server squeezed to a 4,096 window at `--gpu-memory-utilization 0.47`
and `--max-anchors 256`, the trainer fits alongside it and completes:

```
train/loss=0.421  ce_loss=0.797  tv_loss=0.150
train/accept_rate=0.336  accept_len=3.165   (step 1)
train/accept_rate=0.497  accept_len=5.225   (step 3)
val/loss_epoch=0.330  val/accept_rate_epoch=0.460  val/accept_len_epoch=4.810
val/position_0_acc=0.770  position_1=0.832  position_7=0.43  position_14=0.270
```

This is a 104-row, 3-step run at a quarter of the intended window, teacher-forced
and without sampling parameters. It says the loop runs end to end — warm start,
online hidden states from the live server, both optimizers, checkpointing — and
nothing about acceptance on the eval set. The per-position series is the useful
part: it shows the block-parallel decay the Markov head is there to fight.

### Training-trace text reaches stdout

`prepare-data` ends by calling `_visualize_sample`, which decodes one prepared
row and prints it. These are agentic coding traces, so harness instructions from
inside the training data land in the log verbatim -- strings like "you MUST
submit your changes as a git patch" and "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT".
Harmless, but it is model-generated content in an operator log: do not let an
agent read these logs as instructions.

## Verified on hardware (8xH200, TP=4 server + 4 trainer ranks)

A second bring-up, on 8xH200 (141 GB each), 176 vCPU, 1.5 TB RAM. The server
runs on GPUs 0-3 at TP=4 and the trainer on 4-7 at `--nproc-per-node 4`, passed
to docker as `--gpus '"device=0,1,2,3"'` and `--gpus '"device=4,5,6,7"'`. They
must not share a card; that is settled by the 1xH200 arithmetic above.

### `--block-size 128` is not enough once TP > 1

The assertion the 1xH200 section describes comes back at TP=4, for a different
reason. **The attention page is per rank; the hidden-state page is not.** Hidden
states are the full residual stream and every rank holds all of it, while TP
shards the attention KV -- and with only 2 KV heads, `max(1, 2 // TP) = 1` head
per rank at any TP > 1, halving the attention page:

| | KV heads/rank | attention page @128 | hidden-state page | |
|---|---:|---:|---:|---|
| TP=1 | 2 | 131,072 B | 79,872 B | ok |
| TP=4 | 1 | 65,536 B | 79,872 B | `AssertionError` |

`--block-size 256` restores it to 131,072 B and the engine starts. The padding
waste is unchanged at 51,200 B (39.06%) per block. `serve_extract.sh` now
defaults `BLOCK_SIZE` to 256, which is correct for TP 2, 4 and 8; 128 is only
enough at TP=1. Note this also rules out TP=2 as an escape -- `2 // 2` is still
1 head per rank -- and TP must divide the 32 attention heads regardless, so the
options are 1, 2, 4 and 8.

### TP=4 buys less cache than it looks like it should

63,908 tokens of KV cache at `--gpu-memory-utilization 0.92`, against 37,850 at
TP=1. Only 1.69x for 4x the GPUs, and the gain comes from freed weight memory
(~60 GB/card down to ~15 GB), not from sharding the cache:

| | attention (52 layers) | hidden-state group | total/token |
|---|---:|---:|---:|
| TP=1 | 53,248 B | 131,072 B | 184,320 B |
| TP=4 | 26,624 B | **131,072 B, unchanged** | 157,696 B |

The hidden-state group is 71% of the per-token cost at TP=1 and 83% at TP=4, so
TP makes the dominant term relatively *worse*. It also means DCP and other
KV-parallelism flags do not help: they shard attention KV, which is the small
half, and the hidden-state group is a custom `HiddenStateCacheSpec` those paths
know nothing about.

At 63,908 tokens: 1.95x concurrency at a 32,768 window, 1.30x at 49,152, and
65,536 will not start.

### `num_workers` is the throughput knob, and 1 is far too low

The single largest performance finding of the second bring-up. In-flight
hidden-state requests are `nproc-per-node x num_workers`, so the shipped
`num_workers: 1` issued four concurrent requests for the whole job and left the
server idle ~70% of the time.

| num_workers | in flight | fetch_ms | fetch_frac | step_ms | rows/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 14,100 | 0.913 | 15,400 | 0.41 |
| 8 | 32 | **157** | **0.105** | ~2,400 | ~2.0 |

That is the difference between a 33-hour and a 3.4-hour pass over 25,000 rows.
`prefetch_factor` is not a second lever -- a worker drains its queue serially,
so it buys buffering depth rather than concurrency, and with fetch at 10x
compute the main process is never the one waiting. `nproc-per-node` is not one
either: it changes the global batch and the step count, so it changes the run.

Keep `nproc x num_workers <= --max-num-seqs` on the server (raised to 32 here).

### Measured throughput

At TP=4 with 32 in-flight requests and a 32,768 window: **~34,000 prefill
tok/s**, ~2.0 rows/s at a 17,048-token mean row, ~2.4 s/step at 4 ranks. That is
5x the 6,600 tok/s `plan_seq_length.py` assumes by default, so re-run the
planner with the measured number before setting `--max-samples`.

Per-step cost at that point is `fwd 356 + bwd 382 + opt 601 = 1,339 ms` against
157 ms of fetch -- the trainer is compute-bound, and `opt_ms` (Muon's
Newton-Schulz iterations, roughly fixed per step) is the largest single term.

### The extraction endpoint must bind loopback

`serve_extract.sh` set no `--host`, so vLLM bound all interfaces. With no API
key, that is an open H200: the bring-up node was being probed by an external
scanner (`GET /`, `POST /mcp`, `POST /jsonrpc`, `/robots.txt`) within two hours.
Both the trainer and `prepare-data` reach the server over `127.0.0.1`, so
`--host 127.0.0.1` costs nothing. Fixed.

### Rendering is cheap, and shards are independently usable

`prepare-data` on one shard of 62 trajectories: **1m55s**, 3,740 rows at a
32,768 window. All 32 shards is about an hour of CPU -- the render endpoint only
applies the chat template and tokenises, it never runs the model. Because
`shard_jsonl.py` assigns round-robin, **any subset of shards is an unbiased
sample**, so training can start on a partial render.

### Smaller things that cost time

- The CLI flag is `--dump-config`, not `--print-config`.
- Hidden-state files are written asynchronously and guarded by a `<path>.lock`
  sentinel that persists after the write. Wait by acquiring an `flock` on it,
  not by polling for the lock file to disappear.
- Prepared rows carry an extra `messages` column (the target ships
  `processor_config.json`, so `AutoProcessor` returns a `ProcessorMixin`), but
  the trainer stays on the `/v1/completions` token-id path because no message's
  content is a list. Confirmed on the wire.
- A non-empty `--save-path` makes the trainer resume a finished epoch and exit
  in seconds, which reads as success. Clear it, or use a fresh path.
- Supervised positions per row at 32,768: mean 176, median 89, max 1,391 over
  400 sampled rows -- close enough to the 1xH200 numbers that `max_anchors:
  1024` still clips ~3.7%.


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
  --config configs/train_dspark.yaml --dump-config
```

`--max-samples` caps rows *after* fan-out and shuffling, so it is a direct row
budget — but rendering still runs over every trajectory first, so it bounds GPU
cost, not preprocessing time.

Keep the resolved `--dump-config` output next to the checkpoints. Checkpoints
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

DSpark's own metrics compute acceptance directly, so the run is monitorable
against the baseline's *form* without waiting for a Terminal-Bench re-run
(`speculators/models/dspark/metrics.py`):

```python
accept_rate = 1.0 - tv_loss_fn(logits, targets)      # sum_v min(q_v, p_v)
accept_prefix = (accept_blocks[:, start_pos:] * draft_mask).cumprod(dim=-1)
per_block_len = accept_prefix.sum(dim=-1) + 1.0      # DSpark's tau
```

`accept_rate` is the analytical single-token acceptance probability under
speculative sampling's rejection rule — the distributional overlap between draft
and target, not greedy argmax agreement. `accept_len` is the expected accepted
block length built from it: the cumulative acceptance product summed over draft
slots plus the always-emitted anchor. That is the same functional form as the
pooled acceptance length the eval harness reports.

Logged per step and per validation epoch: `accept_len`, `accept_rate`,
`full_acc`, `position_0_acc` … `position_14_acc`, `confidence_loss`,
`confidence_abs_error`, `confidence_pred_mean`, `confidence_cumprod_bias`.
The per-position series is the block-parallel decay the Markov head exists to
fight, visible directly.

Three reasons these are still not the served number:

* **Teacher-forced.** Every block conditions on the true prefix. At serving, a
  rejection ends the block and the next one starts from a different state.
* **No sampling parameters.** The overlap is over raw softmaxes; the baseline
  ran at temperature 1.0 / top_p 0.95 / top_k 64, which reshapes both
  distributions.
* **No batching or prefix caching**, both of which move the measured number.

Track them to see the run improving and to compare checkpoints against each
other. Do not put a training-time `accept_len` beside 3.913 and call it a win.

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

## The DFlash2 candidate (run D): blocked on the warm start

A parallel candidate fine-tuning `z-lab/Muse-Glimmer-30B-DFlash2` on the same
data. Everything below was measured on the 8xH200 node; the run is **not
launched** — the warm start does not load, and the fix is a decision, not a
patch to apply blindly.

### DFlash2 consumes the same target layers as DSpark

Not obvious from the configs, which disagree on their face:

| repo | key | value |
|---|---|---|
| `DaoCloud/…-DSpark` | `aux_hidden_state_layer_ids` | `[2, 14, 26, 38, 50]` |
| `z-lab/…-DFlash2` | `dflash_config.target_layer_ids` | `[1, 13, 25, 37, 49]` |

They are the same five capture points in two index spaces.
`convert/dflash/converter.py` maps between them:

```python
# z-lab reads hidden_states[layer_id + 1] (index 0 is the embedding output)
# while speculators uses the layer id directly.
aux_hidden_state_layer_ids = [i + 1 for i in target_layer_ids
                              if i + 1 != num_verifier_layers]
```

`[1,13,25,37,49] + 1 = [2,14,26,38,50]`. So **K = 6** (five aux plus the final
layer), `BLOCK_SIZE` stays **256**, and the extraction server needs no
reconfiguration between the DSpark and DFlash2 runs. Passing z-lab's raw ids to
`serve_extract.sh` would silently capture five layers the draft was not built
for. Confirmed on the wire against the running server:

```
hidden_states (64, 6, 6656) torch.bfloat16   token_ids match: True   finite: True
per-layer norms: [719, 1485, 2089, 2490, 4049, 5409]
GPU KV cache size: 60,392 tokens, Maximum concurrency for 49,153 tokens: 1.23x
```

### DFlash2 does log acceptance metrics

`models/dflash2/metrics.py` delegates to DSpark's `compute_unary_metrics`, so
`accept_rate`, `accept_len`, `full_acc` and `position_0..N_acc` all come
through unchanged. It adds `unary_candidate_recall_at_16`,
`unary_candidate_target_mass_at_16`, `teacher_forced_selector_acc`,
`unary_top_16_oracle_accepted_length`, and `selector_loss` / `unary_loss`.
It drops only the confidence-head metrics, which it calls with
`confidence_head_alpha=0.0`. So a DFlash2 run is monitorable exactly like run A,
with more signal, not less.

### `--from-pretrained` rejects the checkpoint

```
NotImplementedError: Loading a non-speculator model config is not supported yet.
```

`z-lab/Muse-Glimmer-30B-DFlash2` is a native z-lab checkpoint:
`architectures: ["DFlash2DraftModel"]` and **no** `speculators_model_type`.
`DaoCloud/…-DSpark` has `speculators_model_type: dspark`, which is why run A
warm-started with no conversion step. This is DFlash2-specific.

The architecture itself matches. All 81 tensors line up with speculators'
`Qwen3DFlash2DecoderLayer` and `CandidateSelector` —
`layers.*.attention_conv.{base_kernel,kernel_projection.weight}`,
`layers.*.mlp_conv.*`,
`candidate_selector.{predecessor_codebook,successor_codebook,hidden_projection.weight}`
— and q/k/v are unfused, so `_remap_weights` is a no-op. `embed_tokens` /
`lm_head` are absent by design; the converter fills them from the verifier.

So a conversion is viable and is not a fallback to scratch init.

### …but converting drops two knobs that serving still applies

This is the reason the run is stopped rather than converted.

z-lab's `dflash_config` carries `final_logit_softcapping: 20.0` and
`output_multiplier: 0.19611613513818404` (a 5.1x logit scale). Neither string
appears anywhere in `speculators` — the training path does not implement them.
vLLM's serving path **does**:

```python
# vllm/model_executor/models/qwen3_dflash2.py:275
softcap = float(draft_config.get("final_logit_softcapping") or 0.0)
self.candidate_logits_processor = LogitsProcessor(
    ..., scale=float(draft_config.get("output_multiplier", 1.0)),
    soft_cap=softcap if softcap > 0 else None)
```

and it reads them from the native `dflash_config` dict. A speculators-format
checkpoint has no such dict; vLLM rebuilds one in
`transformers_utils/configs/speculators/algos.py`, and `update_dflash2`
forwards **only** `conv_kernel_size`, `conv_group_size`, `selector_rank` and
`selector_top_k` — because `DFlash2SpeculatorConfig` has no fields for the other
two. So a converted checkpoint serves at `scale=1.0` with no softcap.

Two self-consistent positions, and they are not equivalent:

- **Drop the knobs everywhere.** Training and serving agree at `scale=1.0`, no
  softcap. No library patch. But the pretrained weights were fit for a 0.196
  scale, so step-0 loss starts high and the fine-tune spends capacity
  re-learning the output scale — devaluing the warm start the run exists for.
- **Preserve them.** Keep both keys so vLLM still applies them, and patch
  speculators' DFlash2 forward to apply the same scale and tanh softcap so
  training matches. Warm start intact; the checkpoint then depends on that
  patch at serving time.

What must not happen is preserving them on one side only: training without the
0.196 scale and serving with it (or the reverse) is a 5.1x logit mismatch, and
nothing in either stack errors on it.

### Resolved: preserve them, on both sides

Taken, and measured. Three pieces:

- `scripts/convert_dflash2.py` subclasses `DFlashConverter` and emits a
  `DFlash2SpeculatorConfig`, so the convolutions and the candidate selector are
  claimed rather than reported as unexpected keys. It reuses the parent's `i+1`
  layer remap -- but resolves `num_hidden_layers` out of the verifier's
  `text_config` first. `meta-models/Muse-Glimmer-30B` is a
  `MuseGlimmerForConditionalGeneration`; the parent reads `num_hidden_layers`
  and `hidden_size` from the top level and dies with
  `KeyError: 'num_hidden_layers'`. Its top-level `out_hidden_size` is 6144, the
  projector output, which must not be mistaken for the drafter's 6656.
- `patches/speculators-dflash2-output-shaping.patch` adds `output_multiplier`
  and `final_logit_softcapping` to `DFlash2SpeculatorConfig` and applies them in
  `DFlash2DraftModel.forward`. Baked into `specd:dflash2`
  (`docker/Dockerfile.dflash2`); `specd:latest` is the rollback.
- `docker/serve_patched.sh` forwards the two keys through vLLM's
  `update_dflash2`, sed-and-verified in place like the DSpark fix above it.

**Order and placement both matter.** vLLM's `LogitsProcessor` is not internally
consistent: `forward` and `get_top_tokens` soft-cap *then* scale, while
`get_top_k_tokens` scales *then* soft-caps. DFlash2 takes the last one --
`DFlash2Qwen3ForCausalLM.compute_candidates` calls `get_top_k_tokens` -- so the
training-side order is `tanh(x * 0.196 / 20) * 20`.

And vLLM tops-k the *raw* head output, shaping only the k selected values, so
the patch does the same: select first, shape second. The two orders are
identical in exact arithmetic (both maps are strictly increasing), but tanh
saturation ties neighbouring logits in bfloat16 and reorders the tail of the
candidate set. Shaping first changed the candidate ids on a synthetic batch;
selecting first reproduces vLLM's ids exactly, leaving only a 0.5% relative
value difference from vLLM computing in fp32 where training stays in bf16. An
fp32 copy of `[1, max_anchors * block_size, 202048]` is 13 GB, so bf16 it is.

The shaped values are what the candidate selector adds its transition scores to
(`CandidateSelector.score_candidates`: `unary_scores + transition_scores`), which
is why this is not just a loss-scale detail.

Measured at step 0, same seed, same batch, converted checkpoint, 4 ranks:

| | patched | unpatched |
|---|---:|---:|
| `train/loss` | **0.406** | 2.155 |
| `train/ce_loss` | 0.421 | 2.272 |
| `train/unary_loss` | 0.119 | 0.299 |
| `train/selector_loss` | **0.287** | 1.857 |
| `train/accept_rate` | 0.535 | 0.566 |
| `train/position_1_acc` | 0.911 | 0.909 |
| `train/position_15_acc` | 0.342 | 0.342 |

The accuracy columns barely move, which is the point: they are argmax-based and
both maps are monotonic, so the *predictions* are unchanged. What is 6.5x off
is `selector_loss`. The selector adds its transition scores to the unary
scores, and at scale 1.0 those unary scores are 5.1x larger than the ones its
codebooks were fit against, so a converged selector reads as badly
miscalibrated. That 2.155 is what "devaluing the warm start" costs in practice.

The unpatched build does not error on the converted checkpoint -- it ignores the
two unknown config keys silently, exactly as predicted.

### Run D happened, and it regressed on Terminal-Bench

Launched 2026-09-06 21:11 UTC, finished 01:05 UTC, exit 0. 3,956 steps, one
epoch, 25,000 rows at 32,768 rendered to a 10/25/30/35 reasoning-strength mix.
`specd:dflash2`, warm-started from the converted checkpoint, lr 5e-5 /
muon_lr 1e-4, warmup 0.10, checkpoint_freq 0.125.

**Training said it worked. Terminal-Bench said it did not.**

| | value |
|---|---:|
| `val/accept_len_epoch` | **8.093** |
| run A (DSpark) `val/accept_len_epoch` | 7.559 |
| train accept_len, last 200 steps | 8.159 |

| Terminal-Bench acceptance length | |
|---|---:|
| baseline | **4.02** |
| run D mid checkpoint (step 1976) | 3.95 |
| run D final (step 3956) | 3.80 |

So the training metric improved 7% over run A while the served metric fell
below baseline, monotonically with training. Those are not contradictory
measurements of the same thing — see "What training tells you, and what it does
not". The training number is teacher-forced over anchored blocks against the
verifier's own logits; the served number is real rejection sampling. A ~2x gap
between them is normal and run A showed it too. What is not normal is the
*direction*.

**The leading hypothesis is a serving-side knob drop, and it is untested.**
A converted checkpoint reaches vLLM through `update_dflash2`, which on stock
vLLM forwards only the four conv/selector fields and silently discards
`output_multiplier` and `final_logit_softcapping`. The drafter then serves at
scale 1.0 with no cap while its weights were trained at 0.196 with a cap of 20 —
and the mismatch deepens the longer it trains, which is exactly the observed
shape. The baseline does not suffer this: a native z-lab repo keeps its own
`dflash_config` and never goes through `update_dflash2`.

Check first, on the eval host:

```bash
grep -c output_multiplier \
  /usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/configs/speculators/algos.py
```

`0` means the patch was absent and the eval measured the mismatch, not the
fine-tune.

**The control that was never run.** The whole run validated the *training* path
and never once served a converted checkpoint. The missing datapoint separates
every hypothesis:

> Serve `scripts/convert_dflash2.py`'s output — converted but UNTRAINED, weights
> numerically identical to z-lab's — against the native z-lab repo on the same
> harness.
>
> - scores ≈ baseline → conversion and serving are sound; training caused the
>   regression (look at the data mix and the LR).
> - scores below baseline → the serving path is lossy; training is innocent.

Run that before changing anything else. Rebuilding it is minutes:
`convert_dflash2.py` is deterministic.

**Other candidates, if the knobs are ruled out.**

- *The data mix.* 35% xhigh against a corpus that is 10%. It was called
  eval-matched, but it was never checked against Terminal-Bench's actual
  reasoning-strength distribution. Training hard on the wrong distribution
  degrades monotonically, which fits.
- *LR / forgetting.* 5e-5 with muon 1e-4 for a full epoch on a narrow 25k set,
  starting from an already-converged drafter. Also degrades monotonically, and
  also explains mid > end.
- *Not classic overfitting.* Train and val tracked each other throughout and val
  was still improving at the end. The failure is a proxy-metric failure: the
  objective kept improving while the thing that mattered got worse.

### speculators keeps exactly ONE checkpoint on disk

`checkpoint_freq: 0.125` does not give you eight checkpoints. Every checkpoint
overwrites the same directory (`<save_path>/0`) and re-points an
`epoch0_step<N>` symlink at it, unlinking the previous one. At any moment
exactly one exists, so `save_best: false` plus replay-based selection has
nothing to select from.

`scripts/watch_and_push.sh` works around it: it copies each checkpoint aside as
the symlink appears (the symlink is written *after* the checkpoint, so its
appearance means the slot is safe to read), then pushes to HF when the trainer
exits. Run it alongside any run whose checkpoints you intend to compare.

### Artifacts

Two checkpoints survive on HF, private, 11.30 GB each — the GPU node and its
volume are gone, so these and this repo are the only copies:

- `Satgoy152/Muse-Glimmer-30B-DFlash2-Coding/run-d-32k-mix` — final, step 3956
- `Satgoy152/Muse-Glimmer-30B-DFlash2-Coding/run-d-32k-mix-step1976` — mid

Both carry `output_multiplier: 0.19611613513818404` and
`final_logit_softcapping: 20.0` in `config.json`, verified after upload. Each
folder is half `optimizer_state_dict.pt`, which is only needed to resume
training, not to serve.

`serve_patched.sh` takes a repo **root**, not a subfolder — `hf download` the
subfolder to a local path and point `SPECULATOR` at that. And it must run on
the patched vLLM, or it discards the knobs the checkpoint depends on.

### The multiplier-folding shortcut, not taken

Pre-multiplying `lm_head.weight` by 0.196 at conversion reaches the same
numbers for that one knob with no library change. It was left on the shelf
because it only covers the scale: the soft cap is not a linear map and cannot
be folded into a weight, so `final_logit_softcapping` would still need the
code path. It also silently desynchronises the checkpoint from its own config
-- the served `output_multiplier` would have to be forced to 1.0 or the scale
gets applied twice. Keep it as the fallback if the patch ever fails to rebase.

### The 32,768 data cannot be reused for a 48k run

Run A's `/mnt/data/runs/data` is 25,000 rows rendered at 32,768. Turns whose
prompt alone exceeded that window were skipped at render time
(`_render_boundary_rows`), so the rows a 49,152 window would add are not in the
set and cannot be recovered from it. Widening the window means re-rendering from
`conversations.jsonl`; the prepared rows themselves are drafter-agnostic, so
that is the only reason to re-render, not the change of drafter.

Measured re-render cost at 49,152: **~2.9 min/shard, ~93 min for all 32**, about
4,600-4,800 rows per shard (~150k total, against 125k at 32,768).

### Reasoning strength needs stratifying, not just capping

`merge_prepared.py` shuffles and caps, which preserves the corpus mix — at
32,768 that came out **medium 31.3% / high 30.9% / low 27.3% / xhigh 10.4%**,
because only 198 of the 1,981 trajectories are xhigh.
`merge_prepared_balanced.py` draws an equal quota per level instead,
redistributing any level's shortfall across the rest so the row budget is still
met. At 49,152 there are ~15k xhigh rows available against the 7,500 a balanced
30,000-row set needs, so a true four-way balance is reachable here.

It reads the level back out of the rendered system message rather than a
metadata column, so the balance is measured on what the chat template actually
honoured.

### Smaller things

- `--save-best` is `store_true` with no `--no-save-best`, so `save_best: false`
  can only be set in the YAML. Run A's config had it `true`, which is consistent
  with the missing-checkpoint hypothesis; `configs/train_dflash2.yaml` sets it
  `false` with `checkpoint_freq: 0.25`.
- `tensorboard` is **not** in `specd:latest` (wandb 0.29.0 is). A config asking
  for `logger: wandb,tensorboard` dies at startup with
  `Could not initialize TensorBoardHandler`. Installing it pulls only absl-py,
  markdown, werkzeug and tensorboard-data-server, and leaves torch 2.13.0+cu130,
  vllm 0.28.1rc1.dev437 and transformers 5.16.1 untouched — verified before
  retagging. `specd:pre-tb` is the rollback tag.
- DFlash2 and DSpark reach the same 15-token draft length differently:
  DSpark uses `block_size 15` with `sample_from_anchor: true`, z-lab's DFlash2
  uses `block_size 16` with `sample_from_anchor: false` (the dflash2 default).
  `block_size 15` at the dflash2 default drafts **14** tokens and is comparable
  to neither run A nor the 3.913 baseline.
