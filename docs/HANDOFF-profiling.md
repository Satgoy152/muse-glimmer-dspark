# DFlash2 profiling handoff

Updated: 2026-09-08T00:36:00+00:00.

## Scope and authorization

User authorized SSH access, GPU profiling, and a new branch. Profile existing
DFlash2 without changing model weights. User requests very brief updates and
continuous documentation so another agent can resume. No optimization results
have been established yet.

## Locations and state

- Local repo: `/Users/sagoyal/Downloads/code/muse-glimmer-dspark`
- Branch: `codex/profile-dflash2`, based on `be58f12` (same base on node).
- Node: `ssh sagoyal@66.201.5.253`; preemptible 1x H200, 16 vCPU.
- Isolated remote worktree: `/mnt/data/src/muse-glimmer-profile`.
- Existing remote checkout: `/mnt/data/src/muse-glimmer-dspark`; contains
  unrelated untracked benchmark files. Do not overwrite or clean it.
- Local `benchmark/humaneval/analyze.py` was already untracked. Leave it alone.
- GPU was idle at start. Container `codex-profile-dflash2` is now starting
  on localhost8002; startup began around 2026-09-08 00:10:45 UTC.
- Image `specd:latest`, ID `e1f4634bce11`, vLLM
  `0.28.1rc1.dev451+g1970f3ed4`, torch `2.13.0+cu130`.
- Driver `580.173.02`, CUDA 13.0, 143771 MiB GPU memory.
- Host Nsight: `/opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/nsys`.
  This directory can be mounted read-only into the container; tested `--version`.
- Model cache: `/mnt/data/hf`; drafter `/mnt/data/speculators/dflash2`.
  Native checkpoint: BF16, 5 layers, hidden6656, vocabulary202048,
  sliding_window2048, selector_top_k16, speculative block16 (15 drafts).
- Frozen prompts: `/mnt/data/eval/raw.parquet` (1753 Terminal-Bench calls).
- Data Python: `/mnt/data/src/muse-glimmer-dspark/.venv/bin/python`.
- Intended outputs: `/mnt/data/profiles/dflash2-20260908` (created).

## Status: the profile is complete

All four cases ran, all four Nsight captures were taken and analysed. Results
and their caveats are in `docs/RESULTS-profiling-dflash2.md`; committed data is
under `benchmark/profiling/results/`. Headline: target verification is ~83% of
GPU kernel time, the whole drafter ~12%, the target LM head ~3.4%, and the two
eager draft phases under 1% combined. The GPU is 97% busy, so launch overhead
is not the bottleneck. No optimisation has been attempted.

The container has been stopped and the GPU released.

## Plan and methodological constraints

1. Prepare real short/long Terminal-Bench prompts through server `/tokenize`.
   Use exactly those rendered token IDs with `/v1/completions` to avoid chat
   parser buffering. Preserve template/tools when rendering.
2. Greedy, fixed 512 generated tokens (`ignore_eos=true`), concurrency 1 and 10.
   This is a controlled speed microbenchmark, not a quality evaluation.
3. Warm prefixes and run at least three unprofiled timed waves per case.
   Throughput = output tokens / batch makespan, NOT sum of request latencies.
4. Separate Nsight captures (`--trace=cuda,nvtx --cuda-graph-trace=node`,
   `--capture-range=cudaProfilerApi --capture-range-end=repeat`).
5. Add NVTX wrappers only to the disposable container's vLLM source. Preserve
   CUDA graphs; classify graph replay launches as target vs draft. NVTX inside
   a graph's Python capture does not replay, so subphase attribution needs
   care. Never call host NVTX duration GPU execution time.
6. CUDA kernel attribution: correlationId AND process ID to runtime launch,
   then deepest enclosing NVTX phase. For overlap, use union GPU busy time.
   Exclude idle server periods from GPU gap measurements.

## Findings so far

### Established earlier

- Prefix caching already on; no reason to claim enabling it as an improvement.
- V2 DFlash keeps query forward + candidate selection in a FULL CUDA graph.
- Draft context KV insertion and feature projection run eagerly each cycle.
  They process the current verification tokens, not the entire old context.
- Target LM head and rejection sampling are outside the target forward graph.
- Profiler overhead must be measured separately from normal inference.

### Throughput variance is acceptance variance, not system noise

First short/concurrency-1 run, three rounds, three different prompts:

| prompt | accept_len | ms per target step | tok/s |
|---|---|---|---|
| short-00 | 5.22 | 19.4 | 266 |
| short-01 | 2.05 | 19.3 | 106 |
| short-02 | 12.97 | 19.8 | 645 |

Time per target step is constant to within 3%; the 6x throughput spread is
entirely accepted-length spread. **Report ms/target-step as the system speed
metric.** Throughput = accept_len x (1000 / ms_per_step), and its variance
belongs to the drafter and the prompt, not to the serving stack.

`ignore_eos=true` inflates acceptance: forced continuation past the natural
stop drives the model into repetition, which the drafter predicts almost
perfectly (that is the 12.97). Fixed-length greedy generation is still the
right control for a *speed* microbenchmark, but accepted length measured this
way is an upper bound and must not be quoted as a quality result.

### Rendering drops prior-turn reasoning, so recorded lengths do not reproduce

`/tokenize` renders every recorded request shorter than the `prompt_tokens`
recorded when it was originally served: about 13-19% shorter for short prompts
and 22-47% for long ones. Cause, confirmed by re-rendering with the fields
removed and reproducing the rendered length exactly: assistant turns carry
`reasoning_content` (and `provider_specific_fields`) that this model's chat
template does not re-emit. For one prompt the 241-token gap was 231 tokens of
`reasoning_content` plus its delimiters.

This is benign -- the rendered prompt is exactly what the server processes
today -- but it broke the original selection logic, which filtered candidates
on recorded length and then asserted the *rendered* length fell in the same
band. `prepare_prompts.py` now prefilters on recorded length and accepts on
rendered length, skipping out-of-band candidates instead of aborting.

### vLLM bug: `/start_profile` is broken for every non-torch profiler

`AsyncLLM.__init__` assigns `self.profiler` only inside its
`profiler == "torch"` branch, although it accepts a `profiler` argument and
`start_profile`/`stop_profile` both guard on `self.profiler is not None`. With
`--profiler-config '{"profiler":"cuda"}'` the endpoint returns HTTP 500,
`'AsyncLLM' object has no attribute 'profiler'`, before the request reaches the
engine core -- where `CudaProfilerWrapper` would call `cudaProfilerStart`, the
one thing an Nsight `--capture-range=cudaProfilerApi` session waits for.

`benchmark/profiling/fix_profiler_attr.py` assigns the constructor argument
before the torch branch, in the disposable container only. It is idempotent and
fails loudly if the source shape changes. **This is an upstream bug worth
reporting**; it is not specific to this model or drafter.

`max_iterations` and `delay_iterations` do gate the CUDA backend, but
`active_iterations` does not -- it only feeds the torch profiler's schedule.
The server therefore runs with `delay_iterations=15, max_iterations=50` so each
capture is a bounded steady-state decode window rather than prefill plus the
whole 512-token wave.

## Locations added this session

- `benchmark/profiling/fix_profiler_attr.py` -- the AsyncLLM profiler fix.
- `benchmark/profiling/split_captures.py` -- one Nsight report holds every
  capture, so this cuts it into per-case windows at idle gaps and emits
  `--start-ns/--end-ns` bounds for `analyze_nsys.py`.
- `benchmark/profiling/run_all.sh` -- runs the four cases in order:
  short-c1, short-c10, long-c1, long-c10. Capture bursts follow that order.
- Results under `/mnt/data/profiles/dflash2-20260908/runs/<case>-c<n>/`.
  Superseded pre-fix output is under `superseded/`.

## Next action

Nothing is in flight. To resume:

1. Restart the server (the torch.compile cache is warm, so it is healthy in
   about 140 s):

   ```bash
   sudo docker run -d --name codex-profile-dflash2 --gpus "device=0" \
     --network host --ipc host --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
     -v /mnt/data:/mnt/data \
     -v /opt/nvidia/nsight-systems:/opt/nvidia/nsight-systems:ro \
     specd:latest bash /mnt/data/src/muse-glimmer-profile/benchmark/profiling/serve_profile.sh
   ```

2. `bash benchmark/profiling/run_all.sh`, then
   `sudo docker stop -t 300 codex-profile-dflash2` so nsys finalises.
3. `nsys export --type sqlite` each `capture.N.nsys-rep`, then
   `analyze_nsys.py capture.N.sqlite --output analysis/<case>.json`.
   `split_captures.py <sqlite> --expect 1` confirms a report holds one burst.
4. Move `/mnt/data/profiles/dflash2-20260908` aside first, or pick a new
   `PROFILE_DIR`; `run_workload.py` and `prepare_prompts.py` both refuse to
   overwrite existing output.

Open questions, if this line of work continues:

- Accepted length is where the throughput is (6x spread between prompts), but
  measuring it honestly needs natural stopping, not `ignore_eos`. That is a
  different harness than this one.
- Subphase attribution inside either full CUDA graph needs a different
  approach; NVTX captured into a graph does not replay.
- The `AsyncLLM.profiler` bug is upstream and worth reporting to vLLM.
