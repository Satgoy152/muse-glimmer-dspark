# Handoff: three eval sweeps

Paste the section below to the agent that will run them. Everything it needs is
in this repo plus the node.

---

You are running three speculative-decoding evals on a preemptible 1xH200.

**Node.** `ssh sagoyal@66.201.7.215` (IP changes on re-provision; ask if
unreachable). HF token at `/mnt/data/.hf_token`. Repo at
`/mnt/data/src/muse-glimmer-dspark` — it lags the local branch, so `git pull`
or `scp` before you rely on a script. Drafters are local dirs under
`/mnt/data/speculators/`: `dflash-official`, `dflash2`, `dflash2-run-d-32k-mix`
(final, step 3956), `dflash2-run-d-32k-mix-step1976` (midpoint),
`dspark-community`, `dspark-run-a-32k`, `dspark-run-b-49k`.

**Read first:** `docs/RESULTS.md` (all prior results and their caveats) and
`docs/RESULTS-paired.md` (the metric definitions below, and what is and is not
separable from noise).

## Gotchas that have each cost a run

- vLLM's `method` literal is **`dflash`** for DFlash *and* DFlash2 checkpoints;
  the V2 runner is selected from the checkpoint's `architectures`. Passing
  `"dflash2"` fails engine start with a pydantic literal error.
- A DSpark drafter needs the `dspark patch OK` line in the server log, and a
  DFlash2 checkpoint needs a `V2 Model Runner` line. `scripts/eval_replay.sh`
  and `scripts/eval_humaneval.sh` already assert both — do not bypass them.
- `SPEC_METHOD=none` in `docker/serve_patched.sh` serves the target alone. That
  is the no-spec control.
- `ADAPTIVE=1` adds `enable_adaptive_verification` (DSpark only). It currently
  **fails engine init**; traceback in `/mnt/data/logs/adapt_dbg.log`. Diagnose
  before using it.
- Model load is ~7 min per server start from EXT4. Budget it.
- `sagoyal` is now in the `docker` group, but a shell that predates that needs
  `sg docker -c '...'`. mini-swe-agent shells out to bare `docker`.
- Never `pkill -f` a pattern that also matches your own ssh command line. It
  kills the session.

## Metrics — use these definitions, they are what the prior numbers mean

From the server's Prometheus counters, taken as an after-minus-before delta:

```
t_step = inter_token_latency_seconds_sum / inter_token_latency_seconds_count   # _count == spec steps
TPOT   = request_decode_time_seconds_sum / (request_generation_tokens_sum - n) # == t_step / accept_len
tok/s  = request_generation_tokens_sum   / request_decode_time_seconds_sum     # decode only, prefill excluded
```

`benchmark/analysis/norm_speed.py` computes all of these over `/mnt/data/eval/*.prom`.

Client-side TTFT/TPOT are **invalid** — `--enable-auto-tool-choice` buffers
deltas until a tool call is whole. Use the server-side histograms.

Report **both** poolings, they disagree and both are real: step-weighted
(`1 + sum(accepted)/sum(steps)`, what throughput is made of) and per-request
(mean over calls of each call's accept_len). `benchmark/analysis/paired.py`
and `macro.py` do the paired bootstrap for each.

`t_step` reproduces to <0.2% across workloads at concurrency 1 and swings 5.8%
run-to-run at concurrency 10. Do not quote a throughput difference under ~6% at
concurrency 10.

## Task 1 — Terminal-Bench bucket x concurrency sweep

Frozen 1,753-call set at `/mnt/data/eval/raw.parquet`; harness
`scripts/eval_replay.sh` (`NAME`, `SPEC`, `METHOD`, `CONC`, `SRC`, `PORT`).

- Concurrency 1, 2, 8, 32. Eight configurations: no-spec, `dflash-official`,
  `dflash2`, both DFlash2 run-D checkpoints, `dspark-community`,
  `dspark-run-a-32k`, `dspark-run-b-49k`.
- Buckets `64-128`, `128-256`, `256-1K`, `>=1K` on the **original frozen
  recording's** `completion_tokens`, never the replayed output. Bucketing on the
  replay selects on the outcome and the control then swings as hard as the
  effect — see `docs/RESULTS-paired.md`. `benchmark/analysis/twoway.py` shows
  the join.
- Prefer greedy so every drafter emits identical tokens. Say so in the output:
  the existing TB rows are temperature 1.0 and are **not** comparable to greedy ones.
- Save `/metrics` before and after each run.
- Per bucket and pooled: calls, output tokens, per-request accept_len,
  step-weighted accept_len, `t_step`, decode tok/s, speedup over no-spec, TTFT,
  wall-clock throughput.
- Controls: `dflash2` twice at concurrency 1 and twice at 32.

## Task 2 — SWE-bench Multilingual

Independent of SWE-Gym, so it tests generalisation rather than the training
distribution.

- 24-40 tasks across languages and repos, disjoint from the 2,000 instances in
  `data/training/train_instances.jsonl`.
- Record target-only trajectories **once** through `scripts/proxy.py`, then
  replay the frozen calls through every drafter (`scripts/eval_replay.sh` with
  `SRC=` your converted jsonl). `scripts/swegym_holdout.sh` is a working
  end-to-end example of record-then-replay; copy its shape.
- Additionally run `dflash2`, `dflash2-run-d-32k-mix-step1976` and
  `dspark-run-a-32k` **on-policy** for resolved rate.

## Task 3 — Terminal-Bench replay repeats

Two more full replays each of `dflash2`, `dflash2-run-d-32k-mix-step1976` and
`dflash2-run-d-32k-mix`, at the existing settings (temperature 1.0, top_k 64,
concurrency 10, `NUM_SPEC_TOKENS=15`).

Purpose: the final checkpoint's regression rests on one run that drew a single
58,019-token completion. Report each run's `max(completion_tokens)` and the
pooled number with and without that call, and say whether the
midpoint-beats-final ordering survives. The paired bootstrap CI on pooled
acceptance is +-0.11, so three runs per checkpoint is the minimum that can
answer this.

## Reporting

Append to `docs/RESULTS.md` and keep its caveat discipline: quote the pooling
you used, the concurrency, the temperature, and the control alongside every
delta. A number without its control is not a result.
