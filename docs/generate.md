# Trace generation

## Machine

Must be **x86_64 Linux** — the SWE-Gym images are amd64 only, and they will not run on an ARM
instance. Everything else is sizing for ~40 concurrent containers that spend most of their time
waiting on the API:

| | |
|---|---|
| vCPU | 32 |
| RAM | 128 GB (pandas and MONAI test runs are memory hungry) |
| Disk | 1 TB NVMe (~450 GB of images, plus overlay and traces) |
| OS | Ubuntu 24.04 |

The run is ~8 hours, so instance price barely matters — pick for convenience. A Nebius CPU VM keeps
this in the same account and region as the training node.

## Setup

```bash
curl -fsSL https://get.docker.com | sh
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone <repo> && cd muse-glimmer-dspark
uv sync
uv pip install mini-swe-agent
uv run python scripts/build_dataset_dir.py
export MSWEA_CONFIGURED=true
```

`build_dataset_dir.py` joins problem statements onto the manifest and writes
`data/training/swegym_2k/` as a directory `datasets.load_dataset` can read. The order is shuffled
under the fixed seed, so the slices below are stable.

## Smoke test

Run one instance before anything else. Start the proxy in its own tmux window:

```bash
OPENROUTER_API_KEY=... TRACE_DIR=data/traces/smoke uv run python scripts/proxy.py
```

```bash
uv run mini-extra swebench \
  --subset data/training/swegym_2k --split train --slice 0:1 \
  -c swebench.yaml -c configs/swegym.yaml \
  -w 1 -o runs/smoke
```

Then check the captured request, which is the only place these are observable:

```bash
head -1 data/traces/smoke/calls.jsonl | uv run python -c "import json,sys; r=json.load(sys.stdin)['request']; print({k: r.get(k) for k in ('model','temperature','top_p','top_k','provider')})"
```

`top_k` must be `64` and `chat_template_kwargs` must carry the slice's `reasoning_strength`. If
either is missing, litellm dropped it and the proxy has to set it instead — generation config has
to match serve time or the on-policy property is lost.

Reasoning effort is **not** OpenRouter's `reasoning.effort`. Muse Glimmer reads it from its chat
template as `reasoning_strength`, and vLLM accepts unknown top-level body fields without error, so
sending `reasoning.effort` fails silently: every slice renders "Reasoning strength: high." and the
four-way mix collapses to one setting with nothing in the logs to show it. Measured against the
served model, `low` averages ~530 reasoning chars and `xhigh` ~2750, so a working mix is visible
in the trace lengths.

## Run

Start the recording proxy, then the agent:

```bash
UPSTREAM_URL=https://<endpoint-host>/v1/chat/completions \
  UPSTREAM_API_KEY=<nebius-endpoint-token> \
  TRACE_DIR=data/traces/train uv run python scripts/proxy.py
```

The endpoint token is the one on the serverless endpoint's Token authentication
card, not a model key -- vLLM is started without `--api-key` and ignores the
header, so the token is only there to satisfy the tunnel gateway in front of it.
If the gateway rejects `Bearer`, override the scheme rather than editing the
proxy:

```bash
UPSTREAM_AUTH_SCHEME=Api-Key    # or: UPSTREAM_AUTH_HEADER=X-Api-Key UPSTREAM_AUTH_SCHEME=
```

The proxy forwards the request body unchanged apart from `stream: false`. It used to inject an
OpenRouter `provider` pin and `usage.include`, which are meaningless to vLLM and would have been
recorded verbatim into `calls.jsonl` — that file is the training data, so it must contain only what
the serving stack actually saw.

```bash
uv run python scripts/run_streaming.py --window 60 --workers 30 --out /data/runs/train
```

`run_streaming.py` walks the whole 2000 in the four effort segments, one harness process each, and
runs a janitor alongside that keeps a sliding window of images: it pulls ahead of the frontier and
deletes each instance's image as soon as that instance's trajectory lands. Images are ~6 GB each on
disk after dedup, so the full set is ~12 TB against a 1 TB volume and cannot be resident at once.

Deletion is by explicit image name. A blanket `docker image prune -a` on a timer races in-flight
pulls: a `-w 30` run lost 11/50 instances to `docker run` exiting 125/127 under such a loop.

Verified at `--window 6` over 12 instances: peak 7 resident images, 12/12 trajectories, 0 errors.

To run a single segment by hand instead:

```bash
uv run mini-extra swebench \
  --subset data/training/swegym_2k --split train --slice 0:600 \
  -c swebench.yaml -c configs/swegym.yaml -c configs/effort_low.yaml \
  -w 30 -o runs/train-low
```

mini-swe-agent's config is static per run, so the mix comes from separate runs rather than
per-instance sampling.

`-c swebench.yaml` must be passed explicitly — naming any config replaces the default rather than
adding to it. Configs merge recursively, so `configs/swegym.yaml` only overrides the model block.

## Export

```bash
uv run python scripts/export_traces.py data/traces/train/calls.jsonl data/export/train
```

## Notes

- Start the proxy with `ulimit -n 65535`. The default 1024 is tight once connection churn picks up.
- The agent runs on the host and only `docker exec`s bash into the task container, so the proxy on
  `127.0.0.1` is reachable. Terminal-Bench is different: its adapter installs the agent inside the
  task container, so that run needs the proxy on `0.0.0.0`, `--add-host=host.docker.internal:host-gateway`,
  and `api_base` pointed at `host.docker.internal`.
- `top_k` is passed via `extra_body` because litellm's `drop_params` strips unknown OpenAI params.
  Verified to survive to the wire in the smoke run.
- Serving is self-hosted on Nebius rather than OpenRouter. The OpenRouter route was abandoned
  because cost and rate limiting made it not worth it: 22% of calls returned 429 from DeepInfra,
  which capped throughput below what the harness could drive.
- `cost_tracking: ignore_errors` is set because litellm has no price entry for this model, so
  `cost_limit` cannot fire. `step_limit` is the only bound on a runaway trajectory.
- Sizing, measured on 50 instances at `-w 20` against OpenRouter: 16 minutes, so ~11 hours for
  2000. That number was set by DeepInfra's rate limit, not by our concurrency, so it does not
  carry over to the self-hosted endpoint -- re-measure `-w` against vLLM before sizing the run.
- Disk: ~4.7 GB per instance, since the shuffled order means consecutive instances rarely share a
  base layer. The prune loop is required, not optional — it held the run to 67 GB instead of 233 GB.

```bash
while true; do docker container prune -f >/dev/null 2>&1; docker image prune -af >/dev/null 2>&1; sleep 900; done
```

  The container prune is not optional. Instances whose container is created but never started
  leave it in `Created` state, `--rm` never fires, and a lingering container pins its image — so
  an image-only prune reports 0 B reclaimable while disk keeps climbing.
- Cost is now GPU-hours on the 8xH200 endpoint rather than per-token, so the run should be sized
  to keep the endpoint busy: idle GPU time is the expensive failure mode, not token spend.
- Run the proxy and the agent in separate tmux windows. Both outlive an ssh disconnect that way,
  and the proxy has to stay up for the whole run or capture stops silently.
- Watch `df -h` during the first hour. If disk climbs faster than expected, `docker image prune -a`
  between slices; the shared base layers are re-pulled cheaply.

## Proxy load test

`scripts/loadtest_proxy.py` runs a mock upstream in-process and hammers the proxy, so proxy
behaviour can be checked at any concurrency without spending credits:

```bash
UPSTREAM_URL=http://127.0.0.1:9099/v1/chat/completions TRACE_DIR=/tmp/lt uv run python scripts/proxy.py
uv run python scripts/loadtest_proxy.py 120 300
```

Measured after the backlog and error-handling fixes: ~200 rps at 60 concurrent, ~70 rps at 200,
with no dropped connections. Against a local endpoint where each call takes seconds, that is far
more headroom than the agent can use.
