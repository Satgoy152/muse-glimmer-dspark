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

`top_k` must be `64`. If it is missing, litellm dropped it and the proxy has to set it instead —
generation config has to match serve time or the on-policy property is lost.

## Run

Start the recording proxy, then the agent:

```bash
OPENROUTER_API_KEY=... TRACE_DIR=data/traces/train uv run python scripts/proxy.py
```

```bash
uv run mini-extra swebench \
  --subset data/training/swegym_2k --split train --slice 0:600 \
  -c swebench.yaml -c configs/swegym.yaml \
  -c model.model_kwargs.reasoning_effort=low \
  -w 40 -o runs/train-low
```

Repeat for the remaining slices to get the reasoning mix: `600:1200` medium, `1200:1800` high,
`1800:2000` xhigh. mini-swe-agent's config is static per run, so the mix comes from separate runs
rather than per-instance sampling.

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
- The provider is pinned to `deepinfra/bf16`. Parasail no longer serves this model on OpenRouter;
  of the four remaining providers DeepInfra is the only one declaring bf16 rather than `unknown`.
  Pricing is $0.30/M in, $1.20/M out, $0.04/M cache read.
- `cost_tracking: ignore_errors` is set because litellm has no price entry for this model, so
  `cost_limit` cannot fire. `step_limit` is the only bound on a runaway trajectory.
- Sizing, measured on 50 instances at `-w 20`: 16 minutes, so ~11 hours for 2000. Raising `-w`
  does not help — 22% of calls already return 429 from DeepInfra, so the provider's rate limit is
  the ceiling, not our concurrency. litellm retries with backoff and trajectories still complete.
- Disk: ~4.7 GB per instance, since the shuffled order means consecutive instances rarely share a
  base layer. The prune loop is required, not optional — it held the run to 67 GB instead of 233 GB.

```bash
while true; do docker image prune -af >/dev/null 2>&1; sleep 900; done
```
- Cost: $0.067 per trajectory at `step_limit: 100`, so ~$135 for 2000.
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
