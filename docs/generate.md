# Trace generation

## Machine

We generated our traces on a Nebius CPU only VM. 

| | |
|---|---|
| vCPU | 32 |
| RAM | 128 GB  |
| Disk | 1 TB NVMe  |
| OS | Ubuntu 24.04 |

## Setup

```bash
curl -fsSL https://get.docker.com | sh
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone <repo> && cd muse-glimmer-dspark
uv sync
uv pip install mini-swe-agent
export MSWEA_CONFIGURED=true
```

The pre-processed training dataset is bundled directly in `data/training/swegym_2k/train.jsonl`
as a directory `datasets.load_dataset` can read. The order is shuffled under the fixed seed,
so the slices below are stable.

## Smoke test

Run one instance before anything else. Start the proxy in its own tmux window:

```bash
UPSTREAM_URL=https://<endpoint-host>/v1/chat/completions \
  UPSTREAM_API_KEY=<nebius-endpoint-token> \
  TRACE_DIR=data/traces/smoke uv run python scripts/proxy.py
```

```bash
uv run mini-extra swebench \
  --subset data/training/swegym_2k --split train --slice 0:1 \
  -c swebench.yaml -c configs/swegym.yaml \
  -w 1 -o runs/smoke
```

Then check the captured request:

```bash
head -1 data/traces/smoke/calls.jsonl | uv run python -c "import json,sys; r=json.load(sys.stdin)['request']; print({k: r.get(k) for k in ('model','temperature','top_p','top_k')})"
```

`top_k` must be `64` and `chat_template_kwargs` must carry the slice's `reasoning_strength`. 


## Run

Start the recording proxy, then the agent:

```bash
UPSTREAM_URL=https://<endpoint-host>/v1/chat/completions \
  UPSTREAM_API_KEY=<nebius-endpoint-token> \
  TRACE_DIR=data/traces/train uv run python scripts/proxy.py
```

The endpoint token is the one on the self-hosted endpoint's Token authentication
card. See [Inference Hosting](inference_hosting.md) for how the self-hosted vLLM endpoint is configured.

The proxy forwards the request body unchanged apart from `stream: false`. It records only what
the serving stack actually saw into `calls.jsonl`, which forms the training data.

```bash
uv run python scripts/run_streaming.py --window 60 --workers 30 --out /data/runs/train
```

`run_streaming.py` walks the whole 2000 in the four effort segments, one harness process each (low, medium, high, xhigh), and
runs a janitor alongside that keeps a sliding window of images. Images are ~6 GB each on
disk after dedup, so the full set is ~12 TB against a 1 TB volume and cannot be resident at once.

To run a single segment by hand instead:

```bash
uv run mini-extra swebench \
  --subset data/training/swegym_2k --split train --slice 0:600 \
  -c swebench.yaml -c configs/swegym.yaml -c configs/effort_low.yaml \
  -w 30 -o runs/train-low
```


`-c swebench.yaml` must be passed explicitly — naming any config replaces the default rather than
adding to it. 

## Export

```bash
uv run python scripts/export_traces.py data/traces/train/calls.jsonl data/export/train
```

## Notes

- The agent runs on the host and only `docker exec`s bash into the task container, so the proxy on
  `127.0.0.1` is reachable. Terminal-Bench is different: its adapter pip-installs mini-swe-agent
  *inside* the task container and runs it there, so the proxy must bind `0.0.0.0` and the container
  needs a route to it. Two things measured on the box, both of which contradict the obvious approach:

  - `host.docker.internal` does **not** resolve inside a task container, and Terminal-Bench has no
    `extra_hosts` support anywhere in the package -- tasks come up under per-task docker-compose,
    so there is nowhere to pass `--add-host`. Point `api_base` at the docker bridge gateway
    (`172.17.0.1`, confirm with `docker network inspect bridge`), which containers reach with no
    host configuration. Verified: a container reached the proxy there and got a completion back.
  - Terminal-Bench's mini-swe-agent adapter forwards only `MSWEA_CONFIGURED` and one API key into
    the container, then runs `mini -m MODEL -t INSTRUCTION -y --exit-immediately`. There is no hook
    for `api_base` or sampling params, so a run using it as-is would silently use default sampling
    and lose `top_k` and `reasoning_strength`. Subclass it and override `_env`, selected with
    `--agent-import-path module:Class`.

  Give that run its own proxy port and its own `TRACE_DIR`. Terminal-Bench is the eval set; its
  calls must never land in the training trace file.
- `top_k` is passed via `extra_body` because litellm's `drop_params` strips unknown OpenAI params.
- Serving is self-hosted on Nebius rather than OpenRouter. The OpenRouter route was abandoned
  because cost and rate limiting made it not worth it.
- Disk usage depends on sharing. The prune loop is required.

```bash
while true; do docker container prune -f >/dev/null 2>&1; docker image prune -af >/dev/null 2>&1; sleep 900; done
```

## Proxy load test

`scripts/loadtest_proxy.py` runs a mock upstream in-process and hammers the proxy, so proxy
behaviour can be checked:

```bash
UPSTREAM_URL=http://127.0.0.1:9099/v1/chat/completions TRACE_DIR=/tmp/lt uv run python scripts/proxy.py
uv run python scripts/loadtest_proxy.py 120 300
```

## Architecture

```text
+-----------------------------------------------------------------------------------------+
|                                    Nebius CPU VM                                        |
|                                                                                         |
|   +---------------------------------------------------------------------------------+   |
|   |                         mini-swe-agent Harness / Runner                         |   |
|   |                              (scripts/run_streaming.py)                         |   |
|   +---------------------------------------+-----------------------------------------+   |
|                      |                    |                                             |
|         docker exec  |                    | HTTP POST (localhost:8080)                  |
|         bash session |                    v                                             |
|                      |        +-------------------------+                               |
|                      |        |     Recording Proxy     |                               |
|                      |        |   (scripts/proxy.py)    |                               |
|                      |        +------------+------------+                               |
|                      |                     |                                            |
|                      |                     | writes calls.jsonl                         |
|                      |                     v                                            |
|                      |        +-------------------------+                               |
|                      |        |   data/traces/train/    |                               |
|                      |        +-------------------------+                               |
|                      v                                                                  |
|   +-------------------------------------+                                               |
|   |     Docker Sandbox Containers       |                                               |
|   |   +-------------+ +-------------+   |                                               |
|   |   | Instance #1 | | Instance #2 |   |                                               |
|   |   | (repo env)  | | (repo env)  |   |                                               |
|   |   +-------------+ +-------------+   |                                               |
|   +-------------------------------------+                                               |
+--------------------------------------------|--------------------------------------------+
                                             |
                                             | HTTPS / Token Auth
                                             v
+-----------------------------------------------------------------------------------------+
|                                  Nebius GPU Node (1x H200)                              |
|                                                                                         |
|                          +---------------------------------------+                      |
|                          |             vLLM Endpoint             |                      |
|                          |    meta-models/Muse-Glimmer-30B       |                      |
|                          |   + DFlash drafter speculation (x15)  |                      |
|                          +---------------------------------------+                      |
+-----------------------------------------------------------------------------------------+
```
