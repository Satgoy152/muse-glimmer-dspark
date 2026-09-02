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

- The agent runs on the host and only `docker exec`s bash into the task container, so the proxy on
  `127.0.0.1` is reachable. Terminal-Bench is different: its adapter installs the agent inside the
  task container, so that run needs the proxy on `0.0.0.0`, `--add-host=host.docker.internal:host-gateway`,
  and `api_base` pointed at `host.docker.internal`.
- `top_k` is passed via `extra_body` because litellm's `drop_params` strips unknown OpenAI params.
  Confirm it survives by checking a captured request in `calls.jsonl` before running the full set.
- Sizing: ~2000 instances at ~10 min each is ~8 hours at `-w 40`. Images are ~1.2 GB for the first
  instance of a repo and ~220 MB incremental after that, so budget ~450 GB of disk across 11 repos.
- Run the proxy and the agent in separate tmux windows. Both outlive an ssh disconnect that way,
  and the proxy has to stay up for the whole run or capture stops silently.
- Watch `df -h` during the first hour. If disk climbs faster than expected, `docker image prune -a`
  between slices; the shared base layers are re-pulled cheaply.
