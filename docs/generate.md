# Trace generation

## Setup

```bash
uv sync
uv pip install mini-swe-agent
uv run python scripts/build_dataset_dir.py
```

`build_dataset_dir.py` joins problem statements onto the manifest and writes
`data/training/swegym_2k/` as a directory `datasets.load_dataset` can read. The order is shuffled
under the fixed seed, so the slices below are stable.

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
