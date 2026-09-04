# Node bring-up

Written against a preemptible Nebius node: 1x H200 (143 GB), 16 vCPU, 196 GB
RAM, 1.2 TB root disk, plus a 279 GB attached disk. Adjust `SEQ_LENGTH`, `TP`
and `NPROC` for a bigger node; everything else carries over.

The root disk does not survive re-provisioning. Everything the run cannot cheaply
redo — traces, prepared dataset, checkpoints — lives on the attached disk.

## 1. Persistent disk

```bash
sudo mkfs.ext4 -L specdata /dev/vdc     # only if `sudo blkid /dev/vdc` is empty
sudo mkdir -p /mnt/data
echo "LABEL=specdata /mnt/data ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
sudo mount -a && sudo chown -R $USER:$USER /mnt/data
```

`nofail` matters: without it a node that comes back before the volume attaches
drops to emergency mode instead of booting.

## 2. Sources and image

```bash
mkdir -p /mnt/data/{runs,hf,src}
git clone https://github.com/vllm-project/speculators.git /mnt/data/src/speculators
# plus this repo at /mnt/data/src/muse-glimmer-dspark

mkdir -p /mnt/data/build && cd /mnt/data/build
cp /mnt/data/src/muse-glimmer-dspark/docker/Dockerfile.train .
cp -r /mnt/data/src/speculators speculators && rm -rf speculators/.git
sudo docker build -f Dockerfile.train -t specd:latest .
```

One image runs the extraction server and the trainer, so their torch,
transformers and vLLM are provably the same. See the Dockerfile for why the pip
order is what it is.

## 3. Weights and data

```bash
HF_HOME=/mnt/data/hf hf download meta-models/Muse-Glimmer-30B   # ~56 GB
```

The training dataset is public and the pipeline fetches it on its first stage;
no HF token is needed for either. Pulling unauthenticated is rate-limited but
works.

## 4. Run

`run.env` holds everything both units read, so the window is set in one place:

```bash
cat > /mnt/data/runs/run.env <<'ENV'
SEQ_LENGTH=32768
TP=1
NPROC=1
GPU_MEM_UTIL=0.92
MAX_NUM_SEQS=8
PREP_WORKERS=8
MAX_SAMPLES=50000
SHARDS=32
DATA_ROOT=/mnt/data
HF_HOME=/mnt/data/hf
ENV

sudo cp /mnt/data/src/muse-glimmer-dspark/deploy/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vllm-extract dspark-pipeline
```

Both units are `Restart=always` with `StartLimitIntervalSec=0`, and both are
`WantedBy=multi-user.target`, so a preempted node that comes back resumes on its
own. `RequiresMountsFor=/mnt/data` keeps them from starting against a missing
volume and rebuilding state on the ephemeral root disk.

## What survives a preemption

`scripts/pipeline.sh` marks each finished stage under `/mnt/data/runs/state`:

| stage | marker | on restart |
|---|---|---|
| conversations.jsonl | `01-conversations.done` | skipped |
| shards | `02-shards.done` | skipped |
| prepare-data | `03-shard-NNN.done` each | redoes only the shard in flight |
| merge + verify | `04-merged.done` | skipped |
| train | none | resumes from `checkpoints/` |

Sharding exists for that fourth row. `prepare-data` has no partial progress —
it skips a run whose output directory already holds `*.arrow` and otherwise
starts over — so one shard is the unit of loss. At 32 shards a preemption costs
at most ~3% of the render.

Training needs no marker: speculators resumes from the newest checkpoint unless
`--no-resume-from-checkpoint` is passed, and `checkpoint_freq: 0.1` in the
config writes ten per epoch.

## Watching it

```bash
journalctl -u dspark-pipeline -f
sudo docker logs -f vllm-extract 2>&1 | grep -iE "error|Traceback"
ls /mnt/data/runs/state/            # which stages are done
nvidia-smi
```
