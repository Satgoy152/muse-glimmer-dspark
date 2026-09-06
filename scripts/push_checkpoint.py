"""Mirror a trained speculator checkpoint to a private HF repo.

The GPU node's volume is the only copy of a checkpoint until this runs, and the
node is disposable -- so this is the last stage of a run, not an afterthought.

Checkpoints are ~6-10 GB each (the draft's two 202,048 x 6,656 vocabulary
matrices dominate its parameter count), and `checkpoint_freq: 0.1` writes ten
per epoch. Do not push all of them: Terminal-Bench evaluation costs hours per
checkpoint, so only the one or two per run that will actually be evaluated are
worth the transfer.

One repo, one subfolder per run, so `SPECULATOR=<repo>` in
`docker/serve_patched.sh` can load a run directly by its HF id.

    uv run python scripts/push_checkpoint.py \
        --checkpoint /mnt/data/runs/checkpoints/checkpoint_best \
        --repo Satgoy152/Muse-Glimmer-30B-DSpark-SWEGym \
        --path-in-repo run-b-49k
"""

import argparse
import json
import os
from pathlib import Path

# Written next to the checkpoints by speculators; carries the resolved config,
# which is the only record of the window / block_size / anchors a run used.
RUN_ARTIFACTS = ("run.yaml", "train_command.txt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="local checkpoint directory")
    ap.add_argument("--repo", required=True, help="HF model repo id")
    ap.add_argument("--path-in-repo", required=True,
                    help="subfolder in the repo, e.g. run-b-49k")
    ap.add_argument("--public", action="store_true",
                    help="create the repo public (default: private)")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="additional files to upload alongside (logs, csv)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    ckpt = Path(args.checkpoint)
    if not ckpt.is_dir():
        raise SystemExit(f"not a directory: {ckpt}")
    weights = sorted(ckpt.glob("*.safetensors")) + sorted(ckpt.glob("*.bin"))
    if not weights:
        raise SystemExit(f"no weight shards under {ckpt}; is this a checkpoint?")
    config = ckpt / "config.json"
    if not config.is_file():
        raise SystemExit(f"missing {config}; vLLM cannot load a speculator without it")

    size = sum(p.stat().st_size for p in ckpt.rglob("*") if p.is_file())
    print(f"{ckpt}: {len(weights)} weight shard(s), {size/1e9:.1f} GB")
    print(json.dumps(json.loads(config.read_text()), indent=2)[:600])

    if args.dry_run:
        print("dry run; nothing uploaded")
        return
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        raise SystemExit("set HF_TOKEN (needs write scope) before uploading")

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=not args.public,
                    exist_ok=True)
    api.upload_folder(folder_path=str(ckpt), repo_id=args.repo,
                      repo_type="model", path_in_repo=args.path_in_repo,
                      commit_message=f"checkpoint {args.path_in_repo}")

    # The resolved config lives beside the checkpoints, not inside one.
    for name in RUN_ARTIFACTS:
        src = ckpt.parent / name
        if src.is_file():
            api.upload_file(path_or_fileobj=str(src), repo_id=args.repo,
                            repo_type="model",
                            path_in_repo=f"{args.path_in_repo}/{name}")
    for extra in args.extra:
        p = Path(extra)
        if p.is_file():
            api.upload_file(path_or_fileobj=str(p), repo_id=args.repo,
                            repo_type="model",
                            path_in_repo=f"{args.path_in_repo}/{p.name}")

    print(f"uploaded -> https://huggingface.co/{args.repo}/tree/main/{args.path_in_repo}")
    print(f"serve it with:  SPEC_METHOD=dspark SPECULATOR={args.repo} "
          f"NUM_SPEC_TOKENS=15 bash docker/serve_patched.sh")
    print("note: serve_patched.sh takes a repo root; for a subfolder, "
          "`hf download` it to a local path and pass that instead.")


if __name__ == "__main__":
    main()
