#!/usr/bin/env python3
"""Pick the SWE-bench Multilingual holdout for the transfer eval.

The claim this set has to support is "outside the training distribution", so
disjointness is checked at BOTH levels and the check is printed, not assumed:

  * instance level -- no instance_id in data/training/train_instances.jsonl
  * repository level -- no repo in that file either. Instance-level
    disjointness alone would leave the fine-tune having seen hundreds of other
    issues in the same codebase, which is not independence.

Selection is round-robin over (language, repo) so no single repo or language
dominates a 32-task sample, and an instance is only kept if its evaluation image
actually resolves on Docker Hub -- a missing image fails hours later, inside the
agent run, one task at a time.
"""
import argparse, json, os, random, concurrent.futures as cf
from collections import Counter, defaultdict

os.environ.setdefault("HF_HOME", "/mnt/data/hf")
import requests
from datasets import load_dataset

ACCEPT = ("application/vnd.docker.distribution.manifest.v2+json,"
          "application/vnd.oci.image.manifest.v1+json,"
          "application/vnd.docker.distribution.manifest.list.v2+json,"
          "application/vnd.oci.image.index.v1+json")


# The dataset ships the evaluation image name in its `image` column; only fall
# back to constructing one if a row is missing it.
def image_name(row):
    return (row.get("image") or
            f"swebench/sweb.eval.x86_64.{row['instance_id'].replace('__', '_1776_')}:latest").lower()


# There is no `language` column. `log_parser` names the build tool, which is a
# one-to-one stand-in for the language family in this dataset.
LANG = {"parse_log_maven": "java", "parse_log_gradle": "java",
        "parse_log_go": "go", "parse_log_cargo": "rust",
        "parse_log_jest": "javascript", "parse_log_mocha": "javascript",
        "parse_log_vitest": "javascript", "parse_log_npm": "javascript",
        "parse_log_phpunit": "php", "parse_log_rspec": "ruby",
        "parse_log_minitest": "ruby", "parse_log_ctest": "cpp",
        "parse_log_cmake": "cpp", "parse_log_make": "c"}


def lang(row):
    return LANG.get(row.get("log_parser", ""), row.get("log_parser", "?"))


def exists(image):
    s = requests.Session()
    try:
        tok = s.get("https://auth.docker.io/token",
                    params={"service": "registry.docker.io",
                            "scope": f"repository:{image}:pull"}, timeout=30).json()["token"]
        r = s.head(f"https://registry-1.docker.io/v2/{image}/manifests/latest",
                   headers={"Authorization": f"Bearer {tok}", "Accept": ACCEPT}, timeout=30)
        return r.status_code == 200
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/mnt/data/src/muse-glimmer-dspark/data/training/train_instances.jsonl")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--out", default="/mnt/data/bench/swebench_ml/test.jsonl")
    a = ap.parse_args()

    train = [json.loads(l) for l in open(a.train)]
    used_ids = {r["instance_id"] for r in train}
    used_repos = {r["repo"] for r in train}
    print(f"training set: {len(train)} instances over {len(used_repos)} repos")

    ds = load_dataset("swe-bench/SWE-Bench_Multilingual", split="test",
                      token=open("/mnt/data/.hf_token").read().strip())
    rows = list(ds)
    print(f"SWE-bench Multilingual: {len(rows)} instances, "
          f"{len(set(r['repo'] for r in rows))} repos, "
          f"{len(set(lang(r) for r in rows))} language families")

    # the disjointness check, stated rather than assumed
    id_overlap = {r["instance_id"] for r in rows} & used_ids
    repo_overlap = {r["repo"] for r in rows} & used_repos
    print(f"instance-id overlap with training: {len(id_overlap)}  {sorted(id_overlap)[:5]}")
    print(f"repository overlap with training : {len(repo_overlap)}  {sorted(repo_overlap)[:5]}")
    rows = [r for r in rows if r["instance_id"] not in used_ids and r["repo"] not in used_repos]
    print(f"after enforcing both: {len(rows)} candidates")

    by = defaultdict(list)
    for r in rows:
        by[(lang(r), r["repo"])].append(r)
    for v in by.values():
        v.sort(key=lambda r: r["instance_id"])
    random.seed(20260908)
    keys = sorted(by)
    random.Random(20260908).shuffle(keys)
    # round-robin over (language, repo): take the k-th instance of every key
    # before taking anyone's (k+1)-th.
    pool, i = [], 0
    while len(pool) < a.n * 2:
        prog = False
        for k in sorted(keys, key=lambda k: (k[0], k[1])):
            if i < len(by[k]):
                pool.append(by[k][i]); prog = True
            if len(pool) >= a.n * 2:
                break
        if not prog:
            break
        i += 1

    print(f"checking {len(pool)} images on Docker Hub...")
    with cf.ThreadPoolExecutor(16) as ex:
        ok = list(ex.map(exists, [image_name(r).rsplit(":", 1)[0] for r in pool]))
    pool = [r for r, g in zip(pool, ok) if g]
    print(f"{len(pool)} of them have an image")

    # re-round-robin the survivors so the final n stays language-balanced
    by2 = defaultdict(list)
    for r in pool:
        by2[(lang(r), r["repo"])].append(r)
    sel, i = [], 0
    while len(sel) < a.n:
        prog = False
        for k in sorted(by2):
            if i < len(by2[k]):
                sel.append(by2[k][i]); prog = True
            if len(sel) >= a.n:
                break
        if not prog:
            break
        i += 1

    STR = ["low", "medium", "high", "xhigh"]
    sel.sort(key=lambda r: r["instance_id"])
    out = []
    for j, r in enumerate(sel):
        out.append(dict(instance_id=r["instance_id"], repo=r["repo"],
                        base_commit=r["base_commit"],
                        language=lang(r), log_parser=r.get("log_parser", ""),
                        created_at=str(r.get("created_at", "")),
                        source="swe-bench/SWE-Bench_Multilingual",
                        image_name=image_name(r),
                        problem_statement=r["problem_statement"],
                        reasoning_strength=STR[j % 4]))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"\npicked {len(out)} -> {a.out}")
    print("languages:", dict(Counter(r["language"] for r in out)))
    print("repos    :", dict(Counter(r["repo"] for r in out)))
    print("strengths:", dict(Counter(r["reasoning_strength"] for r in out)))
    json.dump({"n": len(out), "instance_ids": [r["instance_id"] for r in out],
               "repos": sorted(set(r["repo"] for r in out)),
               "languages": dict(Counter(r["language"] for r in out)),
               "train_repos": sorted(used_repos),
               "instance_overlap": sorted(id_overlap),
               "repo_overlap": sorted(repo_overlap)},
              open(os.path.dirname(a.out) + "/selection.json", "w"), indent=1)


if __name__ == "__main__":
    main()
