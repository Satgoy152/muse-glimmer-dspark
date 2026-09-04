import collections, json, random
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
import requests

SEED = 20260830
N = 2000
REGISTRY = "xingyaoww/sweb.eval.x86_64"
ACCEPT = ("application/vnd.docker.distribution.manifest.v2+json,"
          "application/vnd.oci.image.manifest.v1+json")


def image_name(instance_id):
    return f"{REGISTRY}.{instance_id.replace('__', '_s_')}".lower()


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
    tb = json.load(open("data/raw/tb_tasks.json"))
    excluded = {r.removesuffix(".git") for t in tb["tasks"] for r in t["repos_mentioned"]}

    t = pq.read_table("data/raw/swegym.parquet",
                      columns=["instance_id", "repo", "base_commit", "created_at"])
    rows = [r for r in t.to_pylist() if r["repo"].lower() not in excluded]
    for r in rows:
        r["source"] = "SWE-Gym/SWE-Gym"
        r["image_name"] = image_name(r["instance_id"])

    with ThreadPoolExecutor(32) as ex:
        ok = list(ex.map(exists, [r["image_name"] for r in rows]))
    pool = [r for r, good in zip(rows, ok) if good]

    rng = random.Random(SEED)
    by_repo = collections.defaultdict(list)
    for r in sorted(pool, key=lambda r: r["instance_id"]):
        by_repo[r["repo"]].append(r)
    for v in by_repo.values():
        rng.shuffle(v)

    repos = sorted(by_repo)
    rng.shuffle(repos)
    picked = []
    while len(picked) < N:
        progressed = False
        for repo in repos:
            if len(picked) >= N:
                break
            if by_repo[repo]:
                picked.append(by_repo[repo].pop())
                progressed = True
        if not progressed:
            break

    picked.sort(key=lambda r: r["instance_id"])
    with open("data/training/train_instances.jsonl", "w") as f:
        for r in picked:
            f.write(json.dumps(r) + "\n")

    meta = {
        "name": "swe-gym-traces-2k",
        "seed": SEED,
        "n": len(picked),
        "total_instances": len(rows),
        "image_verified_pool": len(pool),
        "dropped_missing_image": len(rows) - len(pool),
        "excluded_repos": sorted(excluded),
        "by_repo": dict(collections.Counter(r["repo"] for r in picked)),
    }
    json.dump(meta, open("data/training/train_instances_meta.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
