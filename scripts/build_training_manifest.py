#!/usr/bin/env python3
"""Sample ~2k SWE-Gym / SWE-bench-extra instances for trace generation, repo-disjoint from eval."""
import json, random, collections
import pyarrow.parquet as pq

SEED = 20260830
N = 2000
# SWE-Gym is 11 large mature repos (deep, long-horizon trajectories); SWE-bench-extra is a
# 1974-repo long tail (breadth). Round-robin alone would collapse to the long tail, so quota them.
QUOTA = {"SWE-Gym/SWE-Gym": 1000, "nebius/SWE-bench-extra": 1000}
CAP_FRAC = 0.05          # no repo may exceed this share of its source quota

tb = json.load(open("data/raw/tb_tasks.json"))
# conservative: exclude every repo referenced by ANY Terminal-Bench task, not just the frozen 40
excluded = {r.removesuffix(".git") for t in tb["tasks"] for r in t["repos_mentioned"]}

rows = []
for src, path in [("SWE-Gym/SWE-Gym", "data/raw/swegym.parquet"),
                  ("nebius/SWE-bench-extra", "data/raw/swebench_extra.parquet")]:
    t = pq.read_table(path, columns=["instance_id", "repo", "base_commit", "created_at"])
    for d in t.to_pylist():
        d["source"] = src
        rows.append(d)

seen, pool, dropped_leak, dropped_dup = set(), [], 0, 0
for r in rows:
    if r["repo"].lower() in excluded:
        dropped_leak += 1
        continue
    if r["instance_id"] in seen:
        dropped_dup += 1
        continue
    seen.add(r["instance_id"])
    pool.append(r)

rng = random.Random(SEED)
picked, taken = [], collections.Counter()
for src, quota in sorted(QUOTA.items()):
    by_repo = collections.defaultdict(list)
    for r in sorted((r for r in pool if r["source"] == src), key=lambda r: r["instance_id"]):
        by_repo[r["repo"]].append(r)
    for v in by_repo.values():
        rng.shuffle(v)
    cap = max(1, int(quota * CAP_FRAC))
    if cap * len(by_repo) < quota:                    # few repos -> even split instead
        cap = -(-quota // len(by_repo))
    repos = sorted(by_repo)
    rng.shuffle(repos)
    n = 0
    while n < quota:
        progressed = False
        for repo in repos:
            if n >= quota:
                break
            if by_repo[repo] and taken[repo] < cap:
                picked.append(by_repo[repo].pop())
                taken[repo] += 1
                n += 1
                progressed = True
        if not progressed:
            break

# backfill to N from whatever is left (small repos exhaust below their cap)
if len(picked) < N:
    chosen = {r["instance_id"] for r in picked}
    left = [r for r in pool if r["instance_id"] not in chosen]
    rng.shuffle(left)
    left.sort(key=lambda r: taken[r["repo"]])          # prefer under-represented repos
    for r in left[: N - len(picked)]:
        picked.append(r)
        taken[r["repo"]] += 1

picked.sort(key=lambda r: r["instance_id"])
with open("data/training/train_instances.jsonl", "w") as f:
    for r in picked:
        f.write(json.dumps(r) + "\n")

meta = {
    "name": "swe-traces-train-2k",
    "seed": SEED,
    "n": len(picked),
    "pool_size": len(pool),
    "total_instances": len(rows),
    "dropped_repo_leakage": dropped_leak,
    "dropped_duplicate_id": dropped_dup,
    "excluded_repos": sorted(excluded),
    "n_repos": len(taken),
    "by_source": dict(collections.Counter(r["source"] for r in picked)),
    "top_repos": taken.most_common(10),
}
json.dump(meta, open("data/training/train_instances_meta.json", "w"), indent=1)
print(json.dumps(meta, indent=1)[:1400])
