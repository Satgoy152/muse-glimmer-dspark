#!/usr/bin/env python3
"""Freeze a 40-task Terminal-Bench eval subset, stratified by difficulty x category."""
import json, random, collections

SEED = 20260830
N = 40
CANON = {"games": "games", "game": "games", "math": "mathematics", "mathematics": "mathematics",
         "file_operations": "file-operations", "file-operations": "file-operations"}

d = json.load(open("data/raw/tb_tasks.json"))
tasks = [t for t in d["tasks"] if t["difficulty"] in ("easy", "medium", "hard")]
for t in tasks:
    t["category_norm"] = CANON.get(t["category"], t["category"])

rng = random.Random(SEED)
# proportional allocation over difficulty
counts = collections.Counter(t["difficulty"] for t in tasks)
alloc = {}
for k, v in counts.items():
    alloc[k] = round(N * v / len(tasks))
while sum(alloc.values()) != N:                       # fix rounding drift
    k = max(alloc, key=lambda k: counts[k] / max(alloc[k], 1))
    alloc[k] += 1 if sum(alloc.values()) < N else -1

picked = []
for diff, k in sorted(alloc.items()):
    pool = [t for t in tasks if t["difficulty"] == diff]
    # round-robin over categories so no single category dominates a difficulty band
    by_cat = collections.defaultdict(list)
    for t in sorted(pool, key=lambda t: t["task_id"]):
        by_cat[t["category_norm"]].append(t)
    for v in by_cat.values():
        rng.shuffle(v)
    cats = sorted(by_cat)
    rng.shuffle(cats)
    while k > 0:
        progressed = False
        for c in cats:
            if k == 0:
                break
            if by_cat[c]:
                picked.append(by_cat[c].pop())
                k -= 1
                progressed = True
        if not progressed:
            break

picked.sort(key=lambda t: t["task_id"])
out = {
    "name": "terminal-bench-frozen-40",
    "seed": SEED,
    "source_repo": d["repo"],
    "source_tree_sha": d["tree_sha"],
    "n": len(picked),
    "difficulty": dict(collections.Counter(t["difficulty"] for t in picked)),
    "category": dict(collections.Counter(t["category_norm"] for t in picked)),
    "excluded_repos": sorted({r for t in picked for r in t["repos_mentioned"]}),
    "tasks": [{"task_id": t["task_id"], "difficulty": t["difficulty"],
               "category": t["category_norm"], "repos_mentioned": t["repos_mentioned"]}
              for t in picked],
}
json.dump(out, open("data/benchmark/terminal_bench_frozen40.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("n", "difficulty", "category", "excluded_repos")}, indent=1))
