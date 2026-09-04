import json, random
import pyarrow.parquet as pq
from pathlib import Path

SEED = 20260830
OUT = Path("data/training/swegym_2k")

t = pq.read_table("data/raw/swegym.parquet", columns=["instance_id", "problem_statement"])
statements = dict(zip(t.column("instance_id").to_pylist(),
                      t.column("problem_statement").to_pylist()))

rows = [json.loads(l) for l in open("data/training/train_instances.jsonl")]
for r in rows:
    r["problem_statement"] = statements[r["instance_id"]]

random.Random(SEED).shuffle(rows)
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "train.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"{len(rows)} instances -> {OUT}")
