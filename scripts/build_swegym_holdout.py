import json, os, random, concurrent.futures as cf
os.environ.setdefault("HF_HOME","/mnt/data/hf")
from datasets import load_dataset
import requests
REGISTRY="xingyaoww/sweb.eval.x86_64"
ACCEPT=("application/vnd.docker.distribution.manifest.v2+json,"
        "application/vnd.oci.image.manifest.v1+json")
def image_name(i): return f"{REGISTRY}.{i.replace('__','_s_')}".lower()
def exists(image):
    s=requests.Session()
    try:
        tok=s.get("https://auth.docker.io/token",params={"service":"registry.docker.io",
            "scope":f"repository:{image}:pull"},timeout=30).json()["token"]
        return s.head(f"https://registry-1.docker.io/v2/{image}/manifests/latest",
            headers={"Authorization":f"Bearer {tok}","Accept":ACCEPT},timeout=30).status_code==200
    except Exception: return False
used={json.loads(l)["instance_id"] for l in open("/mnt/data/src/muse-glimmer-dspark/data/training/train_instances.jsonl")}
trainrepos={json.loads(l)["repo"] for l in open("/mnt/data/src/muse-glimmer-dspark/data/training/train_instances.jsonl")}
ds=load_dataset("SWE-Gym/SWE-Gym", split="train", token=os.environ.get("HF_TOKEN"))
rows=[r for r in ds if r["instance_id"] not in used and r["repo"] in trainrepos]
print("held out, in a repo the fine-tune saw:", len(rows))
random.seed(20260830)
by={}
for r in rows: by.setdefault(r["repo"],[]).append(r)
for v in by.values(): v.sort(key=lambda r:r["instance_id"]); random.shuffle(v)
order=sorted(by,key=lambda k:-len(by[k]))
pool=[]; i=0
while len(pool)<40:
    prog=False
    for k in order:
        if i<len(by[k]): pool.append(by[k][i]); prog=True
        if len(pool)>=40: break
    if not prog: break
    i+=1
imgs=[image_name(r["instance_id"]) for r in pool]
with cf.ThreadPoolExecutor(16) as ex: ok=list(ex.map(exists,imgs))
pool=[r for r,g in zip(pool,ok) if g][:16]
STR=["low","medium","high","xhigh"]
out=[]
for j,r in enumerate(pool):
    out.append(dict(instance_id=r["instance_id"], repo=r["repo"], base_commit=r["base_commit"],
                    created_at=str(r["created_at"]), source="SWE-Gym/SWE-Gym",
                    image_name=image_name(r["instance_id"]),
                    problem_statement=r["problem_statement"],
                    reasoning_strength=STR[j%4]))
os.makedirs("/mnt/data/bench/swegym_holdout",exist_ok=True)
with open("/mnt/data/bench/swegym_holdout/train.jsonl","w") as f:
    for r in out: f.write(json.dumps(r)+"\n")
from collections import Counter
print("picked", len(out), Counter(r["repo"] for r in out))
print("strengths", Counter(r["reasoning_strength"] for r in out))
print("\n".join(r["image_name"] for r in out))
