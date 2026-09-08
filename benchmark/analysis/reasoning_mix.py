import json, os
os.environ.setdefault("HF_HOME","/mnt/data/hf")
from datasets import load_dataset
import pyarrow.parquet as pq
def parts(m):
    rz=m.get("reasoning") or m.get("reasoning_content") or ""
    if isinstance(rz,list): rz="".join(str(x) for x in rz)
    act=m.get("content") or ""
    for t in (m.get("tool_calls") or []): act+=(t.get("function") or {}).get("arguments") or ""
    return len(rz), len(act)
def summarize(rows,title):
    import statistics as st
    sh=[r/max(1,r+a) for r,a in rows]
    sh.sort(); n=len(sh)
    q=lambda p: sh[int(p*(n-1))]
    tr=sum(r for r,_ in rows); ta=sum(a for _,a in rows)
    print(f"\n== {title}   n={n}")
    print(f"   reasoning share of generated chars: p25 {q(.25):.2f}  median {q(.5):.2f}  p75 {q(.75):.2f}")
    print(f"   corpus-wide: reasoning {tr/(tr+ta):.1%} of all generated chars")
    for lo,hi,nm in [(0,.28,"Q1 least reasoning"),(.28,.57,"Q2"),(.57,.84,"Q3"),(.84,1.01,"Q4 most reasoning")]:
        c=sum(1 for s in sh if lo<=s<hi)
        print(f"     {nm:<20}{c/n:6.1%} of turns")
raw=load_dataset("Satgoy152/Muse-Glimmer-SWE-Gym-2k","raw",split="train",token=os.environ.get("HF_TOKEN"))
tr=[]
for i in range(0,len(raw),5):
    r=raw[i]
    if r["is_error"] or not r["response"]: continue
    tr.append(parts(((json.loads(r["response"]).get("choices") or [{}])[0].get("message") or {})))
summarize(tr,"TRAINING corpus (SWE-Gym), 1-in-5 sample")
tb=[]
for r in pq.read_table("/mnt/data/eval/raw.parquet").to_pylist():
    if r["is_error"] or not r["response"]: continue
    tb.append(parts(((json.loads(r["response"]).get("choices") or [{}])[0].get("message") or {})))
summarize(tb,"EVAL (Terminal-Bench frozen 40)")
