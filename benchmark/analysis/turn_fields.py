import json, os
os.environ.setdefault("HF_HOME","/mnt/data/hf")
import pyarrow.parquet as pq
from datasets import load_dataset
from collections import Counter
def audit(iter_rows, title, limit=None):
    c=Counter(); ratio=[]; n=0
    for r in iter_rows:
        if r["is_error"] or not r["response"]: continue
        resp=json.loads(r["response"]) if isinstance(r["response"],str) else r["response"]
        ch=(resp.get("choices") or [{}])[0]; m=ch.get("message") or {}
        rz=m.get("reasoning") or m.get("reasoning_content") or ""
        if isinstance(rz,list): rz="".join(str(x) for x in rz)
        ct=m.get("content") or ""
        tc=m.get("tool_calls") or []
        args="".join((t.get("function") or {}).get("arguments") or "" for t in tc)
        c["turns"]+=1
        c["has reasoning"]+= bool(rz)
        c["has content"]+= bool(ct)
        c["has tool_calls"]+= bool(tc)
        c["reasoning+tool, no content"]+= bool(rz and tc and not ct)
        c["content but no tool_call"]+= bool(ct and not tc)
        c["nothing at all"]+= not (rz or ct or tc)
        c[f"finish={ch.get('finish_reason')}"]+=1
        d=(resp.get("usage") or {}).get("completion_tokens_details") or {}
        c["usage reports reasoning_tokens>0"]+= bool(d.get("reasoning_tokens"))
        cto=(resp.get("usage") or {}).get("completion_tokens") or 0
        if cto: ratio.append((len(rz)+len(ct)+len(args))/cto)
        n+=1
        if limit and n>=limit: break
    ratio.sort()
    print(f"\n== {title}")
    for k,v in c.most_common():
        print(f"   {k:<38}{v:8d}" + (f"  {v/c['turns']:7.1%}" if k!="turns" else ""))
    print(f"   chars(reasoning+content+tool_args) / completion_tokens: median {ratio[len(ratio)//2]:.2f}")
audit(pq.read_table("/mnt/data/eval/raw.parquet").to_pylist(), "EVAL: Terminal-Bench frozen 40")
raw=load_dataset("Satgoy152/Muse-Glimmer-SWE-Gym-2k","raw",split="train",token=os.environ.get("HF_TOKEN"))
audit((raw[i] for i in range(0,len(raw),20)), "TRAINING: SWE-Gym, 1-in-20 sample")
