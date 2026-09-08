"""Assistant-turn length distribution in the training traces vs the TB eval.

Answers one question: did the fine-tune see mostly short assistant turns?
Lengths are in tokens under the target's own tokenizer, so they are on the same
scale as the eval's completion_tokens.
"""
import json, os, statistics as st
from collections import Counter
os.environ.setdefault("HF_HOME","/mnt/data/hf")
from datasets import load_dataset
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained("meta-models/Muse-Glimmer-30B", token=open("/mnt/data/.hf_token").read().strip())
ds=load_dataset("Satgoy152/Muse-Glimmer-SWE-Gym-2k", split="train",
                token=open("/mnt/data/.hf_token").read().strip())
print(ds, flush=True)
print(ds.column_names, flush=True)
lens=[]; per_traj=[]
for i,row in enumerate(ds):
    msgs=row.get("messages")
    if isinstance(msgs,str): msgs=json.loads(msgs)
    if not msgs: continue
    n=0
    for m in msgs:
        if m.get("role")!="assistant": continue
        c=m.get("content") or ""
        if isinstance(c,list): c="".join(x.get("text","") for x in c if isinstance(x,dict))
        tc=m.get("tool_calls") or []
        for t in tc:
            c+= (t.get("function") or {}).get("arguments") or ""
        L=len(tok.encode(c, add_special_tokens=False))
        lens.append(L); n+=1
    per_traj.append(n)
    if i%200==0: print(i, len(lens), flush=True)
lens.sort()
q=lambda p: lens[int(p*(len(lens)-1))]
tot=sum(lens)
print("\nassistant turns:", len(lens), "trajectories:", len(per_traj),
      "turns/traj median", st.median(per_traj))
print("tokens: total", tot, "mean", tot/len(lens), "median", q(.5),
      "p75", q(.75), "p90", q(.9), "p99", q(.99), "max", lens[-1])
B=[(0,64),(64,256),(256,1024),(1024,4096),(4096,10**9)]
N=["<64","64-256","256-1K","1K-4K",">4K"]
print(f"{'bucket':<10}{'turns':>8}{'% turns':>9}{'tokens':>12}{'% tokens':>10}")
for (lo,hi),nm in zip(B,N):
    s=[x for x in lens if lo<=x<hi]
    print(f"{nm:<10}{len(s):8d}{len(s)/len(lens):9.1%}{sum(s):12d}{sum(s)/tot:10.1%}")
json.dump(lens, open("/mnt/data/train_turn_lens.json","w"))
