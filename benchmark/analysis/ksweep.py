import json
from collections import defaultdict
D=json.load(open("an_percall.json"))
def hist_pool(recs):
    H=[0]*16; S=0
    for r in recs.values():
        h=r["hist"]
        for i,v in enumerate(h): H[i]+=v
        S+=r["steps"]
    return H,S
order=["dflash2","dflash2-repeat2","dflash-official","dflash2-run-d-32k-mix-step1976",
       "dflash2-run-d-32k-mix","dspark-run-a-32k","dspark-run-b-49k","dspark-community"]
print("accept_len(K) simulated from accepted-length histogram (c10 runs)")
print("drafter".ljust(36)+"".join(f"K={k}".rjust(8) for k in [1,2,3,4,5,7,9,11,15]))
for name in order:
    if name not in D: continue
    H,S=hist_pool(D[name])
    row=[]
    for K in [1,2,3,4,5,7,9,11,15]:
        acc=sum(min(i,K)*H[i] for i in range(16))
        row.append(1+acc/S)
    print(name.ljust(36)+"".join(f"{v:8.3f}" for v in row))
