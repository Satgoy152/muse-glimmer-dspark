import json, random, statistics as st
D=json.load(open("an_percall.json"))
random.seed(0)
def pooled(rs, ids=None):
    a=s=0
    for k,r in rs.items():
        if ids and k not in ids: continue
        a+=r["acc"]; s+=r["steps"]
    return 1+a/s if s else float("nan")
def cmp(A,B,label):
    a,b=D[A],D[B]
    ids=sorted(set(a)&set(b))
    # paired bootstrap over calls
    ka=[(a[i]["acc"],a[i]["steps"]) for i in ids]
    kb=[(b[i]["acc"],b[i]["steps"]) for i in ids]
    pa=1+sum(x[0] for x in ka)/sum(x[1] for x in ka)
    pb=1+sum(x[0] for x in kb)/sum(x[1] for x in kb)
    n=len(ids); deltas=[]
    for _ in range(2000):
        idx=[random.randrange(n) for _ in range(n)]
        aa=1+sum(ka[i][0] for i in idx)/sum(ka[i][1] for i in idx)
        bb=1+sum(kb[i][0] for i in idx)/sum(kb[i][1] for i in idx)
        deltas.append(aa-bb)
    deltas.sort()
    lo,hi=deltas[50],deltas[1949]
    # per-call win rate (calls with >=5 steps)
    w=l=0
    for i in ids:
        if a[i]["steps"]<5 or b[i]["steps"]<5: continue
        x=a[i]["acc"]/a[i]["steps"]; y=b[i]["acc"]/b[i]["steps"]
        if x>y: w+=1
        elif x<y: l+=1
    # per-trajectory
    tw=tl=0
    trajs=set(a[i]["traj"] for i in ids)
    for t in trajs:
        sub=[i for i in ids if a[i]["traj"]==t]
        x=1+sum(a[i]["acc"] for i in sub)/max(1,sum(a[i]["steps"] for i in sub))
        y=1+sum(b[i]["acc"] for i in sub)/max(1,sum(b[i]["steps"] for i in sub))
        if x>y: tw+=1
        else: tl+=1
    print(f"{label}\n  n={n}  {pa:.4f} vs {pb:.4f}  delta={pa-pb:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]"
          f"  call-win {w}/{w+l} ({w/(w+l):.1%})  traj-win {tw}/{tw+tl}")
cmp("dspark-run-a-32k","dspark-community","OURS dspark-run-a-32k  vs  dspark-community (its warm start)")
cmp("dflash2-run-d-32k-mix-step1976","dflash-official","OURS dflash2-run-d-step1976  vs  dflash-official (baseline)")
cmp("dspark-run-a-32k","dflash-official","OURS dspark-run-a-32k  vs  dflash-official (baseline)")
cmp("dflash2-run-d-32k-mix-step1976","dflash2","OURS dflash2-run-d-step1976  vs  dflash2 (best baseline)")
cmp("dflash2","dflash2-repeat2","CONTROL dflash2 vs dflash2 repeat (noise floor)")
