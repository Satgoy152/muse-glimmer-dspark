"""Terminal-Bench, reweighted to the training corpus's turn-length mix.

The eval and the training corpus differ mainly in how long a turn is; this asks
what the same 1,753 calls would have scored if their turn-length mix matched the
one the drafter was fine-tuned on. Weights are per output-length bucket, from
the recorded SWE-Gym calls (median 93 tok) against the TB calls (median 129).
Nothing else is changed -- same prompts, same drafters, same runs.
"""
import json, math, random
D=json.load(open("an_percall.json")); C=json.load(open("an_class.json"))
random.seed(11)
# t_step, ms, measured at concurrency 1
TS={"dflash2":19.153,"dflash2-repeat2":19.153,"dflash-official":19.374,
    "dflash2-run-d-32k-mix-step1976":19.155,"dspark-run-a-32k":20.024,
    "dspark-community":20.000}
# training turn-length mix, by share of TURNS, from the raw SWE-Gym recordings
TRAIN={"<64":0.1161,"64-256":0.6912,"256-1K":0.1838,">=1K":0.0089}
def cb(v):
    if v is None: return None
    return "<64" if v<64 else "64-256" if v<256 else "256-1K" if v<1024 else ">=1K"
runs=["dflash2","dflash-official","dflash2-run-d-32k-mix-step1976",
      "dspark-run-a-32k","dspark-community","dflash2-repeat2"]
# empirical TB mix, from the ORIGINAL recording so it is one fixed partition
ids0=[i for i in C]
tbmix={}
for k in TRAIN:
    tbmix[k]=sum(1 for i in ids0 if cb(C[i]["ctok"])==k)/sum(1 for i in ids0 if cb(C[i]["ctok"]))
W={k:(TRAIN[k]/tbmix[k] if tbmix[k] else 0) for k in TRAIN}
print("turn-length mix, share of turns")
print(f"{'bucket':<10}{'TB eval':>10}{'training':>10}{'weight':>9}")
for k in TRAIN: print(f"{k:<10}{tbmix[k]:10.1%}{TRAIN[k]:10.1%}{W[k]:9.2f}")
def pooled(r, weighted):
    a=D[r]; A=S=T=0.0
    for i,v in a.items():
        if i not in C or v["steps"]<1 or not cb(C[i]["ctok"]): continue
        w=W[cb(C[i]["ctok"])] if weighted else 1.0
        A+=w*v["acc"]; S+=w*v["steps"]; T+=w*v["ctok"]
    al=1+A/S; ts=TS[r]
    return al, ts/al, 1000*al/ts   # accept_len, TPOT ms, tok/s at c1
print("\n%-34s %-27s %s" % ("drafter","as measured (TB mix)","reweighted to training mix"))
print("%-34s %9s%9s%9s   %9s%9s%9s" % ("","accept","TPOT","tok/s","accept","TPOT","tok/s"))
for r in runs:
    a1,t1,s1=pooled(r,False); a2,t2,s2=pooled(r,True)
    print(f"{r:<34}{a1:9.3f}{t1:9.3f}{s1:9.1f}   {a2:9.3f}{t2:9.3f}{s2:9.1f}")
# paired bootstrap on the reweighted accept_len
def boot(A,B):
    a,b=D[A],D[B]
    ids=[i for i in sorted(set(a)&set(b)) if i in C and cb(C[i]["ctok"]) and a[i]["steps"]>0 and b[i]["steps"]>0]
    w=[W[cb(C[i]["ctok"])] for i in ids]
    ka=[(a[i]["acc"],a[i]["steps"]) for i in ids]; kb=[(b[i]["acc"],b[i]["steps"]) for i in ids]
    f=lambda k,idx: 1+sum(w[j]*k[j][0] for j in idx)/sum(w[j]*k[j][1] for j in idx)
    n=len(ids); base=list(range(n)); d0=f(ka,base)-f(kb,base); ds=[]
    for _ in range(2000):
        idx=[random.randrange(n) for _ in range(n)]
        ds.append(f(ka,idx)-f(kb,idx))
    ds.sort(); return d0, ds[50], ds[1949]
print("\nreweighted accept_len, paired bootstrap:")
for A,B,l in [("dflash2-run-d-32k-mix-step1976","dflash2","ours dflash2-run-d vs dflash2"),
              ("dflash2-run-d-32k-mix-step1976","dflash-official","ours dflash2-run-d vs dflash-official"),
              ("dspark-run-a-32k","dflash-official","ours dspark-run-a vs dflash-official"),
              ("dspark-run-a-32k","dspark-community","ours dspark-run-a vs dspark-community"),
              ("dflash2","dflash2-repeat2","CONTROL dflash2 vs itself")]:
    d,lo,hi=boot(A,B); print(f"  {l:<44} d={d:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]")
