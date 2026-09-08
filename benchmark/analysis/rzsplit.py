"""Is it the reasoning prose or the bash command that acceptance turns on?

reasoning share is taken from the ORIGINAL recording, so it is one fixed
covariate per prompt, identical for every drafter. Output length is controlled
by comparing only inside a length bucket.
"""
import json
D=json.load(open("an_percall.json")); R=json.load(open("an_reason.json"))
def cb(v): return 0 if v<64 else 1 if v<256 else 2 if v<1024 else 3
CN=["out <64","out 64-256","out 256-1K","out >=1K"]
def share(v): return v["rz"]/max(1,v["rz"]+v["act"])
def level(run):
    a=D[run]; ids=[i for i in a if i in R and a[i]["steps"]>=5 and R[i]["ctok"]]
    print(f"== absolute accept_len for {run}: rows = output length, cols = reasoning share of the turn")
    qs=[0.25,0.5,0.75]
    vals=sorted(share(R[i]) for i in ids)
    cuts=[vals[int(q*(len(vals)-1))] for q in qs]
    print(f"   (reasoning-share quartile cuts: {cuts[0]:.2f} {cuts[1]:.2f} {cuts[2]:.2f})")
    def qb(s): return 0 if s<cuts[0] else 1 if s<cuts[1] else 2 if s<cuts[2] else 3
    print(f"{'':<12}"+"".join(f"{n:>16}" for n in ["Q1 least rz","Q2","Q3","Q4 most rz"]))
    for c in range(4):
        row=[]
        for q in range(4):
            sub=[i for i in ids if cb(R[i]["ctok"])==c and qb(share(R[i]))==q]
            if len(sub)<15: row.append("-"); continue
            al=1+sum(a[i]["acc"] for i in sub)/sum(a[i]["steps"] for i in sub)
            row.append(f"{al:6.3f} (n={len(sub):3d})")
        print(f"{CN[c]:<12}"+"".join(f"{x:>16}" for x in row))
    print()
level("dflash2")
level("dspark-run-a-32k")
def delta(A,B,label):
    a,b=D[A],D[B]
    ids=[i for i in sorted(set(a)&set(b)) if i in R and R[i]["ctok"]
         and a[i]["steps"]>=5 and b[i]["steps"]>=5 and cb(a[i]["ctok"])==cb(b[i]["ctok"])]
    vals=sorted(share(R[i]) for i in ids); cut=vals[len(vals)//2]
    print(f"== {label}: accept_len delta, split at the median reasoning share ({cut:.2f})")
    for c in range(4):
        line=[]
        for lo,hi,nm in [(0,cut,"low reasoning"),(cut,2,"high reasoning")]:
            sub=[i for i in ids if cb(a[i]["ctok"])==c and lo<=share(R[i])<hi]
            if len(sub)<15: line.append(f"{nm}: -"); continue
            ma=1+sum(a[i]["acc"] for i in sub)/sum(a[i]["steps"] for i in sub)
            mb=1+sum(b[i]["acc"] for i in sub)/sum(b[i]["steps"] for i in sub)
            line.append(f"{nm}: {ma-mb:+.3f} (n={len(sub)})")
        print(f"   {CN[c]:<12} " + "   ".join(line))
    print()
delta("dflash2-run-d-32k-mix-step1976","dflash2","OURS dflash2-run-d vs dflash2")
delta("dspark-run-a-32k","dflash-official","OURS dspark-run-a vs dflash-official")
delta("dflash2","dflash2-repeat2","CONTROL dflash2 vs itself")
