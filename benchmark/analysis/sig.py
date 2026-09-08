import json, random
D=json.load(open("an_percall.json")); C=json.load(open("an_class.json"))
random.seed(7)
def lb(c):
    v=c.get("ctok")
    if v is None: return "?"
    return "<64" if v<64 else "64-256" if v<256 else "256-1K" if v<1024 else ">=1K"
def boot(A,B,label,cls):
    a,b=D[A],D[B]
    ids=[i for i in sorted(set(a)&set(b))
         if i in C and lb(C[i])==cls and a[i]["steps"]>0 and b[i]["steps"]>0]
    n=len(ids)
    if n<20: print(f"  {label:<46} n={n} too few"); return
    ka=[(a[i]["acc"],a[i]["steps"]) for i in ids]; kb=[(b[i]["acc"],b[i]["steps"]) for i in ids]
    f=lambda k,idx: 1+sum(k[j][0] for j in idx)/sum(k[j][1] for j in idx)
    base=list(range(n)); pa,pb=f(ka,base),f(kb,base); ds=[]
    for _ in range(3000):
        idx=[random.randrange(n) for _ in range(n)]
        ds.append(f(ka,idx)-f(kb,idx))
    ds.sort()
    print(f"  {label:<46} n={n:4d}  {pa:.3f} vs {pb:.3f}  d={pa-pb:+.3f}  95%CI[{ds[75]:+.3f},{ds[2924]:+.3f}]")
for cls in ["<64","64-256","256-1K"]:
    print(f"turns whose ORIGINAL recording produced {cls} completion tokens:")
    for A,B,l in [("dflash2-run-d-32k-mix-step1976","dflash2","ours dflash2-run-d vs dflash2"),
                  ("dflash2-run-d-32k-mix-step1976","dflash-official","ours dflash2-run-d vs dflash-official"),
                  ("dspark-run-a-32k","dflash2","ours dspark-run-a vs dflash2"),
                  ("dspark-run-a-32k","dspark-community","ours dspark-run-a vs dspark-community"),
                  ("dflash2","dflash2-repeat2","CONTROL dflash2 vs itself")]:
        boot(A,B,l,cls)
    print()
