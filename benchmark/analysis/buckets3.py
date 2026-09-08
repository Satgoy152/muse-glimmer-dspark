import json, math
D=json.load(open("an_percall.json"))
def bk(v): return 0 if v<64 else 1 if v<256 else 2 if v<1024 else 3 if v<4096 else 4
names=["<64","64-256","256-1K","1K-4K",">4K"]
def tab(A,B,label):
    a,b=D[A],D[B]
    ids=[i for i in sorted(set(a)&set(b)) if a[i]["steps"]>=5 and b[i]["steps"]>=5 and bk(a[i]["ctok"])==bk(b[i]["ctok"])]
    print(f"== {label}   (both sides in same completion-length bucket)")
    print("  bucket      n   macroA macroB   d      microA microB    d     win%")
    for j in range(5):
        sub=[i for i in ids if bk(a[i]["ctok"])==j]
        if len(sub)<5: continue
        xa=[1+a[i]["acc"]/a[i]["steps"] for i in sub]; xb=[1+b[i]["acc"]/b[i]["steps"] for i in sub]
        mia=1+sum(a[i]["acc"] for i in sub)/sum(a[i]["steps"] for i in sub)
        mib=1+sum(b[i]["acc"] for i in sub)/sum(b[i]["steps"] for i in sub)
        n=len(sub); w=sum(1 for u,v in zip(xa,xb) if u>v)
        print(f"  {names[j]:<8}{n:5d} {sum(xa)/n:6.3f} {sum(xb)/n:6.3f} {sum(xa)/n-sum(xb)/n:+6.3f}   "
              f"{mia:6.3f} {mib:6.3f} {mia-mib:+6.3f}  {w/n:5.1%}")
    mia=1+sum(a[i]["acc"] for i in ids)/sum(a[i]["steps"] for i in ids)
    mib=1+sum(b[i]["acc"] for i in ids)/sum(b[i]["steps"] for i in ids)
    print(f"  ALL     {len(ids):5d}  micro {mia:.3f} vs {mib:.3f} ({mia-mib:+.3f})\n")
tab("dflash2-run-d-32k-mix-step1976","dflash2","OURS dflash2-run-d-step1976 vs dflash2")
tab("dspark-run-a-32k","dflash-official","OURS dspark-run-a-32k vs dflash-official")
tab("dspark-run-a-32k","dflash2","OURS dspark-run-a-32k vs dflash2")
tab("dflash2","dflash2-repeat2","CONTROL dflash2 vs itself")
tab("dflash-official","dflash2","BASELINE dflash-official vs dflash2")
