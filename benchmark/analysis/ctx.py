import json, math
D=json.load(open("an_percall.json"))
E=[2048,8192,16384,32768,65536,10**9]; N=["<2K","2-8K","8-16K","16-32K","32-64K","64K+"]
def cb(v):
    for i,e in enumerate(E):
        if v<e: return i
    return 5
def tab(A,B,label):
    a,b=D[A],D[B]
    # ptok is identical across runs for the same replay_of (same prompt), so no
    # selection bias here -- unlike bucketing on completion length.
    ids=[i for i in sorted(set(a)&set(b)) if a[i]["steps"]>=5 and b[i]["steps"]>=5]
    print(f"== {label}   (bucket by prompt tokens; prompt is identical across runs)")
    print("  ctx        n    microA microB    d     macroA macroB    d     win%")
    for j in range(6):
        sub=[i for i in ids if cb(a[i]["ptok"])==j]
        if len(sub)<10: continue
        mia=1+sum(a[i]["acc"] for i in sub)/sum(a[i]["steps"] for i in sub)
        mib=1+sum(b[i]["acc"] for i in sub)/sum(b[i]["steps"] for i in sub)
        xa=[1+a[i]["acc"]/a[i]["steps"] for i in sub]; xb=[1+b[i]["acc"]/b[i]["steps"] for i in sub]
        n=len(sub); w=sum(1 for u,v in zip(xa,xb) if u>v)
        print(f"  {N[j]:<8}{n:5d}  {mia:6.3f} {mib:6.3f} {mia-mib:+6.3f}   {sum(xa)/n:6.3f} {sum(xb)/n:6.3f} {sum(xa)/n-sum(xb)/n:+6.3f}  {w/n:5.1%}")
    print()
tab("dflash2-run-d-32k-mix-step1976","dflash2","OURS dflash2-run-d vs dflash2")
tab("dspark-run-a-32k","dflash-official","OURS dspark-run-a vs dflash-official")
tab("dspark-run-a-32k","dspark-community","OURS dspark-run-a vs dspark-community")
tab("dflash2","dflash2-repeat2","CONTROL dflash2 vs itself")
