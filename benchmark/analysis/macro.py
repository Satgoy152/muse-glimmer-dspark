import json, random, math
D=json.load(open("an_percall.json"))
random.seed(1)
def stats(A,B,label,minsteps=5):
    a,b=D[A],D[B]
    ids=[i for i in sorted(set(a)&set(b)) if a[i]["steps"]>=minsteps and b[i]["steps"]>=minsteps]
    xa=[1+a[i]["acc"]/a[i]["steps"] for i in ids]
    xb=[1+b[i]["acc"]/b[i]["steps"] for i in ids]
    d=[u-v for u,v in zip(xa,xb)]
    n=len(d); m=sum(d)/n
    sd=math.sqrt(sum((x-m)**2 for x in d)/(n-1)); se=sd/math.sqrt(n)
    w=sum(1 for x in d if x>0); l=sum(1 for x in d if x<0)
    # sign test p
    from math import comb
    k=min(w,l); N=w+l
    p=2*sum(comb(N,i) for i in range(k+1))/2**N if N<1200 else 2*(0.5*math.erfc(abs(w-N/2)/math.sqrt(N/4)/math.sqrt(2)))
    print(f"{label}\n  n={n} macro-mean {sum(xa)/n:.4f} vs {sum(xb)/n:.4f}  delta={m:+.4f} +-{1.96*se:.4f}"
          f"  | median {sorted(xa)[n//2]:.3f} vs {sorted(xb)[n//2]:.3f}"
          f"  | win {w}/{N} ({w/N:.1%}) sign-p={p:.2g}")
stats("dflash2-run-d-32k-mix-step1976","dflash2","OURS dflash2-run-d-step1976 vs dflash2 (best baseline)")
stats("dflash2-run-d-32k-mix-step1976","dflash-official","OURS dflash2-run-d-step1976 vs dflash-official")
stats("dspark-run-a-32k","dflash-official","OURS dspark-run-a-32k vs dflash-official")
stats("dspark-run-a-32k","dflash2","OURS dspark-run-a-32k vs dflash2")
stats("dspark-run-a-32k","dspark-community","OURS dspark-run-a-32k vs dspark-community")
stats("dflash2","dflash2-repeat2","CONTROL dflash2 vs itself")
stats("dflash-official","dflash2","dflash-official vs dflash2 (baseline vs baseline)")
