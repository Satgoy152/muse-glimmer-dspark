import json, math
D=json.load(open("an_percall.json")); C=json.load(open("an_class.json"))
def pb(v): return 0 if v<8192 else 1 if v<32768 else 2      # prompt tokens
def cb(v): return 0 if v<64 else 1 if v<256 else 2 if v<1024 else 3
PN=["ctx <8K","ctx 8-32K","ctx >32K"]; CN=["out <64","out 64-256","out 256-1K","out >=1K"]
def grid(A,B,label):
    a,b=D[A],D[B]
    # both conditions come from the CALL ITSELF in each run; prompt length is
    # identical across runs by construction, output length is matched so the
    # cell is not selected on one side's outcome
    ids=[i for i in sorted(set(a)&set(b)) if a[i]["steps"]>=5 and b[i]["steps"]>=5
         and cb(a[i]["ctok"])==cb(b[i]["ctok"])]
    print(f"== {label}   (micro accept_len delta, A-B)")
    print(f"{'':<12}"+"".join(f"{n:>16}" for n in CN))
    for p in range(3):
        row=[]
        for c in range(4):
            sub=[i for i in ids if pb(a[i]["ptok"])==p and cb(a[i]["ctok"])==c]
            if len(sub)<15: row.append(f"{'-':>16}"); continue
            ma=1+sum(a[i]["acc"] for i in sub)/sum(a[i]["steps"] for i in sub)
            mb=1+sum(b[i]["acc"] for i in sub)/sum(b[i]["steps"] for i in sub)
            row.append(f"{ma-mb:+7.3f} (n={len(sub):3d})")
        print(f"{PN[p]:<12}"+"".join(f"{x:>16}" for x in row))
    print()
for A,B,l in [("dflash2-run-d-32k-mix-step1976","dflash2","OURS dflash2-run-d vs dflash2"),
              ("dspark-run-a-32k","dflash-official","OURS dspark-run-a vs dflash-official"),
              ("dflash2","dflash2-repeat2","CONTROL dflash2 vs itself")]:
    grid(A,B,l)
# marginal effect of each axis on the level, not the delta
print("absolute accept_len for dflash2, to show which axis actually drives it:")
a=D["dflash2"]
print(f"{'':<12}"+"".join(f"{n:>14}" for n in CN))
for p in range(3):
    row=[]
    for c in range(4):
        sub=[i for i in a if a[i]["steps"]>=5 and pb(a[i]["ptok"])==p and cb(a[i]["ctok"])==c]
        row.append(f"{1+sum(a[i]['acc'] for i in sub)/sum(a[i]['steps'] for i in sub):6.3f} (n={len(sub):3d})" if len(sub)>=15 else "-")
    print(f"{PN[p]:<12}"+"".join(f"{x:>14}" for x in row))
