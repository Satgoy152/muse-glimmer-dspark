import json
D=json.load(open("an_percall.json"))
C=json.load(open("an_class.json"))
# t_step, ms, measured at concurrency 1 (stable to 0.01-0.17% across workloads)
TSTEP={"dflash2":19.153,"dflash-official":19.374,
       "dflash2-run-d-32k-mix-step1976":19.155,
       "dspark-run-a-32k":20.024,"dspark-community":20.000,
       "dflash2-repeat2":19.153}
runs=["dflash2","dflash-official","dflash2-run-d-32k-mix-step1976","dspark-run-a-32k","dspark-community"]
def show(labeller, names, title):
    print(f"### {title}")
    hdr=f"{'drafter':<34}" + "".join(f"{n:>18}" for n in names)
    print(hdr)
    base={}
    for r in runs:
        a=D[r]; cells=[]
        for k in names:
            sub=[i for i in a if i in C and labeller(C[i])==k]
            st=sum(a[i]["steps"] for i in sub); ac=sum(a[i]["acc"] for i in sub)
            if not st: cells.append("-"); continue
            al=1+ac/st; tps=1000*al/TSTEP[r]
            cells.append(f"{al:.3f} / {tps:6.1f}")
        print(f"{r:<34}" + "".join(f"{c:>18}" for c in cells))
    ns=[sum(1 for i in D['dflash2'] if i in C and labeller(C[i])==k) for k in names]
    print(f"{'calls':<34}" + "".join(f"{n:>18d}" for n in ns))
    print()
def lb_ctok(c):
    v=c["ctok"]
    return "<64 tok" if v<64 else "64-256" if v<256 else "256-1K" if v<1024 else ">=1K"
def lb_arg(c):
    if not c["tool"]: return "no tool"
    v=c["arglen"]
    return "cmd <200ch" if v<200 else "cmd 200-800" if v<800 else "cmd >=800ch"
show(lb_ctok, ["<64 tok","64-256","256-1K",">=1K"],
     "accept_len / tok-s, by turn length (class from the ORIGINAL recording, so the partition is identical for every drafter)")
show(lb_arg, ["cmd <200ch","cmd 200-800","cmd >=800ch"],
     "accept_len / tok-s, by size of the bash command the turn emits")
