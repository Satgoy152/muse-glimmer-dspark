import json, os
D="/mnt/data/hf/hub/datasets--RedHatAI--speculator_benchmarks/snapshots/2ae86affa2cb97a972b7fc681dd51c04fbff083e"
doms=["tool_call","rag","qa","math_reasoning","summarization","writing","question","translation"]
out=open("/mnt/data/bench/domains_all.jsonl","w")
tc=open("/mnt/data/bench/domain_tool_call.jsonl","w")
n={}
for d in doms:
    for l in open(f"{D}/{d}.jsonl"):
        j=json.loads(l)
        r={"task_id": f"{d}/{j['question_id']}", "domain": d,
           "category": j.get("category"), "prompt": j["prompt"]}
        out.write(json.dumps(r)+"\n")
        if d=="tool_call": tc.write(json.dumps(r)+"\n")
        n[d]=n.get(d,0)+1
out.close(); tc.close()
print(n, sum(n.values()))
