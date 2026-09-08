import re, os, json, sys, glob
def load(p):
    d={}
    for line in open(p):
        if line.startswith("#") or not line.strip(): continue
        m=re.match(r"^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-\deE.+]+|NaN)$", line.strip())
        if not m: continue
        k=m.group(1)
        if "bucket" in k: continue
        try: v=float(m.group(3))
        except: continue
        d[k]=d.get(k,0.0)+v
    return d
def metrics(after, before=None):
    a=load(after); b=load(before) if before and os.path.exists(before) else {}
    g=lambda k: a.get(k,0.0)-b.get(k,0.0)
    n     = g("vllm:e2e_request_latency_seconds_count")
    gen   = g("vllm:request_generation_tokens_sum")
    e2e   = g("vllm:e2e_request_latency_seconds_sum")
    ttft  = g("vllm:time_to_first_token_seconds_sum")
    pre   = g("vllm:request_prefill_time_seconds_sum")
    dec   = g("vllm:request_decode_time_seconds_sum")
    itl_s = g("vllm:inter_token_latency_seconds_sum")
    steps = g("vllm:inter_token_latency_seconds_count")
    acc   = g("vllm:spec_decode_num_accepted_tokens_total")
    drf   = g("vllm:spec_decode_num_draft_tokens_total")
    dfts  = g("vllm:spec_decode_num_drafts_total")
    if not n or not steps: return None
    decode = dec if dec else itl_s
    al = 1 + acc/dfts*0 + (acc/steps if steps else 0) if dfts else 1+acc/steps
    al = 1 + acc/steps
    return dict(n=int(n), gen=int(gen), steps=int(steps),
                accept_len=al,
                t_step_ms=1000*itl_s/steps,
                tpot_ms=1000*decode/max(gen-n,1),
                decode_tok_s=gen/decode if decode else 0,
                prefill_s=pre, decode_s=decode, e2e_s=e2e,
                ttft_mean_ms=1000*ttft/n,
                draft_rate=acc/drf if drf else 0)
rows=[]
for p in sorted(glob.glob("/mnt/data/eval/*.prom")):
    if p.endswith(".before.prom"): continue
    name=os.path.basename(p)[:-5]
    m=metrics(p, p[:-5]+".before.prom")
    if m: rows.append((name,m))
print(f"{'run':<44}{'n':>5}{'accept':>8}{'t_step':>9}{'TPOT':>8}{'tok/s':>8}{'drate':>7}")
print(f"{'':<44}{'':>5}{'_len':>8}{'ms':>9}{'ms':>8}{'decode':>8}{'':>7}")
for name,m in rows:
    print(f"{name:<44}{m['n']:5d}{m['accept_len']:8.3f}{m['t_step_ms']:9.3f}{m['tpot_ms']:8.3f}{m['decode_tok_s']:8.1f}{m['draft_rate']:7.3f}")
json.dump({k:v for k,v in rows}, open("/mnt/data/norm_speed.json","w"), indent=1)
