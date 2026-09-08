#!/usr/bin/env python3
"""Assemble the bucket x concurrency sweep into tables.

Every server-side number is an after-minus-before delta over the cell's own
/metrics snapshots. Taking the after snapshot alone -- which is all
`eval_replay.sh` saved -- would fold warmup and every earlier cell on the same
server into the counters, and 16 cells share one server here.

Two poolings are reported side by side because they disagree and both are real:

  step-weighted  1 + sum(accepted)/sum(steps)        -- what throughput is made of
  per-request    mean over calls of 1 + acc/steps    -- the average call

Per-request excludes calls with fewer than 5 spec steps on the grounds that a
3-step call's ratio is mostly quantisation noise; that is the same cut
`benchmark/analysis/macro.py` uses, so the numbers stay comparable.
"""
import argparse, glob, hashlib, json, math, os, re
from collections import defaultdict

ROOT = "/mnt/data/eval/sweep"
BUCKETS = ["b64_128", "b128_256", "b256_1K", "bge1K"]
BLABEL = {"b64_128": "64-128", "b128_256": "128-256",
          "b256_1K": "256-1K", "bge1K": ">=1K", "full": "full set"}


def load_prom(p):
    d = {}
    if not os.path.exists(p):
        return d
    for line in open(p):
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r"^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-\deE.+]+|NaN)$", line.strip())
        if not m or "bucket" in m.group(1):
            continue
        try:
            d[m.group(1)] = d.get(m.group(1), 0.0) + float(m.group(3))
        except ValueError:
            pass
    return d


def server_side(before, after):
    a, b = load_prom(after), load_prom(before)
    g = lambda k: a.get(k, 0.0) - b.get(k, 0.0)
    n = g("vllm:e2e_request_latency_seconds_count")
    gen = g("vllm:request_generation_tokens_sum")
    dec = g("vllm:request_decode_time_seconds_sum")
    itl_s = g("vllm:inter_token_latency_seconds_sum")
    steps = g("vllm:inter_token_latency_seconds_count")
    acc = g("vllm:spec_decode_num_accepted_tokens_total")
    drf = g("vllm:spec_decode_num_draft_tokens_total")
    ttft = g("vllm:time_to_first_token_seconds_sum")
    e2e = g("vllm:e2e_request_latency_seconds_sum")
    if not n or not steps:
        return None
    decode = dec or itl_s
    return dict(n=int(n), gen=int(gen), steps=int(steps),
                accept_len_sw=1 + acc / steps,
                t_step_ms=1000 * itl_s / steps,
                tpot_ms=1000 * decode / max(gen - n, 1),
                decode_tok_s=gen / decode if decode else 0.0,
                ttft_mean_s=ttft / n, e2e_s=e2e,
                draft_rate=acc / drf if drf else 0.0)


def per_call(trace):
    """Per-call acceptance from the response metrics blocks."""
    out = {}
    if not os.path.exists(trace):
        return out
    for line in open(trace):
        if not line.strip():
            continue
        r = json.loads(line)
        resp = r["response"]
        if "error" in resp:
            continue
        sd = ((resp.get("metrics") or {}).get("speculative_decoding")) or {}
        u = resp.get("usage") or {}
        out[r["replay_of"]] = {
            "traj": r["traj"],
            "steps": sd.get("num_spec_steps", 0) or 0,
            "acc": sd.get("num_accepted_draft_tokens", 0) or 0,
            "drafted": sd.get("num_draft_tokens", 0) or 0,
            "ctok": u.get("completion_tokens", 0) or 0,
            "ptok": u.get("prompt_tokens", 0) or 0,
            "ttft": r.get("ttft"),
            "fr": (resp.get("choices") or [{}])[0].get("finish_reason"),
            # replay.py reconstructs `content` from the streamed deltas but does
            # not reassemble tool_call deltas, and this model routes ~99.5% of a
            # turn into tool_calls -- so content is usually empty and the sha
            # alone cannot prove two drafters emitted the same tokens. Greedy
            # completion_tokens is the check that actually bites: identical
            # decoding gives an identical token count on every call.
            "sha": hashlib.sha1(
                ((resp.get("choices") or [{}])[0].get("message", {}) or {})
                .get("content", "").encode()).hexdigest()[:12],
        }
    return out


def collect():
    cells = {}
    for p in sorted(glob.glob(f"{ROOT}/*/*.json")):
        if p.endswith(".before.prom"):
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        m = d.get("_meta", {})
        cell = m.get("cell")
        if not cell:
            continue
        base = p[:-5]
        srv = server_side(base + ".before.prom", base + ".prom")
        pc = per_call(m.get("trace", ""))
        macro = [1 + v["acc"] / v["steps"] for v in pc.values() if v["steps"] >= 5]
        # Calls that ran into the bucket's max_tokens. The cap exists to stop a
        # greedy repetition loop from becoming the whole bucket; how often it
        # actually bites is a property of the result and has to be shown.
        capped = sum(1 for v in pc.values() if v.get("fr") == "length")
        cells[cell] = {
            "meta": m, "client": d.get("pooled", {}), "server": srv,
            "n_calls": len(pc),
            "out_tok": sum(v["ctok"] for v in pc.values()),
            "max_ctok": max((v["ctok"] for v in pc.values()), default=0),
            "accept_len_pr": (sum(macro) / len(macro)) if macro else float("nan"),
            "n_pr": len(macro), "capped": capped,
            "percall": pc,
        }
    return cells


def fmt(x, w=8, p=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return f"{'-':>{w}}"
    return f"{x:>{w}.{p}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default="/mnt/data/eval/sweep/report.json")
    ap.add_argument("--md-out", default="/mnt/data/eval/sweep/report.md")
    a = ap.parse_args()
    cells = collect()

    # no-spec denominator, per (bucket, concurrency)
    nospec = {}
    for c, v in cells.items():
        m = v["meta"]
        if m["label"] == "nospec" and m["repeat"] == "r1" and v["server"]:
            nospec[(m["bucket"], m["concurrency"])] = v["server"]["decode_tok_s"]

    labels, concs, buckets = [], set(), set()
    for v in cells.values():
        if v["meta"]["label"] not in labels:
            labels.append(v["meta"]["label"])
        concs.add(v["meta"]["concurrency"])
        buckets.add(v["meta"]["bucket"])
    concs = sorted(concs)

    lines = []
    # `err` is not decoration. Requests carry no max_tokens, greedy decoding can
    # run away into a repetition loop, and replay.py times a call out at 900 s --
    # which a 55 tok/s no-spec server hits at ~49K tokens and a 200 tok/s drafter
    # does not. An error count that differs across drafters in the same cell
    # means the cell is not comparing the same call set.
    W = ("| drafter | calls | err | capped | out tok | max ctok | accept_len (step-w) | "
         "accept_len (per-req) | t_step ms | TPOT ms | decode tok/s | vs no-spec | "
         "TTFT mean s | wall tok/s |")
    SEP = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"

    for c in concs:
        for b in BUCKETS + ["full"]:
            rows = [(k, v) for k, v in sorted(cells.items())
                    if v["meta"]["concurrency"] == c and v["meta"]["bucket"] == b]
            if not rows:
                continue
            lines.append(f"\n### concurrency {c}, output bucket {BLABEL.get(b, b)}\n")
            lines.append(W); lines.append(SEP)
            for k, v in rows:
                s = v["server"] or {}
                den = nospec.get((b, c))
                sp = (s.get("decode_tok_s", 0) / den) if den else None
                wall = v["out_tok"] / v["meta"]["wall_s"] if v["meta"].get("wall_s") else None
                lines.append(
                    f"| `{k}` | {v['n_calls']} | {v['meta'].get('errored', 0)} | "
                    f"{v['capped']} | {v['out_tok']:,} | {v['max_ctok']:,} | "
                    f"{fmt(s.get('accept_len_sw'),1)} | {fmt(v['accept_len_pr'],1)} | "
                    f"{fmt(s.get('t_step_ms'),1)} | {fmt(s.get('tpot_ms'),1)} | "
                    f"{fmt(s.get('decode_tok_s'),1,1)} | "
                    f"{(f'{sp:.2f}x' if sp else '-')} | "
                    f"{fmt(s.get('ttft_mean_s'),1)} | "
                    f"{(f'{wall:.1f}' if wall else '-')} |")

    md = "\n".join(lines)
    open(a.md_out, "w").write(md)
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "percall"}
               for k, v in cells.items()}, open(a.json_out, "w"), indent=1, default=str)
    # per-call dump for the paired bootstrap
    json.dump({k: v["percall"] for k, v in cells.items()},
              open("/mnt/data/eval/sweep/percall.json", "w"))
    print(md)
    print(f"\n{len(cells)} cells -> {a.md_out}, {a.json_out}")


if __name__ == "__main__":
    main()
