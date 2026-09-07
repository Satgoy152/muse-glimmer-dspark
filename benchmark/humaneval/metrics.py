#!/usr/bin/env python3
"""Pooled acceptance for a HumanEval/MBPP generation trace, plus pass@1.

Acceptance pools exactly as benchmark/terminal_bench/metrics.py does --
1 + sum(num_accepted_draft_tokens)/sum(num_spec_steps) over the per-request
speculative_decoding blocks -- so a row here is computed the same way a
Terminal-Bench row is, even though the two are different decoding regimes and
must not be put in one table.

Two guards the Terminal-Bench work taught us:
  * max(completion_tokens) is printed, not hidden. One 58K-token runaway once
    moved a pooled acceptance number by 0.08.
  * The server counters in /metrics are an independent path to the same
    quantity. --prom-before/--prom-after take the counter delta across this
    run, rather than the raw post-run value, so a health probe or warmup
    request cannot leak into the cross-check.

pass@1 executes model-written code. It runs each program in its own subprocess
with a wall-clock timeout, which is the standard HumanEval harness, and is why
it is opt-in behind --pass1 rather than always on.
"""
import argparse, json, multiprocessing as mp, os, re, sys
from collections import Counter

FENCE = re.compile(r"```(?:[Pp]ython3?|py)?\s*\n(.*?)(?:```|\Z)", re.S)


def parse_prom(path, needle):
    """Sum the value of every sample whose metric name ends in `needle`."""
    if not path or not os.path.exists(path):
        return None
    tot = None
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("{")[0].split(" ")[0]
        if not name.endswith(needle):
            continue
        try:
            v = float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            continue
        tot = v if tot is None else tot + v
    return tot


def extract_code(text, entry_point, prompt):
    """Recover a runnable program from a chat reply.

    Three shapes show up: a fenced block holding the whole function, a fenced
    block holding only the body, and an unfenced reply. Preferring the *last*
    fenced block that defines the entry point handles replies that show a
    broken attempt first and the fix afterwards.
    """
    blocks = [b.strip() for b in FENCE.findall(text or "")]
    cands = [b for b in blocks if re.search(rf"def\s+{re.escape(entry_point)}\s*\(", b)]
    if cands:
        return cands[-1]
    if blocks:
        # a body-only block: graft it back onto the stub
        return prompt + "\n" + blocks[-1]
    if text and re.search(rf"def\s+{re.escape(entry_point)}\s*\(", text):
        return text
    return prompt + "\n" + (text or "")


def _run(program, q):
    import contextlib, io
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            g = {"__name__": "__main__"}
            exec(compile(program, "<sol>", "exec"), g)
        q.put("pass")
    except BaseException as e:
        q.put(f"fail:{type(e).__name__}")


def check_one(program, timeout):
    q = mp.Queue()
    p = mp.Process(target=_run, args=(program, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.kill(); p.join()
        return "timeout"
    try:
        return q.get_nowait()
    except Exception:
        return "fail:crash"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--name", default="")
    ap.add_argument("--task", default="HumanEval")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--prom-before", default="")
    ap.add_argument("--prom-after", default="")
    ap.add_argument("--pass1", action="store_true")
    ap.add_argument("--exec-timeout", type=float, default=15.0)
    a = ap.parse_args()

    recs, errored = [], 0
    for line in open(a.trace):
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r["response"]:
            errored += 1
            continue
        recs.append(r)

    steps = accepted = drafted = out_tok = latency = 0
    ctoks, ttfts, e2es, finish = [], [], [], Counter()
    no_sd = 0
    concs, temps = set(), set()
    for r in recs:
        resp = r["response"]
        u = resp.get("usage") or {}
        ct = u.get("completion_tokens", 0) or 0
        out_tok += ct
        ctoks.append(ct)
        latency += r.get("latency", 0) or 0
        e2es.append(r.get("latency", 0) or 0)
        if r.get("ttft") is not None:
            ttfts.append(r["ttft"])
        finish[(resp.get("choices") or [{}])[0].get("finish_reason")] += 1
        concs.add(r.get("concurrency")); temps.add(r.get("temperature"))
        sd = ((resp.get("metrics") or {}).get("speculative_decoding")) or {}
        if not sd.get("num_spec_steps"):
            no_sd += 1
            continue
        steps += sd.get("num_spec_steps", 0) or 0
        accepted += sd.get("num_accepted_draft_tokens", 0) or 0
        drafted += sd.get("num_draft_tokens", 0) or 0

    acc = 1 + accepted / steps if steps else float("nan")
    rate = accepted / drafted if drafted else float("nan")

    # independent path to the same number, from the server's own counters
    xa = parse_prom(a.prom_after, "spec_decode_num_accepted_tokens_total")
    xd = parse_prom(a.prom_after, "spec_decode_num_drafts_total")
    ba = parse_prom(a.prom_before, "spec_decode_num_accepted_tokens_total") or 0.0
    bd = parse_prom(a.prom_before, "spec_decode_num_drafts_total") or 0.0
    server_acc = delta = None
    if xa is not None and xd is not None and (xd - bd) > 0:
        server_acc = 1 + (xa - ba) / (xd - bd)
        delta = server_acc - acc

    med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")
    print(f"=== {a.name or a.trace} [{a.task}] ===")
    print(f"calls {len(recs)}  errored {errored}  no-spec-block {no_sd}  "
          f"concurrency {sorted(c for c in concs if c is not None)}  "
          f"temperature {sorted(t for t in temps if t is not None)}")
    print(f"accept_len (per-request pooled) : {acc:.4f}")
    if server_acc is not None:
        print(f"accept_len (/metrics counters)  : {server_acc:.4f}   delta {delta:+.4f}")
    else:
        print("accept_len (/metrics counters)  : n/a")
    print(f"draft_acceptance_rate           : {rate:.4f}")
    print(f"out_tok {out_tok:,}  max(completion_tokens) {max(ctoks) if ctoks else 0}  "
          f"median {med(ctoks)}")
    print(f"finish_reason {dict(finish)}")
    print(f"ttft_p50 {med(ttfts):.3f}  e2e_p50 {med(e2es):.2f}  "
          f"ttft/e2e {med(ttfts)/med(e2es) if e2es and med(e2es) else float('nan'):.3f} "
          "(near 1.0 would mean the client-side split is invalid)")

    out = {"name": a.name, "task": a.task, "trace": a.trace,
           "calls": len(recs), "errored": errored, "no_spec_block": no_sd,
           "concurrency": sorted(c for c in concs if c is not None),
           "temperature": sorted(t for t in temps if t is not None),
           "accept_len": round(acc, 4),
           "accept_len_server": round(server_acc, 4) if server_acc else None,
           "accept_len_delta": round(delta, 4) if delta is not None else None,
           "draft_rate": round(rate, 4), "out_tok": out_tok,
           "max_completion_tokens": max(ctoks) if ctoks else 0,
           "median_completion_tokens": med(ctoks),
           "finish_reason": {str(k): v for k, v in finish.items()},
           "ttft_p50": med(ttfts), "e2e_p50": med(e2es),
           "num_spec_steps": steps, "num_accepted": accepted, "num_drafted": drafted}

    if a.pass1:
        npass = 0
        outcomes = Counter()
        fails = []
        for r in recs:
            if not r.get("test"):
                continue
            msg = (r["response"].get("choices") or [{}])[0].get("message") or {}
            code = extract_code(msg.get("content") or "", r["entry_point"], r["prompt"])
            prog = code + "\n\n" + r["test"] + f"\ncheck({r['entry_point']})\n"
            res = check_one(prog, a.exec_timeout)
            outcomes[res] += 1
            if res == "pass":
                npass += 1
            elif len(fails) < 5:
                fails.append((r["task_id"], res))
        n = sum(outcomes.values())
        print(f"pass@1 {npass}/{n} = {npass/n:.4f}" if n else "pass@1 n/a")
        print(f"  outcomes {dict(outcomes)}")
        if fails:
            print(f"  first failures {fails}")
        out["pass1"] = round(npass / n, 4) if n else None
        out["pass1_n"] = n
        out["pass1_outcomes"] = dict(outcomes)

    if a.json_out:
        json.dump(out, open(a.json_out, "w"), indent=1)
        print(f"wrote {a.json_out}")


if __name__ == "__main__":
    main()
