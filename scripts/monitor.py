#!/usr/bin/env python3
"""Scrape run + serving metrics so progress can be read from one file.

Writes status.json (latest snapshot), status.txt (rendered), and history.csv
(one row per scrape) under --state-dir. Only reads state the run already
writes, so it is safe to start, kill, and restart at any time.

The spec-decode block records acceptance length on this workload, which the
DSpark card lists as unbenchmarked -- it is the baseline a fine-tuned
speculator has to beat, and is cheap to capture now but not reconstructable
later.
"""
import argparse, json, os, re, shutil, subprocess, time, urllib.error, urllib.request
from pathlib import Path

SEGMENTS = [(0, 600, "low"), (600, 1200, "medium"), (1200, 1800, "high"), (1800, 2000, "xhigh")]
METRIC_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9.eE+-]+)$")


def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def nlines(s):
    return len([x for x in s.splitlines() if x.strip()])


def scrape_vllm(url, token):
    """Sum each metric family across labels. Returns {} if the endpoint is down."""
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        body = urllib.request.urlopen(req, timeout=15).read().decode()
    except Exception:
        return {}
    out = {}
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        m = METRIC_RE.match(line.strip())
        if not m:
            continue
        name, val = m.group(1), float(m.group(3))
        out[name] = out.get(name, 0.0) + val
    return out


def rate(cur, prev, key, dt):
    if not prev or key not in cur or key not in prev or dt <= 0:
        return None
    d = cur[key] - prev[key]
    return d / dt if d >= 0 else None


def ratio(cur, prev, num, den):
    """Delta-based ratio, so it reflects the last interval rather than all time."""
    if not prev or num not in cur or den not in cur:
        return None
    dn, dd = cur[num] - prev.get(num, 0), cur[den] - prev.get(den, 0)
    return dn / dd if dd > 0 else None


def disk_io(dev="vdc"):
    txt = sh("iostat", "-xm", "1", "2")
    rows = [l for l in txt.splitlines() if l.strip().startswith(dev)]
    if not rows:
        return {}
    f = rows[-1].split()
    try:
        return {"read_mbs": float(f[2]), "write_mbs": float(f[8]), "util_pct": float(f[-1]),
                "aqu": float(f[-2])}
    except Exception:
        return {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/data/runs/train")
    p.add_argument("--traces", default="/data/traces/train")
    p.add_argument("--subset", default="/data/muse-glimmer-dspark/data/training/swegym_2k")
    p.add_argument("--state-dir", default="/data/monitor")
    p.add_argument("--vllm", default=os.environ.get("VLLM_METRICS_URL",
                                                    "http://127.0.0.1:8000/metrics"))
    p.add_argument("--token", default=os.environ.get("UPSTREAM_API_KEY", ""))
    p.add_argument("--interval", type=int, default=60)
    p.add_argument("--once", action="store_true")
    a = p.parse_args()

    rows = [json.loads(l) for l in open(Path(a.subset) / "train.jsonl")]
    out, traces, state = Path(a.out), Path(a.traces), Path(a.state_dir)
    state.mkdir(parents=True, exist_ok=True)
    hist = state / "history.csv"

    prev_m, prev_t, prev_done, t_start, done_start = None, None, None, time.time(), None
    # calls.jsonl is append-only, so track a byte offset instead of re-reading it
    off, ncalls, nerr, tok_out, tok_in, lat_sum = 0, 0, 0, 0, 0, 0.0

    while True:
        now = time.time()
        snap = {"ts": now, "time": time.strftime("%Y-%m-%d %H:%M:%S")}

        # ---- run progress -------------------------------------------------
        done_idx = [j for j, r in enumerate(rows)
                    if (out / r["instance_id"] / f'{r["instance_id"]}.traj.json').exists()]
        done = len(done_idx)
        if done_start is None:
            done_start, t_start = done, now
        mlog = out / "minisweagent.log"
        failed = sum(1 for l in open(mlog, errors="ignore")
                     if "ERROR - Error processing" in l) if mlog.exists() else 0
        elapsed = now - t_start
        r_hr = (done - done_start) / elapsed * 3600 if elapsed > 120 else None
        seg = next((s for lo, hi, s in SEGMENTS if lo <= done < hi), "complete")
        snap["run"] = {"done": done, "total": len(rows), "failed": failed, "segment": seg,
                       "pct": round(100 * done / len(rows), 2),
                       "rate_per_hr": round(r_hr, 1) if r_hr else None,
                       "eta_hours": round((len(rows) - done) / r_hr, 2) if r_hr else None,
                       "elapsed_hours": round(elapsed / 3600, 2)}

        # ---- traces (incremental read) ------------------------------------
        cf = traces / "calls.jsonl"
        if cf.exists():
            with open(cf) as f:
                f.seek(off)
                for line in f:
                    if not line.endswith("\n"):
                        break                      # partial trailing write
                    off += len(line.encode())
                    ncalls += 1
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    lat_sum += rec.get("latency", 0)
                    rsp = rec.get("response", {})
                    if "error" in rsp:
                        nerr += 1
                    u = rsp.get("usage") or {}
                    tok_out += u.get("completion_tokens", 0) or 0
                    tok_in += u.get("prompt_tokens", 0) or 0
        snap["traces"] = {
            "calls": ncalls, "errors": nerr,
            "bytes": cf.stat().st_size if cf.exists() else 0,
            "completion_tokens": tok_out, "prompt_tokens": tok_in,
            "avg_completion_tokens": round(tok_out / ncalls, 1) if ncalls else None,
            "avg_prompt_tokens": round(tok_in / ncalls, 1) if ncalls else None,
            "avg_latency_s": round(lat_sum / ncalls, 2) if ncalls else None,
            "calls_per_instance": round(ncalls / done, 1) if done else None}

        # ---- system --------------------------------------------------------
        snap["system"] = {
            "containers": nlines(sh("docker", "ps", "-q")),
            "images": nlines(sh("docker", "images", "-q")),
            "blocked_on_pull": nlines(sh("pgrep", "-f", "docker run -d --name minisweagent")),
            "pulling": nlines(sh("pgrep", "-f", "docker pull")),
            "free_gb": round(shutil.disk_usage("/data").free / 1024**3),
            "load1": os.getloadavg()[0], "disk": disk_io()}

        # ---- vLLM ----------------------------------------------------------
        m = scrape_vllm(a.vllm, a.token)
        dt = now - prev_t if prev_t else 0
        if m:
            snap["vllm"] = {
                "up": True,
                "running": m.get("vllm:num_requests_running"),
                "waiting": m.get("vllm:num_requests_waiting"),
                "kv_cache_pct": round(100 * m.get("vllm:kv_cache_usage_perc", 0), 1),
                "gen_tok_s": round(rate(m, prev_m, "vllm:generation_tokens_total", dt) or 0, 1),
                "prompt_tok_s": round(rate(m, prev_m, "vllm:prompt_tokens_total", dt) or 0, 1),
                "preemptions": m.get("vllm:num_preemptions_total"),
                "prefix_cache_hit_rate": rr(ratio(m, prev_m, "vllm:prefix_cache_hits_total",
                                                  "vllm:prefix_cache_queries_total")),
                "e2e_latency_s": rr(ratio(m, prev_m, "vllm:e2e_request_latency_seconds_sum",
                                          "vllm:e2e_request_latency_seconds_count")),
                "ttft_s": rr(ratio(m, prev_m, "vllm:time_to_first_token_seconds_sum",
                                   "vllm:time_to_first_token_seconds_count")),
            }
            # Speculative decoding: acceptance length = 1 bonus token + accepted per draft.
            acc_per_draft = ratio(m, prev_m, "vllm:spec_decode_num_accepted_tokens_total",
                                  "vllm:spec_decode_num_drafts_total")
            snap["spec_decode"] = {
                "acceptance_length": rr(acc_per_draft + 1 if acc_per_draft is not None else None),
                "draft_acceptance_rate": rr(ratio(m, prev_m,
                                                  "vllm:spec_decode_num_accepted_tokens_total",
                                                  "vllm:spec_decode_num_draft_tokens_total")),
                "cum_drafts": m.get("vllm:spec_decode_num_drafts_total"),
                "cum_accepted": m.get("vllm:spec_decode_num_accepted_tokens_total"),
                "cum_drafted": m.get("vllm:spec_decode_num_draft_tokens_total")}
            prev_m, prev_t = m, now
        else:
            snap["vllm"] = {"up": False}

        # ---- persist --------------------------------------------------------
        (state / "status.json").write_text(json.dumps(snap, indent=2))
        (state / "status.txt").write_text(render(snap))
        write_csv(hist, snap)
        print(render(snap), flush=True)

        if a.once:
            return
        time.sleep(a.interval)


def rr(v, nd=3):
    return round(v, nd) if isinstance(v, (int, float)) else None


def render(s):
    r, t, y, v, sd = (s.get(k, {}) for k in ("run", "traces", "system", "vllm", "spec_decode"))
    n = int(40 * r.get("pct", 0) / 100)
    eta = f"{r['eta_hours']:.1f}h" if r.get("eta_hours") else "--"
    L = [f"=== {s['time']}  segment={r.get('segment')} ===",
         f"  [{'#'*n}{'.'*(40-n)}] {r.get('done')}/{r.get('total')} ({r.get('pct')}%)  failed={r.get('failed')}",
         f"  rate={r.get('rate_per_hr')}/hr  eta={eta}  elapsed={r.get('elapsed_hours')}h",
         f"  RUN     containers={y.get('containers')} blocked_pull={y.get('blocked_on_pull')} "
         f"pulling={y.get('pulling')} images={y.get('images')}",
         f"  DISK    free={y.get('free_gb')}G  write={y.get('disk',{}).get('write_mbs')}MB/s  "
         f"util={y.get('disk',{}).get('util_pct')}%  load={y.get('load1'):.0f}",
         f"  TRACES  calls={t.get('calls')} errors={t.get('errors')} "
         f"{t.get('bytes',0)/1e6:.0f}MB  avg_out_tok={t.get('avg_completion_tokens')} "
         f"avg_in_tok={t.get('avg_prompt_tokens')} calls/inst={t.get('calls_per_instance')}"]
    if v.get("up"):
        L += [f"  VLLM    running={v.get('running')} waiting={v.get('waiting')} "
              f"kv={v.get('kv_cache_pct')}%  gen={v.get('gen_tok_s')}tok/s "
              f"prompt={v.get('prompt_tok_s')}tok/s",
              f"          ttft={v.get('ttft_s')}s e2e={v.get('e2e_latency_s')}s "
              f"prefix_hit={v.get('prefix_cache_hit_rate')} preempt={v.get('preemptions')}",
              f"  SPEC    accept_len={sd.get('acceptance_length')} "
              f"draft_accept_rate={sd.get('draft_acceptance_rate')}"]
    else:
        L += ["  VLLM    DOWN / unreachable"]
    return "\n".join(L)


def write_csv(path, s):
    flat = {}
    for sec, d in s.items():
        if isinstance(d, dict):
            for k, val in d.items():
                if not isinstance(val, dict):
                    flat[f"{sec}.{k}"] = val
        else:
            flat[sec] = d
    new = not path.exists()
    with open(path, "a") as f:
        if new:
            f.write(",".join(flat) + "\n")
        f.write(",".join("" if v is None else str(v) for v in flat.values()) + "\n")


if __name__ == "__main__":
    main()
