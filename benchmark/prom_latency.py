"""Server-side latency profile from a vLLM /metrics scrape.

The client-side ttft/tpot in replay_metrics.py are not usable on this workload:
--enable-auto-tool-choice makes the tool-call parser buffer deltas until a call
is whole, and ~all of these requests end in tool_calls, so the "first" delta
arrives at roughly the end. Measured on tb-dflash2: 86.6% of calls have client
TTFT within 5% of their own e2e latency, and 70.6% of tpot values collapse to 0.

The server histograms have no such problem -- they are recorded inside the
engine, before the parser buffers anything. They are server-global, which is
normally unattributable, but each endpoint here served exactly one drafter and
one workload (same argument the existing summary makes for the spec_decode
counters).

Quantiles are linear-interpolated within the containing bucket. vLLM histogram
buckets are coarse in the tail, so p99 is bucket-limited; p50/p90 are the
numbers to compare.
"""
import re
import sys
import json
import glob
import os

HISTS = {
    "ttft": "vllm:time_to_first_token_seconds",
    "itl": "vllm:inter_token_latency_seconds",
    "tpot": "vllm:request_time_per_output_token_seconds",
    "queue": "vllm:request_queue_time_seconds",
    "prefill": "vllm:request_prefill_time_seconds",
    "decode": "vllm:request_decode_time_seconds",
    "infer": "vllm:request_inference_time_seconds",
}


def buckets(path, metric):
    """Return (sorted [(le, cumulative_count)], total, sum) for one histogram."""
    pat = re.compile(r"^" + re.escape(metric) + r"_bucket\{([^}]*)\}\s+(\S+)")
    out, total, ssum = [], None, None
    for line in open(path):
        if line.startswith("#"):
            continue
        m = pat.match(line)
        if m:
            le = re.search(r'le="([^"]+)"', m.group(1))
            if le:
                out.append((float(le.group(1)), float(m.group(2))))
            continue
        if line.startswith(metric + "_count"):
            total = float(line.rsplit(" ", 1)[1])
        elif line.startswith(metric + "_sum"):
            ssum = float(line.rsplit(" ", 1)[1])
    out.sort()
    return out, total, ssum


def quantile(bk, total, q):
    """Linear interpolation inside the bucket that contains the q-th sample."""
    if not bk or not total:
        return None
    target = q * total
    prev_le, prev_c = 0.0, 0.0
    for le, c in bk:
        if c >= target:
            if le == float("inf"):
                return prev_le
            span = c - prev_c
            if span <= 0:
                return le
            frac = (target - prev_c) / span
            return prev_le + frac * (le - prev_le)
        prev_le, prev_c = le, c
    return prev_le


def profile(path):
    row = {}
    for key, metric in HISTS.items():
        bk, total, ssum = buckets(path, metric)
        if not total:
            continue
        row[key] = {
            "p50": quantile(bk, total, 0.50),
            "p90": quantile(bk, total, 0.90),
            "p99": quantile(bk, total, 0.99),
            "mean": (ssum / total) if ssum is not None else None,
            "n": int(total),
        }
    return row


def fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(glob.glob("/mnt/data/eval/*.prom"))
    rows = {}
    for p in paths:
        name = os.path.basename(p)[:-5]
        r = profile(p)
        if r:
            rows[name] = r

    hdr = (f"{'drafter':34s} {'n':>5s} {'TTFT p50':>9s} {'TTFT p90':>9s} "
           f"{'ITL p50':>8s} {'ITL p90':>8s} {'TPOT p50':>9s} {'queue p50':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for n, r in sorted(rows.items()):
        t = r.get("ttft", {})
        i = r.get("itl", {})
        tp = r.get("tpot", {})
        q = r.get("queue", {})
        print(f"{n:34s} {t.get('n', 0):5d} {fmt(t.get('p50')):>9s} "
              f"{fmt(t.get('p90')):>9s} {fmt(i.get('p50'), 4):>8s} "
              f"{fmt(i.get('p90'), 4):>8s} {fmt(tp.get('p50'), 4):>9s} "
              f"{fmt(q.get('p50'), 4):>10s}")

    if "--json" in os.environ.get("ARGS", ""):
        print(json.dumps(rows, indent=1))
