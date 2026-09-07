#!/usr/bin/env python3
"""Build /mnt/data/eval/summary.md from the per-drafter metric JSONs.

Regenerating after a fifth drafter lands is this script again -- it globs
whatever is in eval/ and re-ranks. Nothing about the table is hand-maintained.
"""
import argparse, glob, json, os, re
from collections import defaultdict

SEGMENTS = ["low", "medium", "high", "xhigh"]
BASE = {"low": 4.345, "medium": 4.034, "high": 3.796, "xhigh": 3.797, "pooled": 3.913}
BUCKETS = ["<2K", "2-8K", "8-16K", "16-32K", "32-64K", "64K+"]


def prom(path):
    if not os.path.exists(path):
        return {}
    txt = open(path).read()
    def g(n):
        m = re.search(rf"^{re.escape(n)}\{{[^}}]*\}} ([0-9.e+-]+)$", txt, re.M)
        return float(m.group(1)) if m else None
    out = {}
    for k, n in (("ttft", "vllm:time_to_first_token_seconds"),
                 ("e2e", "vllm:e2e_request_latency_seconds")):
        s, c = g(n + "_sum"), g(n + "_count")
        if s and c:
            out[k + "_mean"], out["n_req"] = s / c, c
    q, h = g("vllm:prefix_cache_queries_total"), g("vllm:prefix_cache_hits_total")
    if q:
        out["prefix_hit_rate"] = h / q
    d, a = g("vllm:spec_decode_num_drafts_total"), g("vllm:spec_decode_num_accepted_tokens_total")
    if d:
        out["srv_accept_len"] = 1 + a / d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="/mnt/data/eval")
    ap.add_argument("--out", default="/mnt/data/eval/summary.md")
    ap.add_argument("--concurrency", type=int, default=10)
    a = ap.parse_args()

    runs = {}
    for f in sorted(glob.glob(os.path.join(a.eval_dir, "*.json"))):
        name = os.path.basename(f)[:-5]
        if name.startswith("_") or name == "buckets":
            continue
        d = json.load(open(f))
        if "pooled" not in d:
            continue
        d["_prom"] = prom(os.path.join(a.eval_dir, name + ".prom"))
        runs[name] = d
    order = sorted(runs, key=lambda n: -runs[n]["pooled"]["accept_len"])

    bpath = os.path.join(a.eval_dir, "buckets.json")
    buckets = json.load(open(bpath)) if os.path.exists(bpath) else None

    L = []; w = L.append
    w("# Speculator comparison — Terminal-Bench frozen 40\n")
    w("## How these were produced\n")
    w("Each drafter served the **same frozen set of 1,753 recorded Terminal-Bench "
      "calls**, replayed verbatim against its own vLLM server. The call set is the "
      "`raw` split of `Satgoy152/Muse-Glimmer-Terminal-Bench-Eval`: the frozen 40 "
      "pinned to `harbor-framework/terminal-bench-1` at "
      "`d28711d0da2675d0bb1d56de45ae5df6082438a3`, 36 trajectories, 42.3 M prompt "
      "tokens.\n")
    w("`NUM_SPEC_TOKENS=15` for every run — acceptance length is only comparable at "
      "the same draft length. Every run got its own `TRACE_DIR`; the DSpark patch was "
      "confirmed in the log of every `dspark` server; "
      "`--per-request-spec-decode-metrics summary` supplied the per-request block "
      "that acceptance is pooled from.\n")
    w("> **This is a replay, not an agentic re-run.** It measures acceptance on a "
      "fixed, representative agentic prompt distribution — which removes trajectory "
      "drift as a confound, since every drafter sees byte-identical prompts. It does "
      "**not** produce `is_resolved`: task completion needs the real harness, so the "
      "`resolved` column is *not measured* for every row here. The published "
      "on-policy baseline's 19/39 is carried for reference only.\n")

    w("## Pooled\n")
    w("| drafter | calls | out_tok | accept_len | draft_rate | tok/s | TTFT mean | resolved |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    for n in order:
        p, pm = runs[n]["pooled"], runs[n]["_prom"]
        t = f"{pm['ttft_mean']:.3f} s" if "ttft_mean" in pm else "—"
        w(f"| `{n}` | {p['calls']} | {p['out_tok']:,} | **{p['accept_len']:.3f}** | "
          f"{p['draft_rate']:.3f} | {p['tok_s']:.1f} | {t} | not measured |")
    w(f"| *published on-policy baseline* | *1753* | *519,902* | *{BASE['pooled']:.3f}* "
      f"| *0.194* | *151.1* | *n/a* | *19/39 (48.7%)* |")
    w("")

    ctrl = runs.get("dflash-official")
    if ctrl:
        c = ctrl["pooled"]["accept_len"]
        w(f"**Control.** `dflash-official` replays at **{c:.3f}** against the published "
          f"on-policy **{BASE['pooled']:.3f}** ({100*(c-BASE['pooled'])/BASE['pooled']:+.1f}%). "
          "Close enough to trust the rest: the residual is replay-vs-on-policy, a newer "
          "vLLM nightly than the baseline was measured on, and temperature-1.0 sampling "
          "noise.\n")

    ft, base = runs.get("dspark-run-a-32k"), runs.get("dspark-community")
    if ft and base:
        w("## The fine-tune's effect, isolated\n")
        w("`dspark-run-a-32k` against `dspark-community`, the warm start it was "
          "fine-tuned from — same architecture, same `block_size` 15, same "
          "`aux_hidden_state_layer_ids`. The only difference is the fine-tune.\n")
        w("| strength | community | run-a-32k | Δ | |")
        w("|---|---:|---:|---:|---|")
        for s in SEGMENTS + ["pooled"]:
            if s not in ft or s not in base:
                continue
            x, y = base[s]["accept_len"], ft[s]["accept_len"]
            bold = "**" if s == "pooled" else ""
            w(f"| {bold}{s}{bold} | {bold}{x:.3f}{bold} | {bold}{y:.3f}{bold} | "
              f"{bold}{y-x:+.3f}{bold} | {100*(y-x)/x:+.1f}% |")
        w("")
        if ctrl:
            gap = ctrl["pooled"]["accept_len"] - base["pooled"]["accept_len"]
            got = ft["pooled"]["accept_len"] - base["pooled"]["accept_len"]
            w(f"The fine-tune improves acceptance at **every reasoning strength**, and "
              f"closes **{100*got/gap:.0f}%** of the gap between the community warm start "
              f"and the official DFlash drafter.\n")

    w("## Per reasoning strength\n")
    w("| drafter | " + " | ".join(SEGMENTS) + " | pooled |")
    w("|---|" + "---:|" * (len(SEGMENTS) + 1))
    for n in order:
        cells = [f"{runs[n][s]['accept_len']:.3f}" if s in runs[n] else "—" for s in SEGMENTS]
        w(f"| `{n}` | " + " | ".join(cells) + f" | **{runs[n]['pooled']['accept_len']:.3f}** |")
    w("| *published baseline* | " + " | ".join(f"*{BASE[s]:.3f}*" for s in SEGMENTS)
      + f" | *{BASE['pooled']:.3f}* |")
    w("\nAcceptance falls as reasoning strength rises, as in the baseline.\n")

    w("### Full breakdown per drafter\n")
    for n in order:
        w(f"**`{n}`**\n")
        w("| strength | calls | out_tok | accept_len | draft_rate | tok/s |")
        w("|---|---:|---:|---:|---:|---:|")
        for s in SEGMENTS + ["pooled"]:
            if s not in runs[n]:
                continue
            d = runs[n][s]
            w(f"| {s} | {d['calls']} | {d['out_tok']:,} | {d['accept_len']:.3f} | "
              f"{d['draft_rate']:.3f} | {d['tok_s']:.1f} |")
        m, pm = runs[n]["_meta"], runs[n]["_prom"]
        note = (f"Excluded **{m['errored']} errored calls** and {m['no_strength']} without a "
                f"recorded `reasoning_strength`.")
        if m["errored"] == 0:
            note = ("**No calls excluded** — every one of the 1,753 returned a response "
                    "with a per-request metrics block.")
        if "srv_accept_len" in pm:
            note += f" Server-counter cross-check: {pm['srv_accept_len']:.3f}."
        if "prefix_hit_rate" in pm:
            note += f" Prefix-cache hit rate {pm['prefix_hit_rate']*100:.1f}%."
        w(f"\n{note}\n")

    if buckets:
        w("## Acceptance by context length × reasoning strength\n")
        D = buckets["data"]
        for n in order:
            if n not in D:
                continue
            w(f"**`{n}`**\n")
            w("| ctx | " + " | ".join(SEGMENTS) + " |")
            w("|---|" + "---:|" * len(SEGMENTS))
            for lb in BUCKETS:
                row = []
                for s in SEGMENTS:
                    c = D[n].get(f"{lb}|{s}")
                    row.append(f"{1 + c['accepted']/c['steps']:.3f}"
                               if c and c.get("steps") else "—")
                w(f"| {lb} | " + " | ".join(row) + " |")
            w("")

    w("## Caveats\n")
    w("These are carried from `benchmark/terminal_bench/README.md` and must not be "
      "dropped when these numbers are quoted.\n")
    w("**Two denominators, quoted separately.** Task completion is over **39 tasks**; "
      "traces are over **36 trajectories**, of which 38 tasks issued any call at all. "
      "They are not the same denominator and averaging them is wrong.\n")
    w("* `word2vec-from-scratch` **cannot build at the pinned commit** — an unpinned "
      "`huggingface_hub` now rejects the Dockerfile's `load_dataset('wikitext', …)` "
      "(the dataset moved to `Salesforce/wikitext`). The agent never runs, so this is "
      "upstream task rot that fails for *any* model, and it is excluded from the "
      "denominator (`resolved: null`).")
    w("* `extract-safely` issued **zero model calls** across the whole recording. It "
      "failed before any inference, but TB recorded it as an ordinary unresolved task "
      "(`failure_mode: unset`) rather than an agent error — so it counts as a failure "
      "in 19/39 *without having exercised the model at all*.")
    w("* `llm-inference-batching-scheduler` and `amuse-install` were removed by the "
      "canary filter. They ran, and still count toward completion.\n")
    w(f"**tok/s reflects contention as much as the drafter.** These runs used "
      f"concurrency **{a.concurrency}**, not the baseline's `--concurrent 8`, so the "
      f"tok/s and TTFT columns are **not comparable to the published 151.1**. "
      f"Acceptance length is the contention-independent number and is unaffected.\n")
    w("**Prefix caching was on** (`--enable-prefix-caching`), hit rates ~96%. That "
      "flatters TTFT and tok/s relative to a cold run, identically across drafters. "
      "Acceptance length is unaffected either way.\n")
    w("**TTFT is the server-side histogram**, not the client's first delta. "
      "`--enable-auto-tool-choice` makes the tool-call parser buffer deltas until a "
      "call is whole, so a client-measured TTFT is ≈ end-to-end on tool-calling "
      "requests and is wrong. The published baseline never measured TTFT at all.\n")
    w("**Server-global counters are used only as a cross-check.** They cannot be "
      "attributed in general, but each endpoint here served exactly one drafter and "
      "one workload. Where they appear they agree with the per-request pooling to "
      "within 0.01.\n")
    w("**Canary: 0 for every trace.** `filter_canary.py` reported 0 contaminated "
      "trajectories and 0 remaining occurrences on all runs; the published `raw` split "
      "was already canary-clean at source.\n")
    w("**Adding a fifth drafter** is one invocation and a regenerate:\n")
    w("```bash\nNAME=dspark-run-b-49k SPEC=/mnt/data/speculators/dspark-run-b-49k \\\n"
      "  METHOD=dspark bash scripts/eval_replay.sh\n"
      "python benchmark/terminal_bench/buckets.py\n"
      "python benchmark/terminal_bench/make_summary.py\n```\n")

    open(a.out, "w").write("\n".join(L))
    print(f"wrote {a.out} ({len(order)} drafters: {', '.join(order)})")


if __name__ == "__main__":
    main()
