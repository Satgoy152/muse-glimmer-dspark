#!/usr/bin/env python3
"""Convert a proxy.py trace into replay.py's input schema.

Only rows that are actually agent calls survive: a response without an error, a
`reasoning_strength` in the chat template kwargs, and a `tools` block. The
health probe that `swegym_holdout.sh` sent through the proxy had none of those
and was the only row in the file it produced.
"""
import argparse, json
from collections import Counter

SEGMENTS = ["low", "medium", "high", "xhigh"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    kept, drop = [], Counter()
    for line in open(a.src):
        if not line.strip():
            continue
        d = json.loads(line)
        req, resp = d["request"], (d.get("response") or {})
        st = (req.get("chat_template_kwargs") or {}).get("reasoning_strength")
        if "error" in resp:
            drop["errored"] += 1; continue
        if st not in SEGMENTS:
            drop["no_reasoning_strength"] += 1; continue
        if not req.get("tools"):
            drop["no_tools"] += 1; continue
        u = resp.get("usage") or {}
        kept.append({"id": d["traj"], "ts": d["ts"], "is_error": False,
                     "reasoning_strength": st, "request": req,
                     "prompt_tokens": u.get("prompt_tokens"),
                     "completion_tokens": u.get("completion_tokens")})

    with open(a.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    if a.quiet:
        print(len(kept))
        return
    print(f"kept {len(kept)} replayable calls -> {a.out}")
    print("dropped:", dict(drop))
    print("strengths:", dict(Counter(r["reasoning_strength"] for r in kept)))
    print("trajectories:", len(set(r["id"] for r in kept)))
    ct = [r["completion_tokens"] or 0 for r in kept]
    if ct:
        print(f"completion tokens: total {sum(ct):,}  mean {sum(ct)/len(ct):.1f}  max {max(ct)}")


if __name__ == "__main__":
    main()
