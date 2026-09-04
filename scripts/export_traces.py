import json, sys
from pathlib import Path

# The chat template only renders its own reasoning block when the system text does not
# already mention one:  {%- if 'reasoning strength' not in (sys_text | lower) -%}
# so this sentence suppresses the template default and pins the recorded level instead.
STRENGTH_SENTENCE = "Reasoning strength: {}."


def load(path, require_strength=False):
    trajectories = {}
    for line in open(path):
        r = json.loads(line)
        # failed calls have no choices to read
        if "error" in r["response"]:
            continue
        msgs = r["request"]["messages"]
        prev = trajectories.get(r["traj"])
        # keep only the longest call per trajectory: an agent loop appends to a fixed message
        # list, so the last call already holds the whole history and nothing needs stitching
        if prev is None or len(msgs) > len(prev[0]):
            strength = (r["request"].get("chat_template_kwargs") or {}).get("reasoning_strength")
            # Hand-sent connectivity probes carry no chat_template_kwargs and would
            # otherwise export as their own trajectory at the chat template's default
            # reasoning strength.
            if require_strength and not strength:
                continue
            trajectories[r["traj"]] = (msgs, r["request"].get("tools"),
                                       r["response"]["choices"][0]["message"], strength, r["traj"])
    return trajectories


def bake_strength(msgs, strength):
    """Move reasoning strength out of the request and into the system message.

    The level lived in extra_body.chat_template_kwargs, which is part of the *request*
    and not part of `messages`. Anything that re-renders these traces from `messages`
    alone -- speculators' /render step, for one -- never sees that kwarg and would
    silently render all 2000 traces at the template default of "high", mismatching the
    1400 actually generated at low/medium/xhigh.
    """
    if not strength:
        return msgs
    sentence = STRENGTH_SENTENCE.format(strength)
    msgs = [dict(m) for m in msgs]
    if msgs and msgs[0].get("role") == "system":
        text = msgs[0].get("content") or ""
        if "reasoning strength" in text.lower():
            return msgs                      # already pinned, do not double up
        msgs[0]["content"] = f"{text}\n\n{sentence}".strip()
    else:
        msgs.insert(0, {"role": "system", "content": sentence})
    return msgs


def main(src, outdir, require_strength=False):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    trajectories = load(src, require_strength)
    # training wants the full conversation, so the final reply is appended; guidellm generates
    # that reply itself, so it gets the same messages without it
    with open(outdir / "train.jsonl", "w") as train, open(outdir / "guidellm.jsonl", "w") as bench:
        for msgs, tools, final, strength, traj in trajectories.values():
            msgs = bake_strength(msgs, strength)
            assistant = {k: v for k, v in final.items() if v is not None}
            # `messages` is the key speculators' build_speculator_training_dataset reads;
            # it renders and derives the loss mask itself, so nothing here is tokenized.
            train.write(json.dumps({"id": traj, "messages": msgs + [assistant], "tools": tools,
                                    "reasoning_strength": strength}) + "\n")
            bench.write(json.dumps({"id": traj, "messages": msgs, "tools": tools,
                                    "reasoning_strength": strength}) + "\n")

    print(f"{len(trajectories)} trajectories -> {outdir}")


if __name__ == "__main__":
    args = sys.argv[1:]
    require = "--require-strength" in args
    args = [a for a in args if not a.startswith("--")]
    main(args[0], args[1], require)
