import json, sys
from pathlib import Path


def load(path):
    trajectories = {}
    for line in open(path):
        r = json.loads(line)
        if "error" in r["response"]:
            continue
        msgs = r["request"]["messages"]
        prev = trajectories.get(r["traj"])
        if prev is None or len(msgs) > len(prev[0]):
            trajectories[r["traj"]] = (msgs, r["request"].get("tools"),
                                       r["response"]["choices"][0]["message"])
    return trajectories


def main(src, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    trajectories = load(src)

    with open(outdir / "train.jsonl", "w") as train, open(outdir / "guidellm.jsonl", "w") as bench:
        for msgs, tools, final in trajectories.values():
            assistant = {k: v for k, v in final.items() if v is not None}
            train.write(json.dumps({"conversations": msgs + [assistant], "tools": tools}) + "\n")
            bench.write(json.dumps({"messages": msgs, "tools": tools}) + "\n")

    print(f"{len(trajectories)} trajectories -> {outdir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
