import json, os, random, subprocess, sys, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import requests

BASE = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8080")
MODEL = os.environ.get("MODEL", "meta/muse-glimmer-30b")
IMAGE = os.environ.get("IMAGE", "python:3.11-slim")
RUN = Path(os.environ.get("TRACE_DIR", "data/traces/dev"))
DOCKER = os.environ.get("NO_DOCKER") != "1"

MAX_TURNS = int(os.environ.get("MAX_TURNS", 40))
MAX_TOOL_CHARS = 4000
EFFORTS = (["low"] * 3) + (["medium"] * 3) + (["high"] * 3) + ["xhigh"]

TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command in the repository checkout.",
        "parameters": {
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
        },
    },
}]

SYSTEM = (
    "You are a software engineer. The bash tool runs inside a checked-out repository at {root}, "
    "which is already your working directory. Investigate the issue, edit files, and run the "
    "tests. Reply without a tool call only when you are done."
)


def problem_statements():
    out = {}
    for p in ("data/raw/swegym.parquet", "data/raw/swebench_extra.parquet"):
        t = pq.read_table(p, columns=["instance_id", "problem_statement"])
        out.update(zip(t.column("instance_id").to_pylist(),
                       t.column("problem_statement").to_pylist()))
    return out


class Env:
    def __init__(self, repo, commit):
        self.name = f"swe-{uuid.uuid4().hex[:12]}"
        self.dir = RUN / "work" / self.name
        self.root = "/repo"
        url = f"https://github.com/{repo}"
        if DOCKER:
            subprocess.run(["docker", "run", "-d", "--rm", "--name", self.name,
                            "--network", "bridge", "--memory", "4g", "--cpus", "2",
                            IMAGE, "sleep", "infinity"], check=True, capture_output=True)
            self.sh(f"git clone --quiet {url} /repo && git -C /repo checkout --quiet {commit}", 600)
        else:
            self.root = str(self.dir / "repo")
            self.dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--quiet", url, str(self.dir / "repo")],
                           check=True, capture_output=True)
            subprocess.run(["git", "-C", str(self.dir / "repo"), "checkout", "--quiet", commit],
                           check=True, capture_output=True)

    def sh(self, cmd, timeout=120):
        argv = ["bash", "-lc", f"cd {self.root} 2>/dev/null; {cmd}"]
        if DOCKER:
            argv = ["docker", "exec", self.name] + argv
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
            out = (r.stdout + r.stderr).strip()
        except subprocess.TimeoutExpired:
            out = f"[timed out after {timeout}s]"
        return out[:MAX_TOOL_CHARS] or "[no output]"

    def close(self):
        if DOCKER:
            subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
        else:
            subprocess.run(["rm", "-rf", str(self.dir)], capture_output=True)


def rollout(inst, statement, effort):
    env = Env(inst["repo"], inst["base_commit"])
    messages = [
        {"role": "system", "content": SYSTEM.format(root=env.root)},
        {"role": "user", "content": f"Repository: {inst['repo']}\n\n{statement}"},
    ]
    try:
        for _ in range(MAX_TURNS):
            r = requests.post(f"{BASE}/v1/chat/completions", timeout=900, json={
                "model": MODEL, "messages": messages, "tools": TOOLS,
                "temperature": 1.0, "top_p": 0.95, "top_k": 64,
                "max_tokens": 8192, "reasoning": {"effort": effort},
            })
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            messages.append({k: v for k, v in msg.items() if v is not None})
            calls = msg.get("tool_calls")
            if not calls:
                break
            for c in calls:
                cmd = json.loads(c["function"]["arguments"]).get("cmd", "")
                messages.append({"role": "tool", "tool_call_id": c["id"],
                                 "content": env.sh(cmd)})
    finally:
        env.close()
    return len(messages)


def main(limit):
    RUN.mkdir(parents=True, exist_ok=True)
    done = RUN / "done"
    done.mkdir(exist_ok=True)

    statements = problem_statements()
    instances = [json.loads(l) for l in open("data/training/train_instances.jsonl")]
    instances = [i for i in instances if not (done / i["instance_id"]).exists()][:limit]
    rng = random.Random(0)

    def one(inst):
        effort = rng.choice(EFFORTS)
        try:
            n = rollout(inst, statements[inst["instance_id"]], effort)
            (done / inst["instance_id"]).write_text(effort)
            print(f"ok   {inst['instance_id']} {effort} {n} msgs", flush=True)
        except Exception as e:
            print(f"fail {inst['instance_id']} {type(e).__name__}: {e}", flush=True)

    with ThreadPoolExecutor(int(os.environ.get("CONCURRENCY", 20))) as ex:
        list(ex.map(one, instances))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
