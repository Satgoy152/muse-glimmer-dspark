#!/usr/bin/env python3
"""Fetch Terminal-Bench task.yaml metadata -> data/raw/tb_tasks.json"""
import json, re, sys
from concurrent.futures import ThreadPoolExecutor
import requests, yaml

REPO = "harbor-framework/terminal-bench-1"
REF = "main"
tree = requests.get(f"https://api.github.com/repos/{REPO}/git/trees/{REF}?recursive=1", timeout=90).json()
sha = tree["sha"]
paths = [t["path"] for t in tree["tree"] if t["path"].startswith("original-tasks/") and t["path"].endswith("task.yaml")]
print(f"{len(paths)} task.yaml @ tree {sha}", file=sys.stderr)

def get(p):
    r = requests.get(f"https://raw.githubusercontent.com/{REPO}/{sha}/{p}", timeout=60)
    r.raise_for_status()
    txt = r.text
    d = yaml.safe_load(txt) or {}
    instr = d.get("instruction", "") or ""
    return {
        "task_id": p.split("/")[1],
        "path": p,
        "difficulty": d.get("difficulty"),
        "category": d.get("category"),
        "tags": d.get("tags") or [],
        "max_agent_timeout_sec": d.get("max_agent_timeout_sec"),
        "canary": bool(re.search(r"terminal-bench-canary GUID (\S+)", txt)),
        "instruction_chars": len(instr),
        # github repos referenced in the instruction, for the leakage audit
        "repos_mentioned": sorted(set(
            m.lower() for m in re.findall(r"github\.com/([\w.-]+/[\w.-]+)", instr)
        )),
    }

with ThreadPoolExecutor(16) as ex:
    tasks = sorted(ex.map(get, paths), key=lambda t: t["task_id"])

json.dump({"repo": REPO, "tree_sha": sha, "n": len(tasks), "tasks": tasks},
          open("data/raw/tb_tasks.json", "w"), indent=1)
print(f"wrote data/raw/tb_tasks.json ({len(tasks)} tasks)", file=sys.stderr)
