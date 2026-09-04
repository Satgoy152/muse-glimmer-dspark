#!/usr/bin/env python3
"""Smoke-test meta/muse-glimmer-30b on OpenRouter, pinned to parasail/bf16."""
import json, os, sys, time, urllib.request, urllib.error

URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "meta/muse-glimmer-30b"
PROVIDER = {"only": ["deepinfra/bf16"], "allow_fallbacks": False}

key = os.environ.get("OPENROUTER_API_KEY")
if not key:
    sys.exit("set OPENROUTER_API_KEY")

body = {
    "model": MODEL,
    "provider": PROVIDER,
    "messages": [{"role": "user", "content": "Write a Python function that reverses a string."}],
    "temperature": 1.0, "top_p": 0.95, "top_k": 64, "max_tokens": 2048,
    "usage": {"include": True},
}
req = urllib.request.Request(
    URL, data=json.dumps(body).encode(),
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
)
t0 = time.time()
try:
    r = json.load(urllib.request.urlopen(req, timeout=180))
except urllib.error.HTTPError as e:
    sys.exit(f"HTTP {e.code}: {e.read().decode()[:2000]}")
dt = time.time() - t0

print(f"provider : {r.get('provider')}")
print(f"model    : {r.get('model')}")
print(f"latency  : {dt:.1f}s")
print(f"usage    : {json.dumps(r.get('usage'))}")
msg = r["choices"][0]["message"]
print(f"reasoning: {(msg.get('reasoning') or '')[:300]}")
content = msg.get("content") or ""
print(f"content  : {content[:800]}")
