import re, collections, sys
from datasets import load_from_disk
d = load_from_disk(sys.argv[1])
c = collections.Counter()
n = len(d)
pat = re.compile(r"reasoning strength:\s*(\w+)", re.I)
for i in range(n):
    s = ""
    for msg in d[i]["messages"]:
        if msg.get("role") == "system":
            s = str(msg.get("content", ""))
            break
    m = pat.search(s)
    c[m.group(1).lower() if m else "NONE"] += 1
print("rows", n)
for k, v in c.most_common():
    print(f"  {k:8s} {v:7d}  {100*v/n:5.1f}%")
