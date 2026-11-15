"""Compare scored outputs ($BENCH_OUT/<system>/scores.json) with the published numbers (results/published.json).

    python benchmark/compare.py [system ...]

Prints every published number with the reproduced one; exit status 1 if any differs by 0.01 or more.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402

published = json.load(open(os.path.join(HERE, "results", "published.json")))
same = differ = absent = 0
for system in sys.argv[1:] or sorted(published):
    path = os.path.join(bench.system_dir(system), "scores.json")
    got = json.load(open(path)) if os.path.exists(path) else {}
    for c, metrics in sorted(published[system].items()):
        for m, v in metrics.items():
            r = got.get(c, {}).get(m)
            if r is None:
                absent += 1
                print(f"  -   {system:22s} {c:16s} {m:9s} published {v:6.2f}  not scored")
            elif abs(r - v) < 0.005:
                same += 1
            else:
                differ += 1
                print(f"DIFF  {system:22s} {c:16s} {m:9s} published {v:6.2f}  reproduced {r:6.2f} ({r - v:+.2f})")
print(f"{same} reproduced exactly, {differ} differ, {absent} not scored")
sys.exit(1 if differ else 0)
