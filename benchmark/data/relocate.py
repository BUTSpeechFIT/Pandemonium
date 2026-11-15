"""Point a copied or downloaded $BENCH_DATA at its new location.

    python benchmark/data/relocate.py            # rewrites every <corpus>/manifest.json and cutset.jsonl.gz

The prepare scripts write absolute audio paths; this rebases each one onto $BENCH_DATA/<corpus>/, keeping the part
after the corpus directory. Standard library only.
"""
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench  # noqa: E402


def rebase(path, corpus):
    parts = path.split("/")
    i = len(parts) - 1 - parts[::-1].index(corpus)  # the last '<corpus>' directory in the old path
    return os.path.join(bench.corpus_dir(corpus), *parts[i + 1:])


def rebase_sources(obj, corpus):
    if isinstance(obj, dict):
        if obj.get("type") == "file" and "source" in obj:
            obj["source"] = rebase(obj["source"], corpus)
        for v in obj.values():
            rebase_sources(v, corpus)
    elif isinstance(obj, list):
        for v in obj:
            rebase_sources(v, corpus)


for corpus in sorted(os.listdir(bench.DATA)):
    if not os.path.exists(bench.manifest_path(corpus)):
        continue
    m = {s: rebase(p, corpus) for s, p in bench.load_manifest(corpus).items()}
    missing = [p for p in m.values() if not os.path.exists(p)]
    json.dump(m, open(bench.manifest_path(corpus), "w"))
    if os.path.exists(bench.cutset_path(corpus)):
        cuts = [json.loads(line) for line in gzip.open(bench.cutset_path(corpus), "rt")]
        for c in cuts:
            rebase_sources(c, corpus)
        with gzip.open(bench.cutset_path(corpus), "wt") as f:
            f.writelines(json.dumps(c) + "\n" for c in cuts)
    print(f"{corpus}: {len(m)} recordings{f', {len(missing)} MISSING' if missing else ''}", flush=True)
