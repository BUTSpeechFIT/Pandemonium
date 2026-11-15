"""NeMo multitalker output ($BENCH_OUT/mt_parakeet_raw/<corpus>.json, SegLST keyed by wav name) ->
$BENCH_OUT/mt_parakeet/<corpus>/<session>.json. Sessions it emitted nothing for get an empty segment list."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bench  # noqa: E402

for c in bench.expand(sys.argv[1:]):
    raw = f"{bench.OUT}/mt_parakeet_raw/{c}.json"
    if not os.path.exists(raw):
        print(f"mt_parakeet {c}: not decoded", flush=True)
        continue
    man = bench.load_manifest(c)
    stem = {os.path.basename(p).split(".")[0]: s for s, p in man.items()}
    segs = {s: [] for s in man}
    for x in json.load(open(raw)):
        segs[stem[x["session_id"]]].append({"start": float(x["start_time"]), "end": float(x["end_time"]),
                                            "speaker": str(x["speaker"]), "text": x["words"]})
    for s, sg in segs.items():
        bench.save(bench.system_dir("mt_parakeet"), c, s, sg)
    print(f"mt_parakeet {c}: {sum(1 for v in segs.values() if v)} of {len(segs)} sessions non-empty", flush=True)
