"""TS-ASR-Whisper decodes ($BENCH_OUT/tsasr/exp/<experiment>_<corpus>/) -> $BENCH_OUT/<system>/<corpus>/<session>.json.

    python benchmark/systems/dicow/collect.py {se_dicow|dicow_v3_3} CORPUS...

Sessions the model emitted nothing for (or DiariZen found no speech in) get an empty segment list.
"""
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bench  # noqa: E402

EXP = {"se_dicow": "SE-DiCoW_greedy_diarizen_full", "dicow_v3_3": "DiCoW_v3_3_greedy_diarizen_full"}
system = sys.argv[1]
for c in bench.expand(sys.argv[2:]):
    # TS-ASR-Whisper names the folder after the eval cutset's file: test_cutset/ for the benchmark's own cutsets
    done = glob.glob(f"{bench.OUT}/tsasr/exp/{EXP[system]}_{c}/test_*/0/all_session_wer.csv")
    if not done:
        print(f"{system} {c}: not decoded", flush=True)
        continue
    d = os.path.dirname(done[0])
    segs = {s: [] for s in bench.load_manifest(c)}
    for f in glob.glob(f"{d}/wer/*/tcp_wer_hyp.json"):
        for s in json.load(open(f)):
            segs.setdefault(s["session_id"], []).append({"start": float(s["start_time"]), "end": float(s["end_time"]),
                                                        "speaker": str(s["speaker"]), "text": s["words"]})
    out = bench.system_dir(system)
    for s, sg in segs.items():
        bench.save(out, c, s, sg)
    print(f"{system} {c}: {sum(1 for v in segs.values() if v)} of {len(segs)} sessions non-empty", flush=True)
