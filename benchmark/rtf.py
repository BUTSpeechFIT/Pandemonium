"""RTFx = audio duration / processing time, over the RTFx subset (data/rtf_subset.json: one median-length recording
per meeting set, 10,022 s in all).

    python benchmark/rtf.py $BENCH_OUT/rtf/<system> [--plus-diarizen $BENCH_OUT/rtf/diarizen]

Each system is first run alone on one GPU over the subset with its runner's --sessions data/rtf_subset.json and
--warmup <a NOTSOFAR-1 recording outside the subset>, writing to $BENCH_OUT/rtf/<system>. `decode_s` of every
output is summed; for a system that consumes DiariZen's diarization, --plus-diarizen adds DiariZen's own time
(diarize.py's timing.json), since processing time includes diarization. API figures are client wall-clock times.
"""
import argparse
import glob
import json
import os

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("system")
ap.add_argument("--plus-diarizen")
a = ap.parse_args()
rows = [json.load(open(f)) for f in glob.glob(f"{a.system}/*/*.json")]
audio = sum(r["duration"] for r in rows)
asr = sum(r["decode_s"] for r in rows)
diar = 0.0
if a.plus_diarizen:
    for r in rows:
        diar += json.load(open(os.path.join(a.plus_diarizen, r["corpus"], "timing.json")))[r["session"]]
print(f"{os.path.basename(a.system.rstrip('/'))}: {len(rows)} recordings, {audio:.0f} s audio, "
      f"{asr:.1f} s{f' + {diar:.1f} s diarization' if diar else ''} -> RTFx {audio / (asr + diar):.1f}")
