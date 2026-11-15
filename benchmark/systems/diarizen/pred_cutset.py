"""The ground-truth cutset with DiariZen's speakers in place of the reference ones: what SE-DiCoW and DiCoW decode.

    python benchmark/systems/diarizen/pred_cutset.py meetings ihm_mix mixtures

Text is a placeholder ('x'); only the timing and speaker labels are used. A session DiariZen found no speech in is
dropped (and scores as all deletions). Writes $BENCH_OUT/diarizen/<corpus>/pred_cutset.jsonl.gz. Needs lhotse.
"""
import json
import sys
from pathlib import Path

from lhotse import CutSet, SupervisionSegment

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bench  # noqa: E402

for c in bench.expand(sys.argv[1:]):
    d = Path(bench.system_dir("diarizen")) / c
    segments = json.load(open(d / "segments.json"))
    out, missing = [], 0
    for cut in CutSet.from_file(bench.cutset_path(c)):
        rid = cut.recording.id
        if not segments.get(rid):
            missing += 1
            continue
        sups = []
        for j, (st, en, spk) in enumerate(segments[rid]):
            st_rel = max(0.0, st - cut.start)
            dur = min(en - st, cut.duration - st_rel)
            if dur > 0:
                sups.append(SupervisionSegment(id=f"{rid}_pred_{j}", recording_id=rid, start=st_rel, duration=dur,
                                               channel=cut.channel, text="x", speaker=str(spk), language="en"))
        new = cut.drop_supervisions()
        new.supervisions = sups
        out.append(new)
    CutSet.from_cuts(out).to_file(d / "pred_cutset.jsonl.gz")
    print(f"{c}: {len(out)} cuts (no speech found in {missing})", flush=True)
