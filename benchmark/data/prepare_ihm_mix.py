"""Headset-mix versions of the NOTSOFAR-1 and AMI eval sets: every participant's close-talk microphone summed into
one signal, i.e. overlapped speech without far-field reverberation.

    python benchmark/data/prepare_ihm_mix.py --nsf-meetings DIR --ami-ihm-mix DIR

    --nsf-meetings  NOTSOFAR-1 eval set's `MTG` directory (`<MTG_id>/close_talk/*.wav`, one wav per participant,
                    equal length, same time base as the far-field recording); mt-asr-data-prep downloads it to
                    data/nsf/ihm/benchmark-datasets/eval_set/240629.1_eval_small_with_GT/MTG
    --ami-ihm-mix   AMI's official headset mixes, `<meeting>/audio/<meeting>.Mix-Headset.wav`
                    (mt-asr-data-prep: data/ami/ihm-mix/wav_db)

AMI ships a headset mix (copied in, so the set is self-contained); NOTSOFAR-1 does not, so its channels are summed
here, scaled down only if the sum would clip. Run prepare_meetings.py for `notsofar` and `ami` first: the reference is the far-field one (the headset and
distant transcripts of a meeting share their timestamps; of NOTSOFAR-1's two devices per meeting, the first is used),
renamed per session. Sessions are `nsfihm_<MTG_id>` and `amiihm_<meeting>`.

Writes $BENCH_DATA/{nsf_ihm_mix,ami_ihm_mix}/: audio/, manifest.json, ref.stm, cutset.jsonl.gz.
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from lhotse import CutSet, MonoCut, Recording, SupervisionSegment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench  # noqa: E402


def far_field_refs(corpus, key):
    """{meeting: [stm fields]} from the far-field reference, one source session per meeting."""
    by = defaultdict(lambda: defaultdict(list))
    for line in open(bench.ref_path(corpus)):
        f = line.split(None, 5)
        by[key(f[0])][f[0]].append(f)
    return {m: sess[min(sess)] for m, sess in by.items()}


def nsf_mix(root):
    def make(meeting, path):
        if not os.path.exists(path):
            chans = [sf.read(p, dtype="float32")[0] for p in sorted(glob.glob(f"{root}/{meeting}/close_talk/*.wav"))]
            assert chans and len({len(c) for c in chans}) == 1, meeting
            mix = np.sum(chans, 0)
            mix /= max(1.0, np.abs(mix).max() / 0.99)
            sf.write(path, mix, 16000, subtype="PCM_16")
        return path
    return make


def prepare(c, refs, audio):
    d = Path(bench.corpus_dir(c))
    (d / "audio").mkdir(parents=True, exist_ok=True)
    manifest, stm, cuts = {}, [], []
    for meeting, lines in sorted(refs.items()):
        sess = f"{c.split('_')[0]}ihm_{meeting}"
        manifest[sess] = audio(meeting, str(d / "audio" / f"{sess}.wav"))
        rec = Recording.from_file(manifest[sess], recording_id=sess)
        sups = []
        for k, (_, ch, spk, st, en, *text) in enumerate(lines):
            st, en = float(st), min(float(en), rec.duration)
            stm.append(f"{sess} {ch} {spk} {st:.3f} {en:.3f} {' '.join(text).strip()}")
            sups.append(SupervisionSegment(id=f"{sess}_{k}", recording_id=sess, start=st, duration=en - st, channel=0,
                                           text=" ".join(text).strip(), language="English", speaker=spk))
        cuts.append(MonoCut(id=sess, start=0.0, duration=rec.duration, channel=0, recording=rec, supervisions=sups))
    json.dump(manifest, open(bench.manifest_path(c), "w"))
    open(bench.ref_path(c), "w").write("\n".join(stm) + "\n")
    CutSet.from_cuts(cuts).to_file(bench.cutset_path(c))
    print(f"{c}: {len(manifest)} sessions, {sum(x.duration for x in cuts) / 3600:.2f} h, {len(stm)} reference "
          f"segments -> {d}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nsf-meetings", required=True)
    ap.add_argument("--ami-ihm-mix", required=True)
    a = ap.parse_args()
    prepare("nsf_ihm_mix", far_field_refs("notsofar", lambda s: re.search(r"MTG_\d+", s).group(0)),
            nsf_mix(a.nsf_meetings))
    prepare("ami_ihm_mix", far_field_refs("ami", lambda s: re.match(r"sdm_(\w+?)-\d+$", s).group(1)),
            lambda m, path: shutil.copyfile(f"{a.ami_ihm_mix}/{m}/audio/{m}.Mix-Headset.wav", path))


if __name__ == "__main__":
    main()
