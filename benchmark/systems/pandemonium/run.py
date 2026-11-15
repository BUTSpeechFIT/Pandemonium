"""Pandemonium over whole recordings, conditioned on one of three diarizations.

    python benchmark/systems/pandemonium/run.py nemotron meetings ihm_mix mixtures
    python benchmark/systems/pandemonium/run.py oracle   meetings
    python benchmark/systems/pandemonium/run.py rttm     notsofar --rttm NOTSOFAR....rttm [--session-offset N]

  nemotron  the released pipeline (demo/pipeline.py): Nemotron-3-Diarization in its offline setting, its *soft*
            speaker probabilities conditioning the model (--hard: thresholded segments instead)
  oracle    reference speaker activity, rasterized from ref.stm
  rttm      any external diarization as hard segments, e.g. the pyannoteAI Precision-3 RTTMs of the paper. RTTM
            recording ids are matched to sessions by dropping the 'sdm_' prefix and the '-N' / '-sdm' / '-chN' tag
            and reading '/' as '_' (MTG_32000/sc_meetup_0, EN2002a, SBC001); --session-offset adds N to the
            trailing number of each RTTM id (pyannote's DiPCo files number the eval sessions S01-S08, ours S25-S32)

Output: $BENCH_OUT/pandemonium_<cond>/<corpus>/<session>.json (--out to override), one segment per word with the
model's own word timestamps. Run in the demo environment (demo/requirements.txt), from any directory.
"""
import argparse
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(ROOT / "demo"))
# The demo parks both models in host RAM between requests (one shared GPU); a benchmark keeps them on the card.
os.environ.setdefault("PANDEMONIUM_OFFLOAD", "0")

import bench  # noqa: E402

import librosa  # noqa: E402
import numpy as np  # noqa: E402


def segments(words_per_speaker):
    """One segment per word: the model times every word, so tcpWER is scored on those times rather than on times
    interpolated across a grouped utterance."""
    return [{"start": w.start, "end": w.end, "speaker": str(spk), "text": w.text}
            for spk, words in enumerate(words_per_speaker) for w in words]


def activity(spans, n, frame):
    """(speakers, frames) 0/1 activity from {speaker: [(start, end)]}; every span claims at least one frame."""
    act = np.zeros((len(spans), n), np.float32)
    for g, s in enumerate(sorted(spans)):
        for st, en in spans[s]:
            act[g, int(st / frame): max(int(np.ceil(en / frame)), int(st / frame) + 1)] = 1.0
    return act


def ref_spans(c):
    spans = defaultdict(lambda: defaultdict(list))
    for line in open(bench.ref_path(c)):
        sess, _, spk, st, en = line.split()[:5]
        spans[sess][spk].append((float(st), float(en)))
    return spans


def rttm_spans(path, offset):
    spans = defaultdict(lambda: defaultdict(list))
    for line in open(path):
        f = line.split()
        rid = re.sub(r"\d+$", lambda m: f"{int(m.group(0)) + offset:0{len(m.group(0))}d}", f[1]) if offset else f[1]
        spans[rid.replace("/", "_")][f[7]].append((float(f[3]), float(f[3]) + float(f[4])))
    return spans


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cond", choices=["nemotron", "oracle", "rttm"])
    ap.add_argument("corpora", nargs="+", help="corpus names or groups: meetings, ihm_mix, mixtures")
    ap.add_argument("--hard", action="store_true", help="nemotron: thresholded segments instead of probabilities")
    ap.add_argument("--rttm")
    ap.add_argument("--session-offset", type=int, default=0)
    ap.add_argument("--sessions", help="JSON {corpus: [session, ...]} to restrict to (data/rtf_subset.json for RTFx)")
    ap.add_argument("--warmup", help="wav decoded once, untimed, before the first session (for RTFx)")
    ap.add_argument("--out")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1, help="split the recordings over N processes (or GPUs)")
    a = ap.parse_args()
    if a.cond == "rttm" and not a.rttm:
        ap.error("rttm needs --rttm")
    name = {"nemotron": "pandemonium_nemotron" + ("_hard" if a.hard else ""), "oracle": "pandemonium_oracle",
            "rttm": "pandemonium_rttm"}[a.cond]
    out = a.out or bench.system_dir(name)

    from pandemonium import FRAME_SHIFT, Pandemonium
    from pipeline import COMPUTE_DTYPE, DemoPipeline

    todo = bench.jobs(a.corpora, out, a.sessions, a.shard, a.nshards)
    print(f"{len(todo)} sessions to do -> {out}", flush=True)
    if not todo:
        return
    rttm = rttm_spans(a.rttm, a.session_offset) if a.rttm else None
    refs = {}
    pipe = DemoPipeline("cuda")
    if a.warmup:  # the first GPU run pays a one-off CUDA start-up cost
        w, _ = librosa.load(a.warmup, sr=16000, mono=True)
        pipe.transcribe(w, pipe.diarize(w))
    for dur, c, sess, path in todo:
        audio, _ = librosa.load(path, sr=16000, mono=True)
        n = Pandemonium.num_frames(len(audio))
        t0 = time.time()
        extra = {}
        if a.cond == "nemotron":
            d = pipe.diarize(audio)
            words = pipe.transcribe(audio, d, soft=not a.hard) if d.num_speakers else []
            extra["nspk"] = d.num_speakers
        else:
            if a.cond == "oracle":
                if c not in refs:
                    refs[c] = ref_spans(c)
                spans = refs[c][sess]
            else:
                key = re.sub(r"-(\d+|sdm|ch\d+)$", "", sess.removeprefix("sdm_"))
                if key not in rttm:
                    raise SystemExit(f"no RTTM recording for session {sess} (looked for {key})")
                spans = rttm[key]
            words = pipe.asr.transcribe(audio, activity(spans, n, FRAME_SHIFT), dtype=COMPUTE_DTYPE) if spans else []
            extra["nspk"] = len(spans)
        bench.save(out, c, sess, segments(words), duration=dur, decode_s=time.time() - t0, **extra)
        print(f"{c} {sess} {dur / 60:.1f} min -> {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
