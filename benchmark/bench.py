"""Shared layout of the benchmark: where each eval set and each system's output live, and the job list.

Standard library only (plus soundfile for durations), so every system's environment can import it.

    $BENCH_DATA/<corpus>/manifest.json   {session: wav}       written by data/prepare_*.py
    $BENCH_DATA/<corpus>/ref.stm         whisper_nsf-normalized reference, one line per segment
    $BENCH_DATA/<corpus>/cutset.jsonl.gz ground-truth Lhotse cutset over the same wavs (DiCoW reads it)
    $BENCH_OUT/<system>/<corpus>/<session>.json
        {"session", "corpus", "duration", "decode_s", "error", "raw", "segments": [{start, end, speaker, text}]}

Session ids never contain '-' except a trailing partition tag ('-0', '-sdm', '-ch24'), which score.py strips on both
sides; the mixture and headset-mix sets are named without one.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("BENCH_DATA", os.path.join(HERE, "work", "data"))
OUT = os.environ.get("BENCH_OUT", os.path.join(HERE, "work", "out"))

MEETINGS = ["notsofar", "ami", "icsi", "mixer6", "sbc", "dipco"]
IHM_MIX = ["nsf_ihm_mix", "ami_ihm_mix"]
MIXTURES = ["lsmix1", "lsmix2", "lsmix3", "libri2mix", "libri2mix_noisy", "libri3mix", "libri3mix_noisy"]
GROUPS = {"meetings": MEETINGS, "ihm_mix": IHM_MIX, "mixtures": MIXTURES}


def corpus_dir(c):
    return os.path.join(DATA, c)


def manifest_path(c):
    return os.path.join(DATA, c, "manifest.json")


def ref_path(c):
    return os.path.join(DATA, c, "ref.stm")


def cutset_path(c):
    return os.path.join(DATA, c, "cutset.jsonl.gz")


def system_dir(system):
    return os.path.join(OUT, system)


def load_manifest(c):
    with open(manifest_path(c)) as f:
        return json.load(f)


def expand(names):
    """Corpus names, with 'meetings' / 'ihm_mix' / 'mixtures' standing for their whole group."""
    return [c for n in names for c in GROUPS.get(n, [n])]


def jobs(corpora, out, sessions=None, shard=0, nshards=1):
    """(duration, corpus, session, wav) still to do in `out`, longest first, LPT-split into `nshards` shards.

    `sessions` is a JSON file {corpus: [session, ...]} restricting the run (e.g. data/rtf_subset.json)."""
    import soundfile as sf
    only = json.load(open(sessions)) if sessions else None
    todo = []
    for c in expand(corpora):
        for sess, path in load_manifest(c).items():
            if only is None or sess in only.get(c, []):
                todo.append((sf.info(path).duration, c, sess, path))
    todo.sort(reverse=True)
    load = [0.0] * nshards
    mine = []
    for j in todo:
        k = min(range(nshards), key=load.__getitem__)
        load[k] += j[0]
        if k == shard:
            mine.append(j)
    return [j for j in mine if not os.path.exists(output_path(out, j[1], j[2]))]


def output_path(out, c, sess):
    return os.path.join(out, c, f"{sess}.json")


def save(out, c, sess, segments, duration=0.0, decode_s=0.0, error=None, raw="", **extra):
    os.makedirs(os.path.join(out, c), exist_ok=True)
    with open(output_path(out, c, sess), "w") as f:
        json.dump({"session": sess, "corpus": c, "duration": duration, "decode_s": decode_s, "error": error,
                   "raw": raw, "segments": segments, **extra}, f)

