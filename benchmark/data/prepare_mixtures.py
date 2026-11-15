"""Render the LibriSpeechMix / LibriMix test mixtures into the benchmark layout.

    python benchmark/data/prepare_mixtures.py /path/to/mt-asr-data-prep/manifests [corpus ...]

The source cutsets are the ones BUTSpeechFIT/mt-asr-data-prep builds (`./prepare.sh --datasets librispeech,
librispeechmix,librimix`); only their test-clean parts are read. Each mixture becomes one mono wav (multi-channel
cuts are averaged), named `<corpus>_<index>` in cutset order (no '-', so the scorer leaves the name alone).

Writes, under $BENCH_DATA/<corpus>/: audio/*.wav, manifest.json, ref.stm (whisper_nsf, as for the meetings),
cutset.jsonl.gz (ground truth over the rendered wavs) and cut_ids.json (session -> original cut id).
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import soundfile as sf
from lhotse import CutSet, MonoCut, Recording, SupervisionSegment, load_manifest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))

import bench  # noqa: E402
from src.data.text_norm import get_text_norm  # noqa: E402

SOURCES = {
    "lsmix1": "librispeechmix/librispeechmix_cutset_test-clean-1mix.jsonl.gz",
    "lsmix2": "librispeechmix/librispeechmix_cutset_test-clean-2mix.jsonl.gz",
    "lsmix3": "librispeechmix/librispeechmix_cutset_test-clean-3mix.jsonl.gz",
    "libri2mix": "librimix/librimix_cutset_libri2mix_test-clean.jsonl.gz",
    "libri2mix_noisy": "librimix/librimix_cutset_libri2mix_test-clean_noisy.jsonl.gz",
    "libri3mix": "librimix/librimix_cutset_libri3mix_test-clean.jsonl.gz",
    "libri3mix_noisy": "librimix/librimix_cutset_libri3mix_test-clean_noisy.jsonl.gz",
}


def render(job):
    cut, path = job
    if not os.path.exists(path):
        audio = cut.load_audio()
        sf.write(path, audio.mean(0) if audio.ndim > 1 else audio, cut.sampling_rate, subtype="PCM_16")
    return path


def prepare(c, src_dir, norm):
    cuts = load_manifest(f"{src_dir}/{SOURCES[c]}")
    assert all(cut.sampling_rate == 16000 for cut in cuts)
    d = Path(bench.corpus_dir(c))
    (d / "audio").mkdir(parents=True, exist_ok=True)
    manifest, ids, stm = {}, {}, []
    for i, cut in enumerate(cuts):
        sess = f"{c}_{i:05d}"
        ids[sess] = cut.id
        manifest[sess] = str(d / "audio" / f"{sess}.wav")
        for s in cut.supervisions:
            text = norm(s.text).strip()
            if text:
                stm.append(f"{sess} 1 {s.speaker} {max(s.start, 0):.3f} {min(s.end, cut.duration):.3f} {text}")
    with ProcessPoolExecutor(16) as ex:
        list(ex.map(render, zip(cuts, manifest.values()), chunksize=32))

    gt = []
    for (sess, path), cut in zip(manifest.items(), cuts):
        rec = Recording.from_file(path, recording_id=sess)
        sups = [SupervisionSegment(id=f"{sess}_{k}", recording_id=sess, start=s.start,
                                   duration=min(s.end, rec.duration) - s.start, channel=0, text=s.text,
                                   language="English", speaker=s.speaker)
                for k, s in enumerate(cut.supervisions)]
        gt.append(MonoCut(id=sess, start=0.0, duration=rec.duration, channel=0, recording=rec, supervisions=sups))

    json.dump(manifest, open(bench.manifest_path(c), "w"))
    json.dump(ids, open(d / "cut_ids.json", "w"))
    open(bench.ref_path(c), "w").write("\n".join(stm) + "\n")
    CutSet.from_cuts(gt).to_file(bench.cutset_path(c))
    print(f"{c}: {len(manifest)} mixtures, {len(stm)} reference segments -> {d}", flush=True)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    norm = get_text_norm("whisper_nsf")
    for c in sys.argv[2:] or bench.MIXTURES:
        prepare(c, sys.argv[1], norm)


if __name__ == "__main__":
    main()
