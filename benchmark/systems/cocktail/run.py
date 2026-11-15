"""Xiaomi-CocktailASR-1 (target-speaker ASR) over the comparison sets, driven by the DiariZen diarization.

    python benchmark/systems/cocktail/run.py meetings ihm_mix mixtures [--shard K --nshards N] [--sessions FILE]
        [--warmup WAV]

CocktailASR transcribes one enrolled speaker from a short clip and returns text only, so a diarizer supplies
both. Per session, from DiariZen's segments (the same ones SE-DiCoW is conditioned on):

* enrollment: each speaker's longest stretch with no other speaker active, its middle REF_S seconds. The model's
  own extractor crops anything longer than 4 s to a *random* 1-4 s window, so a clip of exactly 4 s is used as is
  and the run is deterministic.
* targets: the speaker's segments, merged across gaps <= MERGE_GAP s, grouped into chunks of at most MAX_CHUNK s
  (a longer single segment is split evenly), padded by PAD s. Each chunk is the full mixture over that span; the
  model picks out the speaker. The chunk span is the segment's timestamp.

Needs diarizen/diarize.py to have run. Output: $BENCH_OUT/cocktail/<corpus>/<session>.json; `decode_s` is ASR time
only (DiariZen is timed apart). Run in the CocktailASR environment (see setup.sh).
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bench  # noqa: E402

SR, REF_S, MERGE_GAP, PAD = 16000, 4.0, 1.0, 0.2
MAX_CHUNK = float(os.environ.get("COCKTAIL_MAX_CHUNK", 20))


# CocktailASR writes contractions without the apostrophe ("dont", "Im"), which whisper_nsf then cannot expand
# ("dont" stays one word against the reference's "do not"). Restore the unambiguous ones; "its", "well", "id",
# "hes", "ill" are real words too and are left alone. `text_raw` keeps the model's output untouched.
CONTRACTIONS = {w.replace("'", ""): w for w in (
    "i'm you're don't that's can't won't they're isn't didn't doesn't wasn't weren't aren't i've you've we've "
    "they've you'd it'll you'll they'll what's there's here's let's couldn't wouldn't shouldn't haven't hasn't "
    "hadn't she's who's where's ain't y'all").split()}


def restore(text):
    return " ".join(CONTRACTIONS.get(w.lower(), w) for w in text.split())


def load(path):
    a, sr = sf.read(path, dtype="float32", always_2d=True)
    a = torch.from_numpy(a.mean(1))
    if sr != SR:
        a = torchaudio.functional.resample(a, sr, SR)
    return a.numpy()


def subtract(iv, others):
    """Parts of interval iv not covered by any interval in others."""
    parts = [iv]
    for s, e in others:
        parts = [p for a, b in parts for p in ([(a, min(b, s))] if a < s else []) + ([(max(a, e), b)] if e < b else [])
                 if p[1] > p[0]]
    return parts


def enrollment(segs, spk):
    own = [(s, e) for s, e, k in segs if k == spk]
    others = [(s, e) for s, e, k in segs if k != spk]
    clean = [p for iv in own for p in subtract(iv, others)]
    s, e = max(clean or own, key=lambda p: p[1] - p[0])  # a speaker always overlapped falls back to its longest turn
    m = (s + e) / 2
    return max(s, m - REF_S / 2), min(e, m + REF_S / 2)


def chunks(segs, spk):
    own = sorted((s, e) for s, e, k in segs if k == spk)
    merged = []
    for s, e in own:
        if merged and s - merged[-1][1] <= MERGE_GAP:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out = []
    for s, e in merged:
        n = int(np.ceil((e - s) / MAX_CHUNK))
        out += [(s + i * (e - s) / n, s + (i + 1) * (e - s) / n) for i in range(n)]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("corpora", nargs="+")
    p.add_argument("--out", default=bench.system_dir("cocktail"))
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--sessions")
    p.add_argument("--warmup")
    a = p.parse_args()

    todo = bench.jobs(a.corpora, a.out, a.sessions, a.shard, a.nshards)
    print(f"shard {a.shard}/{a.nshards}: {len(todo)} to do", flush=True)
    if not todo:
        return

    from transformers import AutoModel
    model = AutoModel.from_pretrained("Ease3/Xiaomi-CocktailASR-1", trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).cuda().eval()
    if a.warmup:
        w = load(a.warmup)
        model(w[SR * 10:SR * 25], w[SR * 30:SR * 34])

    seg_cache = {}
    for dur, c, sess, path in todo:
        if c not in seg_cache:
            seg_cache[c] = json.load(open(f"{bench.system_dir('diarizen')}/{c}/segments.json"))
        segs = seg_cache[c].get(sess, [])
        audio = load(path)
        cut = lambda s, e: audio[int(max(0, s) * SR):int(min(dur, e) * SR)]
        out_segs, n_chunks, t_asr = [], 0, 0.0
        for spk in sorted({k for _, _, k in segs}):
            ref = cut(*enrollment(segs, spk))
            for s, e in chunks(segs, spk):
                s, e = max(0.0, s - PAD), min(dur, e + PAD)
                torch.cuda.synchronize()
                t0 = time.time()
                with torch.inference_mode():
                    text = model(cut(s, e), ref)
                torch.cuda.synchronize()
                t_asr += time.time() - t0
                n_chunks += 1
                if text.strip():
                    out_segs.append({"start": round(s, 3), "end": round(e, 3), "speaker": spk,
                                     "text": restore(text.strip()), "text_raw": text.strip()})
        bench.save(a.out, c, sess, out_segs, duration=dur, decode_s=t_asr, chunks=n_chunks)
        print(f"{c} {sess} {dur / 60:.1f} min -> {t_asr:.1f} s ({dur / max(t_asr, 1e-6):.1f}x), "
              f"{n_chunks} chunks, {len(out_segs)} non-empty", flush=True)


if __name__ == "__main__":
    main()
