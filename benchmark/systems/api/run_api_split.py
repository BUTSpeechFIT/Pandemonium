"""Fallback for recordings an API refuses whole: transcribe two overlapping halves and stitch them.

    python benchmark/systems/api/run_api_split.py PROVIDER CORPUS SESSION [--overlap 120]

Used for the two sessions Mistral refused whole (ICSI Bro021, DiPCo S25).

Halves are [0, mid + overlap/2] and [mid - overlap/2, end]. Speaker labels are independent per request, so the
second half's speakers are mapped onto the first's by Hungarian matching on how long their segments overlap in
time inside the shared window, using only the provider's own output. Unmatched speakers keep a fresh label.
Each half then contributes the segments starting on its side of the midpoint. `decode_s` is the sum of both
requests; the JSON records `split` so these sessions can be reported as such.
"""
import argparse, json, os, sys, tempfile, time

import numpy as np
import soundfile as sf
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_api import PROVIDERS, bench, read_key


def overlap(segs_a, segs_b, lo, hi):
    def clip(s):
        return max(s["start"], lo), min(s["end"], hi)
    tot = 0.0
    for a in segs_a:
        a0, a1 = clip(a)
        for b in segs_b:
            b0, b1 = clip(b)
            tot += max(0.0, min(a1, b1) - max(a0, b0))
    return tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("provider", choices=list(PROVIDERS))
    p.add_argument("corpus")
    p.add_argument("session")
    p.add_argument("--overlap", type=float, default=120.0)
    p.add_argument("--out")
    a = p.parse_args()
    a.out = a.out or bench.system_dir(a.provider)
    key = read_key(a.provider)
    path = bench.load_manifest(a.corpus)[a.session]
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(1)
    dur = len(audio) / sr
    mid = dur / 2
    spans = [(0.0, mid + a.overlap / 2), (mid - a.overlap / 2, dur)]

    halves, t_total, resps = [], 0.0, []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (s, e) in enumerate(spans):
            f = os.path.join(tmp, f"{a.session}_part{i}.wav")
            sf.write(f, audio[int(s * sr):int(e * sr)], sr)
            for attempt in range(12):
                t0 = time.time()
                try:
                    resp, segs = PROVIDERS[a.provider](f, key)
                    break
                except Exception as ex:
                    print(f"  part {i} attempt {attempt + 1} failed: {repr(ex)[:200]}", flush=True)
                    if not any(x in repr(ex) for x in ("HTTP 429", "HTTP 5", "Timeout", "Connection")) or attempt == 11:
                        raise
                    time.sleep(min(60 * (attempt + 1), 300))
            t_total += time.time() - t0
            resps.append(resp)
            halves.append([{**g, "start": g["start"] + s, "end": g["end"] + s} for g in segs])
            print(f"  part {i} [{s:.0f}, {e:.0f}] s: {len(segs)} segs", flush=True)

    lo, hi = spans[1][0], spans[0][1]
    spk_a = sorted({g["speaker"] for g in halves[0] if g["speaker"]})
    spk_b = sorted({g["speaker"] for g in halves[1] if g["speaker"]})
    cost = np.array([[-overlap([g for g in halves[0] if g["speaker"] == x],
                               [g for g in halves[1] if g["speaker"] == y], lo, hi) for y in spk_b] for x in spk_a])
    mapping = {}
    if cost.size:
        for r, c in zip(*linear_sum_assignment(cost)):
            if cost[r, c] < 0:  # only link speakers that actually co-occur in the overlap
                mapping[spk_b[c]] = spk_a[r]
    fresh = iter(f"B_{y}" for y in spk_b)
    for y in spk_b:
        mapping.setdefault(y, next(fresh))
    segs = [g for g in halves[0] if g["start"] < mid] + \
           [{**g, "speaker": mapping.get(g["speaker"], g["speaker"])} for g in halves[1] if g["start"] >= mid]
    print(f"speaker map (part 1 -> part 0): {mapping}", flush=True)

    os.makedirs(f"{a.out}/{a.corpus}", exist_ok=True)
    json.dump({"session": a.session, "corpus": a.corpus, "duration": dur, "decode_s": t_total, "error": None,
               "raw": "", "segments": segs, "response": resps,
               "split": {"spans": spans, "mid": mid, "speaker_map": mapping}},
              open(f"{a.out}/{a.corpus}/{a.session}.json", "w"))
    print(f"{a.corpus} {a.session} {dur / 60:.1f} min -> {t_total:.1f} s, {len(segs)} segs (split)", flush=True)


if __name__ == "__main__":
    main()
