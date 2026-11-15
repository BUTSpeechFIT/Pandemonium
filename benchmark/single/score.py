"""Corpus WER (Whisper English normalizer, as the Open ASR Leaderboard) + RTFx of single/run.py outputs.

    python benchmark/single/score.py

Needs jiwer and whisper_normalizer."""
import glob
import json
import os
import sys

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import bench  # noqa: E402

ROOT = os.path.join(bench.OUT, "single")
SETS = ["librispeech.clean", "librispeech.other", "voxpopuli", "earnings22", "ami", "gigaspeech", "spgispeech"]
SYSTEMS = ["parakeet", "pand_oracle", "pand_nemotron"]
norm = EnglishTextNormalizer()

table = {}
for s in SYSTEMS:
    for d in SETS:
        rows = [json.loads(l) for f in glob.glob(f"{ROOT}/{s}/{d}.*.jsonl") for l in open(f)]
        rows = {r["id"]: r for r in rows}.values()
        refs, hyps = [], []
        for r in rows:
            ref = norm(r["ref"])
            if ref.strip():  # leaderboard drops empty references
                refs.append(ref)
                hyps.append(norm(r["hyp"]))
        if refs:
            wer = 100 * jiwer.wer(refs, hyps)
            rtfx = sum(r["dur"] for r in rows) / max(sum(r["sec"] for r in rows), 1e-9)
            table[s, d] = (wer, rtfx, len(refs))

print("| system | " + " | ".join(SETS) + " | mean | RTFx |")
print("|---" * (len(SETS) + 3) + "|")
for s in SYSTEMS:
    cells = [table.get((s, d)) for d in SETS]
    wers = [c[0] for c in cells if c]
    mean = f"{sum(wers) / len(wers):.2f}" if len(wers) == len(SETS) else "-"
    rt = [c[1] for c in cells if c]
    rtfx = f"{sum(rt) / len(rt):.0f}" if rt else "-"
    print(f"| {s} | " + " | ".join(f"{c[0]:.2f}" if c else "-" for c in cells) + f" | {mean} | {rtfx} |")
print("\nitems:", {f"{s}/{d}": table[s, d][2] for s, d in table})
