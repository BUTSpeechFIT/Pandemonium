"""Single-speaker English (Open ASR Leaderboard test sets): Pandemonium vs its Parakeet-v3 base.

    python benchmark/single/run.py pandemonium SET [RANK WORLD]   # whole clip as one speaker + Nemotron, one pass
    python benchmark/single/run.py parakeet SET                    # nvidia/parakeet-tdt-0.6b-v3, batch 32

SET is one of librispeech.clean, librispeech.other, voxpopuli, earnings22, ami, gigaspeech, spgispeech (TED-LIUM is no
longer in the dataset). Data: hf-audio/esb-datasets-test-only-sorted in $BENCH_DATA/esb, e.g.
    hf download hf-audio/esb-datasets-test-only-sorted --repo-type dataset --local-dir $BENCH_DATA/esb
Writes $BENCH_OUT/single/{system}/{SET}.{rank}.jsonl (id, ref, hyp, dur, sec), resumable by id. Run in the demo
environment; RANK/WORLD shard a set over several GPUs.
"""
import glob
import io
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "demo"))
os.environ.setdefault("PANDEMONIUM_OFFLOAD", "0")
import bench  # noqa: E402

DATA, ROOT = os.path.join(bench.DATA, "esb"), os.path.join(bench.OUT, "single")


def load_set(name):
    folder, split = (name.split(".") + ["test"])[:2]
    pat = f"{DATA}/{folder}/" + (f"test.{split}-*.parquet" if folder == "librispeech" else "test-*.parquet")
    for f in sorted(glob.glob(pat)):
        for r in pq.read_table(f, columns=["id", "text", "audio", "audio_length_s"]).to_pylist():
            yield r


def decode(r):
    a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    if sr != 16000:
        from scipy.signal import resample_poly
        g = np.gcd(sr, 16000)
        a = resample_poly(a, 16000 // g, sr // g).astype(np.float32)
    return a


def writer(path):
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["id"] for l in open(path)}
    return done, open(path, "a")


def words_text(ws):
    return " ".join(w.text for w in sorted(ws, key=lambda w: w.start))


def run_pandemonium(name, rank, world):
    from pipeline import DemoPipeline, COMPUTE_DTYPE
    from pandemonium import Pandemonium

    pipe = DemoPipeline("cuda")
    os.makedirs(f"{ROOT}/pand_oracle", exist_ok=True)
    os.makedirs(f"{ROOT}/pand_nemotron", exist_ok=True)
    done_o, fo = writer(f"{ROOT}/pand_oracle/{name}.{rank}.jsonl")
    done_n, fn = writer(f"{ROOT}/pand_nemotron/{name}.{rank}.jsonl")
    for i, r in enumerate(load_set(name)):
        if i % world != rank or (r["id"] in done_o and r["id"] in done_n):
            continue
        a = decode(r)
        n = Pandemonium.num_frames(len(a))
        base = {"id": r["id"], "ref": r["text"], "dur": len(a) / 16000}
        if r["id"] not in done_o:
            t = time.perf_counter()
            ws = pipe.asr.transcribe(a, np.ones((1, n), np.float32), dtype=COMPUTE_DTYPE)
            fo.write(json.dumps({**base, "hyp": words_text(ws[0]), "sec": time.perf_counter() - t}) + "\n")
        if r["id"] not in done_n:
            t = time.perf_counter()
            d = pipe.diarize(a)
            # Single-speaker reference: score every detected speaker's words merged in time order.
            ws = pipe.transcribe(a, d) if d.num_speakers else []
            hyp = words_text([w for s in ws for w in s])
            fn.write(json.dumps({**base, "hyp": hyp, "nspk": d.num_speakers,
                                 "sec": time.perf_counter() - t}) + "\n")
        if i % 500 < world:
            fo.flush(); fn.flush()
            print(name, rank, i, flush=True)
    fo.close(); fn.close()
    print("DONE", name, rank, flush=True)


def run_parakeet(name, bs=32):
    import torch
    from nemo.collections.asr.models import ASRModel

    m = ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3", map_location="cuda").eval()
    os.makedirs(f"{ROOT}/parakeet", exist_ok=True)
    done, f = writer(f"{ROOT}/parakeet/{name}.0.jsonl")
    rows = [r for r in load_set(name) if r["id"] not in done]
    for k in range(0, len(rows), bs):
        chunk = rows[k:k + bs]
        audio = [decode(r) for r in chunk]
        t = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
            hyps = m.transcribe(audio, batch_size=bs, verbose=False)
        sec = (time.perf_counter() - t) / len(chunk)
        for r, a, h in zip(chunk, audio, hyps):
            f.write(json.dumps({"id": r["id"], "ref": r["text"], "dur": len(a) / 16000,
                                "hyp": getattr(h, "text", h), "sec": sec}) + "\n")
        f.flush()
    f.close()
    print("DONE", name, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "pandemonium":
        run_pandemonium(sys.argv[2], *(map(int, sys.argv[3:5]) if len(sys.argv) > 3 else (0, 1)))
    else:
        run_parakeet(sys.argv[2])
