"""DiariZen (BUT-FIT/diarizen-wavlm-large-s80-md-v2) over the benchmark sets: the diarization SE-DiCoW, DiCoW v3.3
and CocktailASR are conditioned on.

    python benchmark/systems/diarizen/diarize.py meetings ihm_mix mixtures [--device cuda:0]

Writes $BENCH_OUT/diarizen/<corpus>/segments.json ({session: [[start, end, speaker], ...]}), timing.json ({session:
seconds}, for RTFx) and rttm/. Run in the
DiariZen environment (see setup.sh). VBx clustering is CPU-bound: cap OMP_NUM_THREADS when running several at once,
or they slow each other down threefold.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bench  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpora", nargs="+")
    ap.add_argument("--model", default="BUT-FIT/diarizen-wavlm-large-s80-md-v2")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    import torch
    # torch>=2.6 defaults torch.load to weights_only=True, which the pyannote/lightning checkpoint loader trips on.
    _load = torch.load
    torch.load = lambda *x, **k: _load(*x, **{**k, "weights_only": False})
    from diarizen.pipelines.inference import DiariZenPipeline

    pipe = None
    for c in bench.expand(a.corpora):
        out = Path(bench.system_dir("diarizen")) / c
        if (out / "segments.json").exists():
            print(f"{c}: done", flush=True)
            continue
        (out / "rttm").mkdir(parents=True, exist_ok=True)
        if pipe is None:
            pipe = DiariZenPipeline.from_pretrained(a.model, rttm_out_dir=str(out / "rttm"))
            pipe.to(torch.device(a.device))
        pipe.rttm_out_dir = str(out / "rttm")
        segments, timing, t0 = {}, {}, time.time()
        for sess, wav in bench.load_manifest(c).items():
            t = time.time()
            ann = pipe(wav, sess_name=sess)
            timing[sess] = time.time() - t
            segments[sess] = [[float(t.start), float(t.end), f"speaker_{k}"]
                              for t, _, k in ann.itertracks(yield_label=True) if t.end > t.start]
        json.dump(segments, open(out / "segments.json", "w"))
        json.dump(timing, open(out / "timing.json", "w"))
        print(f"{c}: {len(segments)} sessions in {time.time() - t0:.0f} s -> {out}", flush=True)


if __name__ == "__main__":
    main()
