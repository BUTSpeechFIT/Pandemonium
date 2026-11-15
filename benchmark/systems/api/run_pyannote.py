"""pyannoteAI diarization API (Precision-3 by default) over the RTFx subset, one recording at a time, timed end to end.

    python benchmark/systems/api/run_pyannote.py meetings --sessions benchmark/data/rtf_subset.json

Per recording: get an upload URL, PUT the wav, submit /v1/diarize, poll /v1/jobs/{id} every 0.5 s. Wall-clock covers
all of it (upload included), as for every other API system. Writes {OUT}/{corpus}/{session}.rttm and .json (timings
+ raw output), $BENCH_OUT/pyannote_<model>/ by default. Key read from $API_KEY_DIR/pyannote, never printed.
The published RTFx row used the RTFx subset only; its word error rates come from the paper's Precision-3 RTTMs
(systems/pandemonium/run.py rttm).
"""
import argparse, json, os, re, sys, time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bench  # noqa: E402

API = "https://api.pyannote.ai/v1"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("corpora", nargs="+")
    p.add_argument("--sessions")
    p.add_argument("--model", default="precision-3")
    p.add_argument("--out")
    a = p.parse_args()
    a.out = a.out or bench.system_dir(f"pyannote_{a.model}")

    key = open(os.path.join(os.path.expanduser(os.environ.get("API_KEY_DIR", "~/.config/api-asr")), "pyannote")).read().strip()
    auth = {"Authorization": f"Bearer {key}"}
    for _, c, sess, path in sorted(bench.jobs(a.corpora, a.out, a.sessions), key=lambda j: j[1:3]):
        os.makedirs(f"{a.out}/{c}", exist_ok=True)
        obj = "media://pandemonium-rtf/" + re.sub(r"[^A-Za-z0-9\-_.]", "_", os.path.basename(path))
        t0 = time.time()
        r = requests.post(f"{API}/media/input", json={"url": obj}, headers=auth, timeout=60)
        r.raise_for_status()
        with open(path, "rb") as f:
            requests.put(r.json()["url"], data=f, timeout=600).raise_for_status()
        t_up = time.time()
        r = requests.post(f"{API}/diarize", json={"url": obj, "model": a.model}, headers=auth, timeout=60)
        r.raise_for_status()
        job = r.json()["jobId"]
        while True:
            time.sleep(0.5)
            j = requests.get(f"{API}/jobs/{job}", headers=auth, timeout=60)
            j.raise_for_status()
            j = j.json()
            if j["status"] in ("succeeded", "failed", "canceled"):
                break
        t_done = time.time()
        assert j["status"] == "succeeded", (sess, j["status"], j.get("output"))
        segs = j["output"]["diarization"]
        with open(f"{a.out}/{c}/{sess}.rttm", "w") as f:
            for s in segs:
                f.write(f"SPEAKER {sess} 1 {s['start']:.3f} {s['end'] - s['start']:.3f} <NA> <NA> {s['speaker']} <NA> <NA>\n")
        json.dump({"session": sess, "corpus": c, "model": a.model, "job": job, "upload_s": t_up - t0,
                   "diarize_s": t_done - t_up, "wall_s": t_done - t0, "warning": j.get("warning"),
                   "output": j["output"]}, open(f"{a.out}/{c}/{sess}.json", "w"))
        print(f"{c} {sess} upload {t_up - t0:.1f}s diarize {t_done - t_up:.1f}s speakers "
              f"{len({s['speaker'] for s in segs})}", flush=True)


if __name__ == "__main__":
    main()
