"""cpWER and tcpWER (5 s collar) of system outputs against the benchmark references.

    python benchmark/score.py $BENCH_OUT/<system> [corpus|group ...] [--partial]

Groups: meetings, ihm_mix, mixtures (default: every prepared corpus the system has output for). Hypothesis text is
whisper_nsf-normalized like the reference; segments without a speaker (non-speech events such as "[Music]") are
dropped. A session with no output file counts as all deletions unless --partial, which scores finished sessions
only (for a run still in progress). Writes <system>/scores.json and prints one RESULT line per corpus.

Run in an environment with meeteval and this repository's requirements (src.data.text_norm).
"""
import argparse
import glob
import inspect
import json
import os
import re
import sys

import meeteval
import meeteval.wer.wer.cp as _cp
from meeteval.io import STM, SegLST

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import bench  # noqa: E402
from src.data.text_norm import get_text_norm  # noqa: E402

norm = get_text_norm("whisper_nsf")

# meeteval refuses cpWER/tcpWER above 20 speakers as a sanity check; the assignment itself is Hungarian and fine at
# 40. Some APIs over-cluster long meetings into 20-40 labels, which must be scored, not dropped.
exec(inspect.getsource(_cp._minimum_permutation_word_error_rate).replace(") > 20:", ") > 1000:"), _cp.__dict__)


def base(session):
    """Strip the trailing partition tag ('-0', '-sdm', '-ch24'), on hypothesis and reference alike."""
    return re.sub(r"-[^-]+$", "", session)


def salvage(raw):
    """Complete segments from an output whose JSON never closed or whose parser gave up.

    VibeVoice: a repetition loop that runs to the token cap leaves the JSON unterminated, and the official parser
    then returns nothing for the whole session. MOSS: its parser drops the session after one malformed segment."""
    segs = []
    for m in re.finditer(r"\{[^{}]*\}", raw):
        try:
            o = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if "Start" in o and "Content" in o:
            segs.append({"start": o["Start"], "end": o.get("End", o["Start"]), "speaker": str(o.get("Speaker", "")),
                         "text": o["Content"]})
    for m in re.finditer(r"\[(\d+(?:\.\d+)?)\]\[(S\d+)\](.*?)\[(\d+(?:\.\d+)?)\]", raw):
        segs.append({"start": float(m.group(1)), "end": float(m.group(4)), "speaker": m.group(2), "text": m.group(3)})
    return segs


def score(system, c, partial=False):
    files = sorted(glob.glob(f"{system}/{c}/*.json"))
    if not files:
        return None
    segs, errors, salvaged, done = [], [], [], set()
    for f in files:
        d = json.load(open(f))
        done.add(base(d["session"]))
        if d.get("error"):
            errors.append(d["session"])
        if not d["segments"] and (d.get("raw") or "").strip():
            d["segments"] = salvage(d["raw"])
            salvaged.append(d["session"])
        for s in d["segments"]:
            words = norm(s["text"]).strip() if s["speaker"] else ""
            if words:
                st, en = float(s["start"]), float(s["end"])
                segs.append({"session_id": base(d["session"]), "speaker": s["speaker"] or "?", "words": words,
                             "start_time": st, "end_time": max(en, st)})
    ref = SegLST([{**s, "session_id": base(s["session_id"]), "start_time": float(s["start_time"]),
                   "end_time": float(s["end_time"])} for s in STM.load(bench.ref_path(c)).to_seglst()])
    ref_sessions = {s["session_id"] for s in ref}
    if partial:
        ref = ref.filter(lambda s: s["session_id"] in done)
    er = lambda r: round(100 * meeteval.wer.combine_error_rates(r).error_rate, 2)
    missing = sorted(ref_sessions - done) if not partial else []
    # A session without hypothesis words (no output file, or a system that emitted nothing) is an empty hypothesis,
    # i.e. all deletions, not a skipped session: give it an empty placeholder segment.
    silent = {s["session_id"] for s in ref} - {s["session_id"] for s in segs}
    hyp_full = SegLST(segs + [{"session_id": s, "speaker": "?", "words": "", "start_time": 0.0, "end_time": 0.0}
                              for s in sorted(silent)])
    return {"tcpWER@5": er(meeteval.wer.tcpwer(ref, hyp_full, collar=5.0)), "cpWER": er(meeteval.wer.cpwer(ref, hyp_full)),
            "sessions": len(done & ref_sessions), "ref_sessions": len(ref_sessions), "missing": missing,
            "errors": errors, "salvaged": salvaged}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("system")
    ap.add_argument("corpora", nargs="*")
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    corpora = bench.expand(a.corpora) if a.corpora else [c for g in bench.GROUPS.values() for c in g]
    path = f"{a.system}/scores.json"
    res = json.load(open(path)) if os.path.exists(path) else {}
    for c in corpora:
        if not os.path.exists(bench.ref_path(c)):
            continue
        r = score(a.system, c, a.partial)
        if r is None:
            continue
        res[c] = r
        print(f"RESULT {c:16s} tcpWER@5 {r['tcpWER@5']:6.2f} cpWER {r['cpWER']:6.2f} sessions "
              f"{r['sessions']}/{r['ref_sessions']} errors {len(r['errors'])} salvaged {len(r['salvaged'])}"
              f"{' MISSING ' + str(len(r['missing'])) if r['missing'] else ''}", flush=True)
    json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
