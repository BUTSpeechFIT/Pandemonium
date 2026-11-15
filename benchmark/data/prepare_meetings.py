"""Turn a meeting eval set (a Lhotse cutset of whole recordings) into the benchmark layout.

    python benchmark/data/prepare_meetings.py notsofar /path/to/notsofar1-sdm_eval.tar
    python benchmark/data/prepare_meetings.py ami /path/to/ami-sdm_test.jsonl.gz

The input is either a Lhotse Shar set (a .tar of `cuts.NNNNNN.jsonl.gz` + `recording.NNNNNN.tar`, or that directory
extracted) or an ordinary cutset whose recordings point at audio files. One cut per recording, `cut.start == 0`,
supervisions with speaker and text. See benchmark/README.md for which recordings, channel and reference each set uses.

Writes, under $BENCH_DATA/<corpus>/:
  audio/            one wav per recording, at its native sampling rate (SBC is 22.05 kHz; runners resample)
  cutset.jsonl.gz   the same cuts, file-backed
  manifest.json     {session: wav}, session = cut id
  ref.stm           whisper_nsf-normalized reference on the recording timeline; a speaker whose every segment
                    normalizes to empty keeps an empty line, so its words count as deletions rather than vanish
"""
import argparse
import json
import os
import sys
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))  # repo root: src.data.text_norm, utils.stm

import bench  # noqa: E402
from lhotse import CutSet  # noqa: E402
from meeteval.io.seglst import SegLstSegment  # noqa: E402

from src.data.lhotse_utils import cut_session_id  # noqa: E402
from src.data.text_norm import get_text_norm  # noqa: E402
from utils.stm import write_stm  # noqa: E402


def load_cuts(src, scratch):
    p = Path(src)
    if p.suffix == ".tar":
        with tarfile.open(p) as tf:
            tf.extractall(scratch)
        p = next(d for d in [Path(scratch), *sorted(Path(scratch).rglob("*"))] if d.is_dir() and any(d.glob("cuts.*.jsonl.gz")))
    if p.is_dir():
        return CutSet.from_shar(fields={"cuts": sorted(map(str, p.glob("cuts.*.jsonl.gz"))),
                                        "recording": sorted(map(str, p.glob("recording.*.tar")))})
    return CutSet.from_file(p)


def write_ref(cuts, out):
    """The reference every system is scored against (identical to the paper's evaluation)."""
    norm = get_text_norm("whisper_nsf")
    segments, speakers = defaultdict(list), defaultdict(list)
    for cut in cuts:
        session = cut_session_id(cut)
        for sup in cut.supervisions:
            spk = str(sup.speaker)
            if spk not in speakers[session]:
                speakers[session].append(spk)
            words = norm(sup.text or "")
            if words.strip():
                start = cut.start + sup.start
                segments[session].append(SegLstSegment(session_id=session, speaker=spk, words=words,
                                                       start_time=start, end_time=start + sup.duration))
    sessions = sorted(speakers)
    ref = {s: sorted(segments[s], key=lambda x: (x["start_time"], x["speaker"])) for s in sessions}
    return write_stm(Path(out), ref, {s: sorted(speakers[s]) for s in sessions})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", help="benchmark name: " + ", ".join(bench.MEETINGS))
    ap.add_argument("source", help="Shar .tar / directory, or a cutset .jsonl.gz")
    a = ap.parse_args()

    d = Path(bench.corpus_dir(a.corpus))
    (d / "audio").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="shar_") as scratch:
        # one job: more fork workers that re-import this module, which breaks when it is not run as a file
        cuts = load_cuts(a.source, scratch).save_audios(str(d / "audio"), num_jobs=1).to_eager()
    cuts.to_file(bench.cutset_path(a.corpus))
    manifest = {cut.id: os.path.abspath(cut.recording.sources[0].source) for cut in cuts}
    json.dump(manifest, open(bench.manifest_path(a.corpus), "w"))
    n = write_ref(cuts, bench.ref_path(a.corpus))
    hours = sum(c.duration for c in cuts) / 3600
    print(f"{a.corpus}: {len(cuts)} recordings, {hours:.2f} h, {n} reference lines -> {d}")


if __name__ == "__main__":
    main()
