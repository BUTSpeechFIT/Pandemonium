"""Precomputed examples (see `example/README.md`): a finished result to show without running a model."""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from pandemonium import SAMPLE_RATE, Word
from pipeline import Diarization, utterances

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example")


@dataclass
class Example:
    name: str
    meta: Dict
    audio_path: str                      # the MP3 the browser plays
    audio: np.ndarray                    # decoded at SAMPLE_RATE, for the timeline's waveform lane
    diar: Diarization
    words: List[List[Word]]              # per speaker row
    utts: List[Dict]                     # {speaker, start, end, text}, as `pipeline.utterances` makes them
    downloads: Dict[str, str] = field(default_factory=dict)  # label -> path
    reference: Optional[List[Dict]] = None  # turns in `utts` form, speakers mapped onto the rows
    scores: Optional[Dict] = None


def load_example(name: str, decode=None) -> Example:
    """Load `example/<name>/`. `decode(path) -> float32 mono at SAMPLE_RATE` reads the audio."""
    d = os.path.join(EXAMPLES_DIR, name)

    def read(file):
        path = os.path.join(d, file)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    meta = read("meta.json")
    audio_path = os.path.join(d, meta.get("audio", "audio.mp3"))
    audio = decode(audio_path) if decode else _decode(audio_path)

    npz = np.load(os.path.join(d, "diarization.npz"))
    segments = [(float(s), float(e), int(k)) for s, e, k in npz["segments"]]
    diar = Diarization(probs=npz["probs"].astype(np.float32), frame_shift=float(npz["frame_shift"]),
                       segments=sorted(segments))
    diar.source = audio_path

    words = [[Word(t, float(s), float(e)) for t, s, e in row] for row in read("words.json")]
    downloads = {label: os.path.join(d, file)
                 for label, file in (("utterances", "transcript.seglst.json"),
                                     ("words", "transcript.words.seglst.json"),
                                     ("reference", "reference.seglst.json"))
                 if os.path.exists(os.path.join(d, file))}

    scores = read("scores.json")
    reference = None
    ref = read("reference.seglst.json")
    if ref is not None and scores and scores.get("speaker_mapping"):
        # "FIE088" -> "Speaker 1" -> row 0, so the reference reads in the same colours as the output.
        row = {spk: int(name.rsplit(" ", 1)[-1]) - 1 for spk, name in scores["speaker_mapping"].items()}
        reference = sorted(({"speaker": row[s["speaker"]], "start": s["start_time"], "end": s["end_time"],
                             "text": s["words"]} for s in ref if s["speaker"] in row),
                           key=lambda u: (u["start"], u["speaker"]))

    # The stored turns, with each word's start time added (regrouping the stored words the same way the
    # pipeline does gives exactly these turns; if it ever does not, they are used as stored).
    utts = read("utterances.json")
    regrouped = utterances(words)
    if [(u["speaker"], u["text"]) for u in regrouped] == [(u["speaker"], u["text"]) for u in utts]:
        utts = regrouped
    return Example(name=name, meta=meta, audio_path=audio_path, audio=audio, diar=diar, words=words,
                   utts=utts, downloads=downloads, reference=reference, scores=scores)


def _decode(path):
    from librosa import load

    return load(path, sr=SAMPLE_RATE, mono=True)[0].astype(np.float32)
