"""Nemotron-3-Diarization -> Pandemonium, conditioning on the diarizer's soft outputs."""

import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch

from pandemonium import FRAME_SHIFT, SAMPLE_RATE, Pandemonium, Word

DIARIZATION_MODEL = os.environ.get("DIARIZATION_MODEL", "nvidia/Nemotron-3-Diarization")
# A diarizer channel counts as a speaker once its probability crosses this anywhere.
SPEAKER_THRESHOLD = float(os.environ.get("SPEAKER_THRESHOLD", "0.5"))
COMPUTE_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
    os.environ.get("PANDEMONIUM_DTYPE", "bf16")
]
# Speaker rows per Conformer-layer call; 0 = sized automatically to the audio length.
ROW_CHUNK = int(os.environ.get("PANDEMONIUM_ROW_CHUNK", "0")) or None
# Keep both models in host RAM and move each to the GPU only while it runs, so an idle demo holds no
# GPU memory on a card it shares with other services.
OFFLOAD = os.environ.get("PANDEMONIUM_OFFLOAD", "1").lower() in ("1", "true", "yes")


@dataclass
class Diarization:
    """Diarizer output restricted to the channels that actually hold a speaker."""

    probs: np.ndarray                    # (G, T) speaker probabilities
    frame_shift: float                   # seconds per column of `probs`
    segments: List[Tuple[float, float, int]] = field(default_factory=list)  # (start, end, speaker row)

    @property
    def num_speakers(self) -> int:
        return self.probs.shape[0]

    def soft_activity(self, num_frames: int) -> np.ndarray:
        """Mean probability inside each 80 ms encoder frame -> `(G, num_frames)`."""
        times = (np.arange(self.probs.shape[1]) + 0.5) * self.frame_shift
        frame = np.minimum((times / FRAME_SHIFT).astype(int), num_frames - 1)
        total = np.zeros((self.num_speakers, num_frames))
        count = np.bincount(frame, minlength=num_frames).astype(float)
        for g in range(self.num_speakers):
            total[g] = np.bincount(frame, weights=self.probs[g], minlength=num_frames)
        return (total / np.maximum(count, 1)).astype(np.float32)

    def hard_activity(self, num_frames: int) -> np.ndarray:
        """The diarizer's post-processed segments rasterized exactly as an RTTM is in training."""
        mask = np.zeros((self.num_speakers, num_frames), dtype=np.float32)
        for start, end, spk in self.segments:
            mask[spk, int(start / FRAME_SHIFT): int(end / FRAME_SHIFT)] = 1.0
        return mask


class DemoPipeline:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.offload = OFFLOAD and self.device != "cpu"
        home = "cpu" if self.offload else self.device  # where the models wait between requests
        # One GPU job at a time, whoever asks (UI events, API calls): two would each bring a model in.
        self._gpu = threading.Lock()
        from nemo.collections.asr.models import SortformerEncLabelModel

        self.diarizer = SortformerEncLabelModel.from_pretrained(DIARIZATION_MODEL, map_location=home).eval()
        # The model card's "offline" setting (30.4 s buffer), its most accurate configuration. In
        # 80 ms frames; the checkpoint loads with a different, lower-latency default.
        sm = self.diarizer.sortformer_modules
        sm.chunk_len, sm.chunk_right_context, sm.fifo_len, sm.spkcache_update_period = 340, 40, 40, 300
        sm.spkcache_len = 264
        self.diarizer._check_streaming_parameters()
        self.asr = Pandemonium.from_pretrained(device=home)

    def warm_up(self, seconds: float = 8.0) -> None:
        """One throwaway run of both models, before the first visitor's.

        A new process pays a one-off cost on its first GPU run (CUDA libraries and kernels are loaded on
        first use): 2-37 s measured here, depending on whether those files are still in the OS cache. A
        warm run of the same recording then takes ~1 s. Low noise with two "speakers" covers every path.
        """
        audio = (np.random.default_rng(0).standard_normal(int(seconds * SAMPLE_RATE)) * 1e-3).astype(np.float32)
        self.diarize(audio)
        n = Pandemonium.num_frames(len(audio))
        activity = np.zeros((2, n), np.float32)
        activity[0, : n // 2] = activity[1, n // 2:] = 0.9
        with self._on_device(self.asr):
            self.asr.transcribe(audio, activity, dtype=COMPUTE_DTYPE, row_chunk=ROW_CHUNK)

    @contextmanager
    def _on_device(self, model):
        """Hold the GPU for one job, with `model` on it (and back in host RAM afterwards)."""
        with self._gpu:
            if self.offload:
                model.to(self.device)
            try:
                yield model
            finally:
                if self.offload:
                    model.to("cpu")
                if self.device == "cuda":
                    torch.cuda.empty_cache()

    @torch.inference_mode()
    def diarize(self, audio: np.ndarray) -> Diarization:
        with self._on_device(self.diarizer):
            segments, probs = self.diarizer.diarize(
                audio=[audio], sample_rate=SAMPLE_RATE, batch_size=1, include_tensor_outputs=True, verbose=False
            )
        probs = probs[0]
        probs = (probs[0] if probs.dim() == 3 else probs).float().cpu().numpy().T  # (8, T)
        frame_shift = len(audio) / SAMPLE_RATE / probs.shape[1]

        # Channels are in arrival order already; keep the ones that ever speak.
        keep = [g for g in range(probs.shape[0]) if probs[g].max() >= SPEAKER_THRESHOLD]
        row = {g: i for i, g in enumerate(keep)}
        segs = []
        for seg in segments[0]:
            start, end, spk = _parse_segment(seg)
            if spk in row:
                segs.append((start, end, row[spk]))
        return Diarization(probs=probs[keep], frame_shift=frame_shift, segments=sorted(segs))

    def transcribe(self, audio: np.ndarray, diar: Diarization, soft: bool = True) -> List[List[Word]]:
        if diar.num_speakers == 0:
            return []
        n = Pandemonium.num_frames(len(audio))
        activity = diar.soft_activity(n) if soft else diar.hard_activity(n)
        with self._on_device(self.asr):
            try:
                return self.asr.transcribe(audio, activity, dtype=COMPUTE_DTYPE, row_chunk=ROW_CHUNK)
            except torch.OutOfMemoryError:
                # The card is shared (another service holds a few GB of it), so a long recording can come
                # up just short: once more with one speaker row per layer call, the smallest footprint.
                torch.cuda.empty_cache()
                return self.asr.transcribe(audio, activity, dtype=COMPUTE_DTYPE, row_chunk=1)


def _parse_segment(seg) -> Tuple[float, float, int]:
    """NeMo returns 'start end speaker_k' strings (or tuples, depending on the version)."""
    if isinstance(seg, str):
        start, end, spk = seg.split()
    else:
        start, end, spk = seg[:3]
    spk = int(str(spk).rsplit("_", 1)[-1])
    return float(start), float(end), spk


def utterances(words_per_speaker: List[List[Word]], max_gap: float = 1.0) -> List[Dict]:
    """Group each speaker's words into utterances at pauses, then interleave them by start time."""
    out = []
    for spk, words in enumerate(words_per_speaker):
        cur = None
        for w in words:
            if cur is None or w.start - cur["end"] > max_gap:
                cur = {"speaker": spk, "start": w.start, "end": w.end, "words": [], "word_starts": []}
                out.append(cur)
            cur["words"].append(w.text)
            cur["word_starts"].append(w.start)  # lets the player follow along word by word
            cur["end"] = w.end
    for u in out:
        u["text"] = " ".join(u.pop("words"))
    return sorted(out, key=lambda u: (u["start"], u["speaker"]))
