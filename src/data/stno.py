"""Silence / Target / Non-target / Overlap (STNO) mask construction.

This is the single source of truth for how diarization information is turned into the
4-channel frame-level mask that conditions the FDDT blocks inside the encoder. Both the
training dataset (`src/data/dataset.py`, from manifest segments) and inference
(`infer.py`, from an RTTM) build their masks here so the two paths cannot drift apart.

The mask lives at the *encoder* frame rate, i.e. after the FastConformer subsampling:

    frame_rate = sample_rate / audio_downsampling_factor
               = 16000 / (16000 * 0.01 * 8) = 16000 / 1280 = 12.5 Hz
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

__all__ = [
    'SpeechSegment',
    'create_stno_masks',
    'frame_rate_from_downsampling_factor',
    'num_encoder_frames',
    'soft_stno_masks',
    'speaker_activity_mask',
    'stno_mask_for_speaker',
]


@dataclass(frozen=True)
class SpeechSegment:
    """A single diarization segment: who speaks when, in seconds."""

    start: float
    duration: float
    speaker: str

    @property
    def end(self) -> float:
        return self.start + self.duration


def frame_rate_from_downsampling_factor(sample_rate: int, audio_downsampling_factor: int) -> float:
    """Encoder frame rate in Hz (12.5 for 16 kHz audio and 8x FastConformer subsampling)."""
    return sample_rate / audio_downsampling_factor


def num_encoder_frames(num_samples: int, audio_downsampling_factor: int) -> int:
    """Number of encoder frames a waveform of `num_samples` samples subsamples to.

    Rounds up, matching the dataset's original computation. The encoder reconciles any
    remaining off-by-one against the actual subsampled length, see `ConformerEncoderSTNO`.
    """
    remainder = num_samples % audio_downsampling_factor
    if remainder != 0:
        num_samples = num_samples + (audio_downsampling_factor - remainder)
    return int(num_samples / audio_downsampling_factor)


def speaker_activity_mask(
    segments: Iterable[SpeechSegment],
    speakers: Sequence[str],
    num_frames: int,
    frame_rate: float,
) -> torch.Tensor:
    """Rasterize diarization segments into a binary (num_speakers, num_frames) mask.

    Segments whose speaker is not in `speakers` are ignored.
    """
    speaker_to_idx = {speaker: i for i, speaker in enumerate(speakers)}
    mask = torch.zeros((len(speakers), num_frames))

    for segment in segments:
        idx = speaker_to_idx.get(segment.speaker)
        if idx is None:
            continue
        begin = int(segment.start * frame_rate)
        end = int((segment.start + segment.duration) * frame_rate)
        mask[idx, begin:end] = 1

    return mask


def create_stno_masks(spk_mask: torch.Tensor, s_index: int) -> torch.Tensor:
    """Turn a (num_speakers, T) activity mask into the (4, T) STNO mask for speaker `s_index`.

    The four channels are mutually exclusive and sum to 1 at every frame:
      0 silence     - nobody speaks
      1 target      - the target speaker speaks alone
      2 non-target  - somebody else speaks but the target does not
      3 overlap     - the target speaks together with somebody else

    Only the target row is distinguished from the rest, so the result does not depend on
    how the remaining speakers are ordered.
    """
    non_target_mask = torch.ones(spk_mask.shape[0], dtype=torch.bool)
    non_target_mask[s_index] = False
    sil_frames = (1 - spk_mask).prod(dim=0)
    anyone_else = (1 - spk_mask[non_target_mask]).prod(dim=0)
    target_spk = spk_mask[s_index] * anyone_else
    non_target_spk = (1 - spk_mask[s_index]) * (1 - anyone_else)
    overlapping_speech = spk_mask[s_index] - target_spk
    return torch.stack([sil_frames, target_spk, non_target_spk, overlapping_speech], dim=0)


def soft_stno_masks(activity: torch.Tensor, speaker_valid: torch.Tensor) -> torch.Tensor:
    """Differentiable twin of `create_stno_masks`, run for every speaker of a session at once.

    The hard `create_stno_masks` takes a binary activity mask and one target index; this takes a
    *soft* per-speaker activity `a_i(t) in [0, 1]` and produces the 4-channel STNO mask for every
    speaker in one shot, recombining each stream's channels from the *other* speakers' activities.
    That cross-speaker coupling is exactly what the cross-speaker attention makes available, and is why the
    encoder can re-estimate the mask internally (see `src/model/modules/cross_speaker_attention.py`).

    With `activity` hard 0/1 and every speaker valid, the per-speaker output equals
    `create_stno_masks(activity[s], i)` frame for frame -- the two are one formula, one soft and
    one hard. The channels are non-negative and sum to 1 at every frame:

      0 silence     (1 - a_i) * E_i
      1 target       a_i      * E_i
      2 non-target  (1 - a_i) * (1 - E_i)
      3 overlap      a_i      * (1 - E_i)

    where `E_i(t) = prod_{j != i, j valid} (1 - a_j(t))` is the probability every *other* speaker
    is silent. Invalid (padding) speakers contribute a factor 1 and get an all-zero mask.

    Args:
        activity: `(S, G, T)` soft speaker activity in [0, 1].
        speaker_valid: `(S, G)` boolean; False marks a padding speaker slot with no real speaker.

    Returns:
        `(S, G, 4, T)` STNO mask, one 4-channel group per speaker slot.
    """
    valid = speaker_valid[..., None].to(activity.dtype)  # (S, G, 1)
    a = activity * valid                                  # padding slots contribute nothing
    one_minus = 1.0 - a
    # A padding speaker must be a factor of 1 in every other speaker's product, so force it to 1
    # regardless of its (ignored) activity.
    one_minus = torch.where(valid.bool(), one_minus, torch.ones_like(one_minus))

    # Leave-one-out product over the speaker axis via prefix/suffix products -- stable (no
    # division) and differentiable. Built by accumulation (no in-place slice writes, which would
    # break autograd); G is small (<= num_speaker_slots), so this is cheap.
    g = a.shape[1]
    ones = torch.ones_like(a[:, 0])
    prefix_terms, acc = [], ones
    for i in range(g):
        prefix_terms.append(acc)
        acc = acc * one_minus[:, i]
    suffix_terms, acc = [None] * g, ones
    for i in range(g - 1, -1, -1):
        suffix_terms[i] = acc
        acc = acc * one_minus[:, i]
    others_silent = torch.stack(prefix_terms, dim=1) * torch.stack(suffix_terms, dim=1)  # (S, G, T)

    silence = (1.0 - a) * others_silent
    target = a * others_silent
    non_target = (1.0 - a) * (1.0 - others_silent)
    overlap = a * (1.0 - others_silent)
    mask = torch.stack([silence, target, non_target, overlap], dim=2)  # (S, G, 4, T)
    return mask * valid[..., None]


def stno_mask_for_speaker(
    segments: Iterable[SpeechSegment],
    speakers: Sequence[str],
    target_speaker: str,
    num_frames: int,
    frame_rate: float,
) -> torch.Tensor:
    """Convenience wrapper: diarization segments -> (4, num_frames) mask for one target speaker."""
    if target_speaker not in speakers:
        raise ValueError(f"Target speaker {target_speaker!r} is not in {list(speakers)!r}")

    spk_mask = speaker_activity_mask(segments, speakers, num_frames, frame_rate)
    return create_stno_masks(spk_mask, speakers.index(target_speaker))
