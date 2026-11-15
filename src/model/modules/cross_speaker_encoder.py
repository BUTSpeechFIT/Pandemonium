"""FastConformer encoder with cross-speaker attention and internal STNO-mask refinement.

`ConformerEncoderCrossSpeaker` is a thin subclass of `ConformerEncoderSTNO`. It adds nothing to the
layer loop's copied body -- that stays in the base class -- and instead fills in the two hooks the
base declares as no-ops:

  - `_cross_speaker_attention`: after selected layers, run `CrossSpeakerAttention` so a session's
    speaker streams attend across one another.
  - `_refine_stno_mask`: after (a subset of) those layers, run `MaskRefiner` to re-estimate the
    STNO mask that conditions the deeper layers.

The batch reaching the encoder is a session's speakers stacked speaker-major: row `s*G + i` is
speaker `i` of session `s`, `G` the padded speaker count. The model calls `set_grouping` before
each forward to declare `S`, `G` and which speaker slots are real. At initialization the
cross-speaker attention is exactly inert (its residual is scaled by `tanh(0) = 0`) and the refiner is inert to
~3e-4 (its blend gate starts where the attention-sink gate does), so a warm-started encoder starts
all but identical to `ConformerEncoderSTNO`.
"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from src.model.modules.cross_speaker_attention import CrossSpeakerAttention, MaskRefiner
from src.model.modules.stno_encoder import ConformerEncoderSTNO

__all__ = ['ConformerEncoderCrossSpeaker']


class ConformerEncoderCrossSpeaker(ConformerEncoderSTNO):
    """`ConformerEncoderSTNO` with cross-speaker attention and STNO-mask refinement.

    Args:
        cross_speaker_layers: Zero-based layer indices after which cross-speaker attention runs.
        refine_layers: Layer indices after which the STNO mask is re-estimated. Must be a subset
            of `cross_speaker_layers` -- refinement reads the hidden states after it.
        cross_speaker_heads: Attention heads for the cross-speaker attention.
        cross_speaker_dropout: Attention dropout for the cross-speaker attention.
        refine_gate_init: Initial blend-gate logit of each `MaskRefiner`; large-negative keeps the
            incoming mask, so refinement fades in from the baseline.
    """

    def __init__(
        self,
        *args,
        cross_speaker_layers: Optional[Sequence[int]] = None,
        refine_layers: Optional[Sequence[int]] = None,
        cross_speaker_heads: int = 8,
        cross_speaker_dropout: float = 0.0,
        refine_gate_init: float = -8.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.cross_speaker_layers = sorted(set(int(i) for i in (cross_speaker_layers or [])))
        self.refine_layers = sorted(set(int(i) for i in (refine_layers or [])))
        extra = set(self.refine_layers) - set(self.cross_speaker_layers)
        if extra:
            raise ValueError(
                f"refine_layers {sorted(extra)} are not in cross_speaker_layers: refinement reads the "
                f"hidden states after cross-speaker attention, so every refine layer must also be one."
            )
        for layer in self.cross_speaker_layers + self.refine_layers:
            if not 0 <= layer < len(self.layers):
                raise ValueError(f"cross-speaker/refine layer {layer} out of range [0, {len(self.layers)})")

        # Keyed by str(layer index): ModuleDict keys must be strings.
        self.cross_speaker = nn.ModuleDict(
            {str(i): CrossSpeakerAttention(self.d_model, cross_speaker_heads, cross_speaker_dropout)
             for i in self.cross_speaker_layers}
        )
        self.mask_refiner = nn.ModuleDict(
            {str(i): MaskRefiner(self.d_model, gate_init=refine_gate_init) for i in self.refine_layers}
        )

        self._num_sessions: Optional[int] = None
        self._num_speakers_padded: Optional[int] = None
        self._speaker_valid: Optional[torch.Tensor] = None

    def set_grouping(self, num_sessions: int, num_speakers_padded: int, speaker_valid: torch.Tensor) -> None:
        """Declare the batch's `(S, G)` speaker grouping for the cross-speaker hooks.

        Args:
            num_sessions: `S`, the number of sessions in the batch.
            num_speakers_padded: `G`, the padded speaker count (`S * G == batch size`).
            speaker_valid: `(S, G)` boolean; False marks a padding speaker slot.
        """
        self._num_sessions = int(num_sessions)
        self._num_speakers_padded = int(num_speakers_padded)
        self._speaker_valid = speaker_valid

    def _sink_offset(self) -> int:
        """Number of prepended sink columns present in the sequence during the layer loop."""
        return self.global_tokens if self.learnable_sink_tokens else 0

    def _cross_speaker_attention(self, audio_signal, lth):
        key = str(lth)
        if key not in self.cross_speaker:
            return audio_signal
        return self.cross_speaker[key](
            audio_signal, self._num_sessions, self._num_speakers_padded, self._speaker_valid
        )

    def _refine_stno_mask(self, audio_signal, stno_mask, lth):
        key = str(lth)
        if key not in self.mask_refiner or stno_mask is None:
            return stno_mask
        module = self.mask_refiner[key]
        refined = module(
            audio_signal, stno_mask, self._num_sessions, self._num_speakers_padded, self._speaker_valid
        )
        # The prepended sink columns carry a fixed class that FDDT never reads; leave them exactly
        # as they were so nothing downstream sees a refined value where it expects the sink prefix.
        sinks = self._sink_offset()
        if sinks:
            refined = torch.cat([stno_mask[:, :, :sinks], refined[:, :, sinks:]], dim=-1)
        return refined

    def cross_speaker_gate_values(self) -> List[torch.Tensor]:
        """Current `tanh(gate)` of every cross-speaker block, for logging."""
        return [torch.tanh(module.gate).squeeze() for module in self.cross_speaker.values()]

    def refine_gate_values(self) -> List[torch.Tensor]:
        """Current `sigmoid(blend_gate)` of every refiner, for logging."""
        return [torch.sigmoid(module.blend_gate).squeeze() for module in self.mask_refiner.values()]
