"""Cross-speaker attention and internal STNO-mask refinement for Pandemonium.

DiCoP encodes each (session, speaker) stream independently. This model batches a session's
speakers together and lets them talk to one another inside the encoder, then lets the model
re-estimate the STNO mask from what that communication reveals. Two modules, inserted into a
handful of Conformer layers by `src/model/modules/cross_speaker_encoder.py`:

  - `CrossSpeakerAttention` -- at each time frame, the speakers of one session attend across
    each other. Cheap: the attention runs over the length-`G` speaker axis (G <= a few), not over
    time. A gated residual (`tanh(gate)`, gate init 0) makes it inert at initialization, so a
    warm-started encoder starts bit-identical to DiCoP.

  - `MaskRefiner` -- reads each stream's (now cross-speaker-informed) hidden state, predicts a
    soft per-speaker activity, and recombines it across the session's speakers into a fresh
    4-channel STNO mask (`soft_stno_masks`). The result is blended with the incoming mask through
    a sigmoid gate whose logit starts very negative, so at initialization the mask is unchanged.
    There is no auxiliary supervision -- the downstream CTC loss is the only signal, and the gate
    lets refinement fade in from the baseline.

Both take the encoder's hidden states already flattened to `(B, T, D)` with `B = S * G`, plus the
grouping `(S, G)` and a per-session speaker-validity mask, so that padding speaker slots (sessions
with fewer than `G` speakers) neither send nor receive information.
"""

import torch
import torch.nn as nn

from src.data.stno import soft_stno_masks

__all__ = ['CrossSpeakerAttention', 'MaskRefiner']


def _key_padding_mask(speaker_valid: torch.Tensor, num_frames: int) -> torch.Tensor:
    """`(S*T, G)` boolean key-padding mask (True = ignore) from `(S, G)` speaker validity.

    Speaker validity is per session, not per frame, so it is broadcast across the `T` frames that
    become the batch axis of the per-frame speaker attention.
    """
    s, g = speaker_valid.shape
    pad = ~speaker_valid  # True where the slot is padding
    return pad[:, None, :].expand(s, num_frames, g).reshape(s * num_frames, g)


class CrossSpeakerAttention(nn.Module):
    """Per-frame multi-head attention across the speakers of one session.

    Args:
        d_model: Hidden width.
        n_heads: Attention heads.
        dropout: Attention dropout.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        # tanh(gate) scales the residual; 0 -> inert, so a warm start is unchanged.
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, num_sessions: int, num_speakers_padded: int,
                speaker_valid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: `(S*G, T, D)` hidden states, speaker-major within each session.
            num_sessions: `S`.
            num_speakers_padded: `G` (the padded speaker count, `S*G == x.shape[0]`).
            speaker_valid: `(S, G)` boolean; False marks a padding speaker slot.

        Returns:
            `(S*G, T, D)`, the input with a gated cross-speaker update added.
        """
        s, g = num_sessions, num_speakers_padded
        t, d = x.shape[1], x.shape[2]

        normed = self.norm(x)
        # (S, G, T, D) -> (S, T, G, D) -> (S*T, G, D): G is the attention sequence, one token per
        # speaker, and (session, frame) is the batch.
        normed = normed.view(s, g, t, d).permute(0, 2, 1, 3).reshape(s * t, g, d)
        key_padding_mask = _key_padding_mask(speaker_valid, t).to(normed.device)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask,
                                need_weights=False)
        # Back to (S*G, T, D).
        attn_out = attn_out.view(s, t, g, d).permute(0, 2, 1, 3).reshape(s * g, t, d)
        return x + torch.tanh(self.gate) * attn_out


class MaskRefiner(nn.Module):
    """Re-estimate the STNO mask from cross-speaker-informed hidden states.

    Predicts a soft per-speaker activity, recombines it across the session's speakers with
    `soft_stno_masks`, and blends the result with the incoming mask through a sigmoid gate whose
    logit starts very negative (so the mask is unchanged at initialization).

    The blend is `sigmoid(gate)`, and at `gate_init = -8.0` it starts at ~3.3e-4 -- the same
    near-zero the encoder's attention-sink gate starts at, and warm-start-safe for the same reason.
    It is a convex blend of two masks that each sum to 1 per frame, so the refined mask does too and
    stays non-negative. Like the sink gate it needs an above-base learning rate to open
    (`mask_refiner_lr_multiplier`), since the sigmoid gradient is small out here.

    Args:
        d_model: Hidden width.
        gate_init: Initial blend-gate logit; large-negative keeps the incoming mask.
    """

    def __init__(self, d_model: int, gate_init: float = -8.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.activity = nn.Linear(d_model, 1)
        # Start the predicted activity near the *input* mask by keeping the head near zero: a zero
        # logit is activity 0.5, but the gate below is what actually holds refinement off at init.
        nn.init.zeros_(self.activity.weight)
        nn.init.zeros_(self.activity.bias)
        self.blend_gate = nn.Parameter(torch.full((1,), float(gate_init)))

    def forward(self, x: torch.Tensor, stno_mask: torch.Tensor, num_sessions: int,
                num_speakers_padded: int, speaker_valid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: `(S*G, T, D)` hidden states.
            stno_mask: `(S*G, 4, T)` incoming STNO mask.
            num_sessions: `S`.
            num_speakers_padded: `G`.
            speaker_valid: `(S, G)` boolean speaker validity.

        Returns:
            `(S*G, 4, T)` refined STNO mask.
        """
        s, g = num_sessions, num_speakers_padded
        t = x.shape[1]

        activity = torch.sigmoid(self.activity(self.norm(x)).squeeze(-1))  # (S*G, T)
        activity = activity.view(s, g, t)
        soft = soft_stno_masks(activity, speaker_valid.to(activity.device))  # (S, G, 4, T)
        soft = soft.reshape(s * g, 4, t).to(stno_mask.dtype)

        blend = torch.sigmoid(self.blend_gate)
        return (1.0 - blend) * stno_mask + blend * soft
