"""Inference-only copy of Pandemonium's encoder building blocks.

Vendored from the PolyKeet training repo (`src/model/modules/{fddt,sink_attention,cross_speaker_attention,
joint_ctc_decoder}.py`, `src/data/stno.py`). Parameter names and layouts are unchanged, so the
released checkpoint loads with `strict=True`; everything that only matters for training (gate
warm-up, logging, lr multipliers) is gone.
"""

import torch
import torch.nn as nn

from nemo.collections.asr.parts.submodules.multi_head_attention import (
    RelPositionMultiHeadAttentionLongformer,
)


# ---------------------------------------------------------------------------------------------
# STNO masks
# ---------------------------------------------------------------------------------------------

def soft_stno_masks(activity: torch.Tensor, speaker_valid: torch.Tensor) -> torch.Tensor:
    """Per-speaker silence / target / non-target / overlap masks from soft speaker activity.

    With `E_i = prod_{j != i} (1 - a_j)` the probability every *other* speaker is silent:

      0 silence     (1 - a_i) * E_i
      1 target       a_i      * E_i
      2 non-target  (1 - a_i) * (1 - E_i)
      3 overlap      a_i      * (1 - E_i)

    With hard 0/1 activity this is exactly the training-time (RTTM) mask, so the same function
    serves the soft and the hard path of the demo.

    Args:
        activity: `(S, G, T)` speaker activity in [0, 1].
        speaker_valid: `(S, G)` boolean; False marks a padding speaker slot.

    Returns:
        `(S, G, 4, T)` masks; all-zero for padding slots.
    """
    valid = speaker_valid[..., None].to(activity.dtype)
    a = activity * valid
    one_minus = torch.where(valid.bool(), 1.0 - a, torch.ones_like(a))

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
    others_silent = torch.stack(prefix_terms, dim=1) * torch.stack(suffix_terms, dim=1)

    mask = torch.stack(
        [(1.0 - a) * others_silent, a * others_silent, (1.0 - a) * (1.0 - others_silent), a * (1.0 - others_silent)],
        dim=2,
    )
    return mask * valid[..., None]


# ---------------------------------------------------------------------------------------------
# FDDT: frame-level diarization-dependent transforms
# ---------------------------------------------------------------------------------------------

class CustomDiagonalLinear(nn.Module):
    def __init__(self, d_model, bias=True, init_eye_val=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.full((d_model,), init_eye_val))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None

    def forward(self, x):
        out = x * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class FDDT(nn.Module):
    """`sum_c W_c(h) * stno[c]` with one diagonal affine transform per STNO class."""

    def __init__(self, d_model, non_target_rate=0.5):
        super().__init__()
        self.target_linear = CustomDiagonalLinear(d_model, init_eye_val=1.0)
        self.non_target_linear = CustomDiagonalLinear(d_model, init_eye_val=non_target_rate)
        self.overlap_linear = CustomDiagonalLinear(d_model, init_eye_val=1.0)
        self.silence_linear = CustomDiagonalLinear(d_model, init_eye_val=non_target_rate)

    def forward(self, h, stno_mask):
        m = stno_mask.to(h.dtype)[..., None]  # (B, 4, T, 1)
        return (
            self.silence_linear(h) * m[:, 0]
            + self.target_linear(h) * m[:, 1]
            + self.non_target_linear(h) * m[:, 2]
            + self.overlap_linear(h) * m[:, 3]
        )


# ---------------------------------------------------------------------------------------------
# Attention-sink gate
# ---------------------------------------------------------------------------------------------

class SinkGatedLongformerAttention(RelPositionMultiHeadAttentionLongformer):
    """Longformer attention whose sink (global-token) logits carry a learned per-head offset."""

    def _compute_global_key_attn(self, *args, **kwargs) -> torch.Tensor:
        scores = super()._compute_global_key_attn(*args, **kwargs)
        return scores + self.sink_gate.view(1, 1, -1, 1)


def enable_sink_gate(encoder: nn.Module) -> None:
    """Re-class every Longformer attention in `encoder` to carry a `sink_gate` parameter."""
    for layer in encoder.layers:
        attention = getattr(layer, 'self_attn', None)
        if isinstance(attention, RelPositionMultiHeadAttentionLongformer) and getattr(attention, 'global_tokens', 0) > 0:
            attention.__class__ = SinkGatedLongformerAttention
            attention.register_parameter('sink_gate', nn.Parameter(torch.zeros(attention.h)))


# ---------------------------------------------------------------------------------------------
# Cross-speaker attention and STNO-mask refinement
# ---------------------------------------------------------------------------------------------

class CrossSpeakerAttention(nn.Module):
    """Per-frame attention across the speakers of a session (over the speaker axis, not time)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, num_speakers: int, frame_chunk: int = 4096):
        """`x` is `(G, T, D)`, one row per speaker of a single session."""
        g, t, d = x.shape
        normed = self.norm(x).permute(1, 0, 2)  # (T, G, D): the frame is the batch
        # Chunked over frames: (T, G, G) per head is tiny, but MultiheadAttention's intermediate
        # projections over a full hour are not.
        out = torch.cat(
            [self.attn(c, c, c, need_weights=False)[0] for c in normed.split(frame_chunk, dim=0)], dim=0
        )
        return x + torch.tanh(self.gate) * out.permute(1, 0, 2)


class MaskRefiner(nn.Module):
    """Predict per-speaker activity from the hidden states and blend the resulting STNO mask in."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.activity = nn.Linear(d_model, 1)
        self.blend_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, stno_mask):
        """`x` is `(G, T, D)`, `stno_mask` is `(G, 4, T)`."""
        g, t, _ = x.shape
        activity = torch.sigmoid(self.activity(self.norm(x)).squeeze(-1)).float()  # (G, T)
        valid = torch.ones(1, g, dtype=torch.bool, device=x.device)
        soft = soft_stno_masks(activity[None], valid)[0].to(stno_mask.dtype)  # (G, 4, T)
        blend = torch.sigmoid(self.blend_gate).to(stno_mask.dtype)
        return (1.0 - blend) * stno_mask + blend * soft


# ---------------------------------------------------------------------------------------------
# CTC head built from Parakeet's TDT joint
# ---------------------------------------------------------------------------------------------

class JointCTCDecoder(nn.Module):
    """`joint_net(enc(f) + pred(0))` with the TDT duration outputs sliced off; blank is last."""

    def __init__(self, feat_in, num_classes, pred_hidden=640, joint_hidden=640, dropout=0.2,
                 num_extra_outputs=5, **_):
        super().__init__()
        self.num_classes_with_blank = num_classes + 1
        self.pred = nn.Linear(pred_hidden, joint_hidden)
        self.enc = nn.Linear(feat_in, joint_hidden)
        # Same layer order as the checkpoint (ReLU, Dropout, Linear), so the keys line up.
        self.joint_net = nn.Sequential(
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(joint_hidden, self.num_classes_with_blank + num_extra_outputs)
        )
        self.register_buffer('sos_embedding', torch.zeros(1, 1, pred_hidden), persistent=False)

    def forward(self, encoder_output):
        """`(B, D, T)` -> `(B, T, V + 1)` log-probabilities."""
        f = self.enc(encoder_output.transpose(1, 2))
        g = self.pred(self.sos_embedding.to(f.dtype))
        logits = self.joint_net(f + g)[..., : self.num_classes_with_blank]
        return torch.log_softmax(logits.float(), dim=-1)
