"""A learnable gate on the attention sinks, so they can be added to a pretrained encoder.

Ungated sinks wreck a warm start: on NOTSOFAR they put epoch 0 at 0.833 cpWER against 0.588
without. The gate is an additive bias on the sink logits, added before the softmax, so a large
negative init makes the sinks invisible (exp(-20) ~ 2e-9) and training can open it.

Weight init cannot do this: with `use_bias: false`, zeroing `global_k` gives a score of 0, not
-inf, which is competitive in a softmax.
"""

from typing import Optional

import torch
import torch.nn as nn

from nemo.collections.asr.parts.submodules.multi_head_attention import (
    RelPositionMultiHeadAttentionLongformer,
)
from nemo.utils import logging

__all__ = ['SinkGatedLongformerAttention', 'enable_sink_gate', 'set_sink_gate']


def set_sink_gate(encoder: nn.Module, value: float) -> None:
    """Force every sink gate in `encoder` to `value`, for annealing it on a schedule."""
    for layer in encoder.layers:
        attention = getattr(layer, 'self_attn', None)
        if isinstance(attention, SinkGatedLongformerAttention):
            with torch.no_grad():
                attention.sink_gate.fill_(value)


class SinkGatedLongformerAttention(RelPositionMultiHeadAttentionLongformer):
    """Longformer attention whose sink logits carry a learnable per-head offset.

    Never constructed directly: `enable_sink_gate` re-classes the modules NeMo already built, so
    the pretrained q/k/v are kept as they are rather than rebuilt and copied.
    """

    def _compute_global_key_attn(self, *args, **kwargs) -> torch.Tensor:
        scores = super()._compute_global_key_attn(*args, **kwargs)
        # (batch, time, head, num_sinks); the gate is per head, hence the trailing axis.
        return scores + self.sink_gate.view(1, 1, -1, 1)


def enable_sink_gate(encoder: nn.Module, init_logit: float = -20.0) -> int:
    """Give every Longformer attention in `encoder` a sink gate, initialized closed.

    Args:
        encoder: A `ConformerEncoder` whose layers use `rel_pos_local_attn`.
        init_logit: Starting value of the gate. At -20 the sinks are numerically invisible; 0
            would reproduce NeMo's ungated behaviour.

    Returns:
        How many attention modules were gated.
    """
    gated = 0
    for layer in encoder.layers:
        attention = getattr(layer, 'self_attn', None)
        if not isinstance(attention, RelPositionMultiHeadAttentionLongformer):
            continue
        if getattr(attention, 'global_tokens', 0) <= 0:
            continue
        # Re-class rather than rebuild: the module keeps its pretrained parameters and buffers,
        # and only gains the gate plus the overridden score computation.
        attention.__class__ = SinkGatedLongformerAttention
        attention.register_parameter(
            'sink_gate', nn.Parameter(torch.full((attention.h,), float(init_logit)))
        )
        gated += 1

    if gated:
        logging.info(
            "Attention sinks gated in %d layers at logit %.1f (softmax weight %.1e)",
            gated,
            init_logit,
            float(torch.tensor(init_logit).exp()),
        )
    return gated
