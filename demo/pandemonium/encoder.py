"""Inference-only Pandemonium encoder: FastConformer + FDDT + attention sinks + cross-speaker attention.

Vendored from the PolyKeet training repo (`ConformerEncoderSTNO` / `ConformerEncoderCrossSpeaker`), with
the training-only paths removed (streaming caches, activation checkpointing, stochastic depth,
interCTC capture) and one inference-only change: the audio is shared by every speaker row and the
first speaker-dependent operation is layer 0's FDDT, so the mel front-end and the 8x subsampling
run ONCE and are expanded to the speaker rows afterwards. Numerically identical to running them
per row, and it is what lets an hour of audio through on one GPU.
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn

from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder

from .modules import FDDT, CrossSpeakerAttention, MaskRefiner, enable_sink_gate

# The class the prepended sink tokens carry in the STNO mask (`target`); FDDT never reads it.
SINK_STNO_CLASS = 1


class PandemoniumEncoder(ConformerEncoder):
    """One session at a time: `features` is the shared audio, `stno_mask` has one row per speaker."""

    def __init__(
        self,
        *args,
        cross_speaker_layers: Optional[Sequence[int]] = None,
        refine_layers: Optional[Sequence[int]] = None,
        cross_speaker_heads: int = 8,
        fddt_non_target_rate: float = 0.5,
        learnable_sink_tokens: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.learnable_sink_tokens = learnable_sink_tokens and self.global_tokens > 0
        if self.learnable_sink_tokens:
            self.sink_embeddings = nn.Parameter(torch.zeros(self.global_tokens, self.d_model))
        enable_sink_gate(self)
        self.fddts = nn.ModuleList([FDDT(self.d_model, fddt_non_target_rate) for _ in self.layers])
        self.cross_speaker = nn.ModuleDict(
            {str(i): CrossSpeakerAttention(self.d_model, cross_speaker_heads)
             for i in (cross_speaker_layers or [])}
        )
        self.mask_refiner = nn.ModuleDict({str(i): MaskRefiner(self.d_model) for i in (refine_layers or [])})

    @torch.inference_mode()
    def encode(self, features: torch.Tensor, feature_len: torch.Tensor, stno_mask: torch.Tensor,
               row_chunk: Optional[int] = None, row_frames: int = 48_000):
        """
        Args:
            features: `(1, n_mels, T_mel)` log-mel features of the session.
            feature_len: `(1,)` number of valid mel frames.
            stno_mask: `(G, 4, T_enc)` one STNO mask per speaker at the 80 ms encoder frame rate.
            row_chunk: run each Conformer layer over this many speaker rows at a time. `None`
                picks it from `row_frames`: as many rows as keep `rows x T_enc` under that budget,
                so typical meetings run all speakers at once and an hour runs one at a time
                (~12 GiB peak for 8 speakers x 1 h, against ~54 GiB unchunked).

        Returns:
            `(G, D, T_enc)` encoder output and its length `T_enc`.
        """
        g = stno_mask.shape[0]
        self.update_max_seq_length(seq_length=features.size(2), device=features.device)

        x, length = self.pre_encode(x=features.transpose(1, 2), lengths=feature_len)
        length = length.to(torch.int64)
        x = x.expand(g, -1, -1)
        length = length.expand(g)

        # Reconcile the mask with the actual subsampled length (it can differ by a frame). Zero
        # padding, as in training -- not "silence".
        t = x.shape[1]
        if stno_mask.shape[-1] > t:
            stno_mask = stno_mask[..., :t]
        elif stno_mask.shape[-1] < t:
            stno_mask = nn.functional.pad(stno_mask, (0, t - stno_mask.shape[-1]))
        stno_mask = stno_mask.to(x.dtype)

        sinks = self.global_tokens if self.learnable_sink_tokens else 0
        if sinks:
            x = torch.cat([self.sink_embeddings.to(x.dtype)[None].expand(g, -1, -1), x], dim=1)
            length = length + sinks
            prefix = stno_mask.new_zeros(g, 4, sinks)
            prefix[:, SINK_STNO_CLASS] = 1.0
            stno_mask = torch.cat([prefix, stno_mask], dim=-1)

        x, pos_emb = self.pos_enc(x=x, cache_len=0)
        pad_mask, att_mask = self._create_masks(
            att_context_size=self.att_context_size,
            padding_length=length,
            max_audio_length=x.size(1),
            offset=None,
            device=x.device,
        )

        # Speaker rows only interact in the cross-speaker attention / refiner blocks between layers, so each
        # Conformer layer can run a few rows at a time: same result, a fraction of the peak memory
        # (the local-attention scores of 8 speakers x 1 h are what dominate it).
        rows = row_chunk or max(1, row_frames // t)
        for lth, layer in enumerate(self.layers):
            x = torch.cat([x[:, :sinks], self.fddts[lth](x[:, sinks:], stno_mask[:, :, sinks:])], dim=1)
            x = torch.cat(
                [
                    layer(x=x[i:i + rows], att_mask=None if att_mask is None else att_mask[i:i + rows], pos_emb=pos_emb,
                          pad_mask=pad_mask[i:i + rows])
                    for i in range(0, g, rows)
                ],
                dim=0,
            )
            key = str(lth)
            if key in self.cross_speaker:
                x = self.cross_speaker[key](x, g)
            if key in self.mask_refiner:
                refined = self.mask_refiner[key](x, stno_mask)
                stno_mask = torch.cat([stno_mask[:, :, :sinks], refined[:, :, sinks:]], dim=-1)

        x = x[:, sinks:]
        if self.out_proj is not None:
            x = self.out_proj(x)
        return x.transpose(1, 2), t
