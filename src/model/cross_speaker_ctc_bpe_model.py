"""Pandemonium: diarization-conditioned CTC with cross-speaker attention + mask refinement.

DiCoP-CTC (`src/model/ctc_bpe_model.py`) encodes each (session, speaker) stream on its own. This
model batches a whole session's speakers together, so its encoder --
`src/model/modules/cross_speaker_encoder.py::ConformerEncoderCrossSpeaker` -- can let those streams attend
across one another at several Conformer layers and re-estimate the STNO mask from what that
reveals, conditioning the deeper layers on a refined mask.

Only the batch plumbing changes here; the head, loss, decoding and multi-talker scoring are the
CTC model's. One dataloader item is a whole session carrying, per speaker (up to
`num_speaker_slots`), that speaker's target-speaker STNO mask and transcript (see
`src/data/dataset.py::build_speaker_group`). The collate stacks sessions and pads the speaker axis
to `Gmax`; this model flattens `(S, Gmax)` to `S*Gmax` encoder rows (the audio, shared by a
session's speakers, is expanded here rather than copied in the loader), tells the encoder the
grouping, and drops the padding speaker slots before the loss and the metric.

At initialization the cross-speaker attention is exactly inert and the refiner all but (its blend gate starts
where the attention-sink gate does), so the encoder starts within ~3e-4 of DiCoP-CTC and a warm
start from Parakeet or a DiCoP-CTC checkpoint loads cleanly (the cross-speaker attention and refiner tensors
are the only missing keys).
"""

from typing import Dict, List, Optional, Tuple

import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig

from nemo.core.classes.mixins import AccessMixin
from nemo.utils import logging

from src.model.ctc_bpe_model import EncDecCTCBPEModelSTNO

__all__ = ['EncDecCrossSpeakerCTCBPEModelSTNO']


class EncDecCrossSpeakerCTCBPEModelSTNO(EncDecCTCBPEModelSTNO):
    """CTC target-speaker model whose encoder communicates across a session's speakers."""

    group_speakers = True

    # The cross-speaker attention and refiner blocks exist in no pretrained checkpoint, on top of the CTC
    # model's own expected-missing set (the FDDT blocks and the borrowed transducer joint).
    PRETRAINED_EXPECTED_MISSING = EncDecCTCBPEModelSTNO.PRETRAINED_EXPECTED_MISSING + (
        '.cross_speaker.',
        '.mask_refiner.',
    )

    def __init__(self, cfg: DictConfig, trainer: Trainer = None, tokenizer=None):
        super().__init__(cfg=cfg, trainer=trainer, tokenizer=tokenizer)
        if not hasattr(self.encoder, 'set_grouping'):
            raise ValueError(
                "head=pandemonium requires encoder._target_ = "
                "src.model.modules.cross_speaker_encoder.ConformerEncoderCrossSpeaker; got "
                f"{type(self.encoder).__name__}."
            )

    @property
    def num_speaker_slots(self) -> int:
        """Maximum speaker streams per session; sessions with more lose the surplus as deletions."""
        return int(self._cfg.get('num_speaker_slots', 6))

    # Grouped-batch plumbing.

    def _flatten_group_batch(self, batch) -> Tuple:
        """Turn a `(S, Gmax, ...)` grouped batch into `S*Gmax` encoder rows plus the grouping.

        Returns `(signal, signal_len, transcript, transcript_len, stno_mask, stno_mask_len,
        utt_ids, spk_ids, speaker_valid, num_sessions, num_speakers_padded)`, everything but the
        last three flattened speaker-major (`row = s*Gmax + i`).
        """
        (audio, audio_len, tokens, token_len, stno_masks, stno_mask_len,
         utt_ids, spk_ids, num_speakers) = batch[:9]

        s, g = tokens.shape[0], tokens.shape[1]
        speaker_valid = torch.arange(g, device=num_speakers.device)[None, :] < num_speakers[:, None]

        signal = audio.repeat_interleave(g, dim=0)
        signal_len = audio_len.repeat_interleave(g, dim=0)
        transcript = tokens.reshape(s * g, tokens.shape[-1])
        transcript_len = token_len.reshape(s * g)
        stno_mask = stno_masks.reshape(s * g, stno_masks.shape[2], stno_masks.shape[3])
        stno_mask_length = stno_mask_len.repeat_interleave(g, dim=0)
        utt_ids_flat = utt_ids.repeat_interleave(g, dim=0)
        spk_ids_flat = spk_ids.reshape(s * g)

        return (signal, signal_len, transcript, transcript_len, stno_mask, stno_mask_length,
                utt_ids_flat, spk_ids_flat, speaker_valid, s, g)

    def _encode_group(self, signal, signal_len, stno_mask, stno_mask_len, speaker_valid, s, g):
        """Declare the grouping to the encoder and run the encoder forward."""
        self.encoder.set_grouping(s, g, speaker_valid)
        return self.forward(
            input_signal=signal,
            input_signal_length=signal_len,
            stno_mask=stno_mask,
            stno_mask_length=stno_mask_len,
        )

    def _gate_logs(self) -> Dict[str, torch.Tensor]:
        """Mean cross-speaker and refiner gate values, so a stuck gate is visible rather than assumed."""
        logs = {}
        gates = self.encoder.cross_speaker_gate_values()
        if gates:
            logs['cross_speaker_gate'] = torch.stack(gates).mean()
        refine = self.encoder.refine_gate_values()
        if refine:
            logs['refine_gate'] = torch.stack(refine).mean()
        return logs

    # Training / evaluation.

    def training_step(self, batch, batch_nb):
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        (signal, signal_len, transcript, transcript_len, stno_mask, stno_mask_len,
         _, _, speaker_valid, s, g) = self._flatten_group_batch(batch)

        encoded, encoded_len = self._encode_group(
            signal, signal_len, stno_mask, stno_mask_len, speaker_valid, s, g
        )
        del signal

        log_probs = self.decoder(encoder_output=encoded)

        # Score only the real speaker streams; padding slots carry a zero-length transcript and
        # would otherwise add the all-blank path's loss for a speaker that does not exist.
        valid = speaker_valid.reshape(s * g)
        loss_value = self.loss(
            log_probs=log_probs[valid],
            targets=transcript[valid],
            input_lengths=encoded_len[valid],
            target_lengths=transcript_len[valid],
        )
        loss_value = self.add_auxiliary_losses(loss_value)

        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        logs_dict = {
            'train_loss': loss_value,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
        }
        logs_dict.update(self._gate_logs())
        self.log_dict(logs_dict)
        return {'loss': loss_value}

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        (signal, signal_len, transcript, transcript_len, stno_mask, stno_mask_len,
         utt_ids, spk_ids, speaker_valid, s, g) = self._flatten_group_batch(batch)

        self.encoder.set_grouping(s, g, speaker_valid)
        encoded, encoded_len = self._eval_encode_with_fallback(
            signal, signal_len, stno_mask, stno_mask_len, processed=False, batch_idx=batch_idx
        )
        del signal

        predictions, prediction_lengths = self._eval_predictions(encoded, encoded_len)

        valid = speaker_valid.reshape(s * g)
        logs_dict = {}
        if self.compute_eval_loss:
            logs_dict = self._eval_loss(
                encoded[valid], encoded_len[valid], predictions[valid], transcript[valid],
                transcript_len[valid],
            )

        # One hypothesis per real (session, speaker); padding slots are dropped so the metric keys
        # on genuine speakers exactly as the target-speaker path does.
        self._eval_metric(dataloader_idx).update(
            predictions=predictions[valid],
            predictions_lengths=prediction_lengths[valid],
            utt_ids=utt_ids[valid],
            spk_ids=spk_ids[valid],
        )

        secondary = self._secondary_metrics(self._eval_stage())
        if secondary is not None:
            secondary[dataloader_idx].update(
                predictions=encoded[valid],
                predictions_lengths=encoded_len[valid],
                utt_ids=utt_ids[valid],
                spk_ids=spk_ids[valid],
            )

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))
        return logs_dict

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        (signal, signal_len, _, _, stno_mask, stno_mask_len,
         utt_ids, spk_ids, speaker_valid, s, g) = self._flatten_group_batch(batch)

        self.encoder.set_grouping(s, g, speaker_valid)
        encoded, encoded_len = self.forward(
            input_signal=signal,
            input_signal_length=signal_len,
            stno_mask=stno_mask,
            stno_mask_length=stno_mask_len,
        )
        del signal

        valid = speaker_valid.reshape(s * g)
        best_hyp_text = self._decode_hypotheses(encoded[valid], encoded_len[valid], return_hypotheses=True)
        return list(zip(utt_ids[valid].cpu().tolist(), spk_ids[valid].cpu().tolist(), best_hyp_text))
