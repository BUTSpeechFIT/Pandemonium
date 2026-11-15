"""Diarization-conditioned RNNT/TDT target-speaker ASR model.

`EncDecRNNTModelSTNO` combines NeMo's `EncDecRNNTModel` with `STNOModelMixin`, which carries
everything this model shares with the CTC variant: the mask-threading encoder `forward`, the
multi-talker cpWER/tcpWER evaluation, the long-form memory workarounds and the FDDT parameter
group. See `src/model/stno_base.py`.

What is left here is what only makes sense for a transducer:

  - `training_step`, including the fused joint/loss path;
  - the three evaluation hooks, which decode and compute the loss through the joint;
  - `transcribe()` stays disabled (in the mixin) because NeMo's transcription path cannot
    supply a mask; `transcribe_stno()` is the conditioned replacement, used by `infer.py`.

Everything else - the module construction, the TDT loss, optimizer setup, export - is
inherited from NeMo unchanged.
"""

from typing import Dict, List, Optional

import torch

from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.core.classes.mixins import AccessMixin

from src.model.stno_base import EVAL_STAGES, STNOModelMixin

__all__ = ['EncDecRNNTModelSTNO', 'EVAL_STAGES']


class EncDecRNNTModelSTNO(STNOModelMixin, EncDecRNNTModel):
    """Base class for diarization-conditioned encoder-decoder RNNT models."""

    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        signal, signal_len, transcript, transcript_len, stno_mask, stno_mask_len, _, _ = batch[:8]

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(
                processed_signal=signal,
                processed_signal_length=signal_len,
                stno_mask=stno_mask,
                stno_mask_length=stno_mask_len,
            )
        else:
            encoded, encoded_len = self.forward(
                input_signal=signal,
                input_signal_length=signal_len,
                stno_mask=stno_mask,
                stno_mask_length=stno_mask_len,
            )
        del signal

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        # Off by default: a greedy decode of every logged step is expensive, and cpWER on the
        # validation set is the metric this model is actually selected on.
        log_training_wer = self.cfg.get('log_training_wer', False)

        if not self.joint.fuse_loss_wer:
            # Compute full joint and loss
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
            loss_value = self.loss(
                log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            logs_dict = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if log_training_wer and (sample_id + 1) % log_every_n_steps == 0:
                self.wer.update(
                    predictions=encoded,
                    predictions_lengths=encoded_len,
                    targets=transcript,
                    targets_lengths=transcript_len,
                )
                _, scores, words = self.wer.compute()
                self.wer.reset()
                logs_dict.update({'training_batch_wer': scores.float() / words})

        else:
            compute_wer = log_training_wer and (sample_id + 1) % log_every_n_steps == 0

            # Fused joint step
            loss_value, wer, _, _ = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=transcript_len,
                compute_wer=compute_wer,
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            logs_dict = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                logs_dict.update({'training_batch_wer': wer})

        self.log_dict(logs_dict)

        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {'loss': loss_value}
    def _eval_predictions(self, encoded, encoded_len):
        """The RNNT decoder consumes the encoder output directly."""
        return encoded, encoded_len

    def _eval_loss(self, encoded, encoded_len, predictions, transcript, transcript_len) -> Dict[str, torch.Tensor]:
        """The transducer loss over the full joint, when `compute_eval_loss` asks for it.

        Off by default: the joint is (B, T, U, V), which on whole sessions is the largest tensor
        in the step by a wide margin, and cpWER is what the model is selected on anyway.
        """
        if not self.compute_eval_loss:
            return {}
        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)
        joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
        return {
            'val_loss': self.loss(
                log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
            )
        }

    def _decode_hypotheses(self, encoded, encoded_len, return_hypotheses: bool = True) -> List[Hypothesis]:
        return self.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded,
            encoded_lengths=encoded_len,
            return_hypotheses=return_hypotheses,
        )
