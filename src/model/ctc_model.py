"""Diarization-conditioned CTC target-speaker ASR model.

The CTC counterpart of `src/model/asr_model.py`: same FDDT-conditioned encoder and multi-talker
evaluation (from `STNOModelMixin`), with the transducer's joint reused as a frame-wise head
(`src/model/modules/joint_ctc_decoder.py`) and its prediction network dropped.

It exists because the TDT loss cannot train on whole sessions at all: its numba kernels launch
one thread per target token (`gpu_rnnt.py`, `[minibatch_, maxU_]`) and a CUDA block caps at
1024, quite apart from the `(B, T, U, V)` tensor. CTC has no `U` axis.

`forward` returns the encoder output, not NeMo's `(log_probs, encoded_len, predictions)` triple
-- `output_types` is overridden to say so, and the head is applied in `training_step` and the
hooks.
"""

from typing import Dict, List, Optional

import torch

from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.classes.mixins import AccessMixin
from nemo.core.neural_types import AcousticEncodedRepresentation, LengthsType, NeuralType
from nemo.utils import logging

from src.model.ctc_timestamps import emission_word_offsets
from src.model.modules.joint_ctc_decoder import JointCTCDecoder
from src.model.modules.sink_attention import set_sink_gate
from src.model.stno_base import EVAL_STAGES, STNOModelMixin

__all__ = ['EncDecCTCModelSTNO', 'EVAL_STAGES']


class EncDecCTCModelSTNO(STNOModelMixin, EncDecCTCModel):
    """Base class for diarization-conditioned encoder-decoder CTC models."""

    # Only the FDDT blocks cannot be filled from a transducer checkpoint. `tdt_joint.*` is
    # missing by construction: those modules are the head's, borrowed, so they load under
    # `decoder.*` instead.
    PRETRAINED_EXPECTED_MISSING = ('.fddts.', 'tdt_joint.')
    PRETRAINED_EXPECTED_UNEXPECTED = ('tdt_prediction',)

    def adapt_pretrained_state_dict(self, state_dict):
        """Two prefix renames, both from the original keys so they cannot chain:

            joint.*              -> decoder.*          the CTC head, which is that joint
            decoder.prediction.* -> tdt_prediction.*   for validation-time TDT decoding
        """
        if not isinstance(self.decoder, JointCTCDecoder):
            return state_dict

        adapted = {}
        for key, value in state_dict.items():
            if key.startswith('joint.'):
                adapted[f"decoder.{key[len('joint.'):]}"] = value
            elif key.startswith('decoder.'):
                adapted[f"tdt_prediction.{key[len('decoder.'):]}"] = value
            else:
                adapted[key] = value
        return adapted

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Encoder output, not log-probabilities: see the module docstring."""
        return {
            "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    def maybe_freeze_decoder(self) -> None:
        """Hold the CTC head at its pretrained values, so only the encoder and FDDT blocks train.

        Keeps the transducer reassemblable: the head is Parakeet's joint, and TDT decoding puts
        it back alongside the prediction network trained against it. Measured effect is small
        either way (0.360 vs 0.366 cpWER at epoch 3), so this is insurance, not a win.

        Also held in eval mode, so its pretrained 0.2 dropout stops sampling noise.
        """
        if not self._cfg.get('freeze_decoder', False):
            return
        for parameter in self.decoder.parameters():
            parameter.requires_grad = False
        self.decoder.eval()
        frozen = sum(p.numel() for p in self.decoder.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(
            "CTC head frozen: %.1fM parameters held at their pretrained values, %.1fM trainable",
            frozen / 1e6,
            trainable / 1e6,
        )

    def _speaker_count_loss(self, num_speakers):
        """Predict how many speakers the session holds, from what the sinks pooled.

        The point is to give the sinks something to carry. Counting speakers cannot be done from
        a 256-frame window, so the gradient rewards exactly the global pooling they exist for,
        and unlike CTC it needs no alignment. Returns `(loss, accuracy)`, or `(None, None)` when
        the objective is off, the batch has no counts, or there are no sinks.
        """
        if getattr(self, 'speaker_count_head', None) is None or num_speakers is None:
            return None, None
        sinks = getattr(self.encoder, 'sink_output', None)
        if sinks is None:
            return None, None

        logits = self.speaker_count_head(sinks.mean(dim=1))  # (B, max_speakers + 1)
        target = num_speakers.to(logits.device).long().clamp(max=logits.shape[-1] - 1)
        loss = torch.nn.functional.cross_entropy(logits, target)
        accuracy = (logits.argmax(dim=-1) == target).float().mean()
        return loss, accuracy

    def on_load_checkpoint(self, checkpoint):
        """Let a checkpoint move between `eval_tdt` on and off.

        `eval_tdt` decides whether `tdt_prediction`/`tdt_joint` exist, and Lightning restores
        strictly, so flipping it would otherwise make every earlier checkpoint unresumable. Those
        tensors are validation-only -- the joint is borrowed from the head and the prediction
        network never trains -- so dropping or missing them costs nothing.
        """
        state = checkpoint.get('state_dict') or {}
        prefixes = ('tdt_prediction.', 'tdt_joint.')
        in_checkpoint = [key for key in state if key.startswith(prefixes)]
        has_stack = getattr(self, 'tdt_decoding', None) is not None

        if in_checkpoint and not has_stack:
            for key in in_checkpoint:
                del state[key]
            logging.info("Dropped %d transducer-eval tensors from the checkpoint; eval_tdt is off",
                         len(in_checkpoint))
        elif has_stack and not in_checkpoint:
            # The reverse: turning eval_tdt on for a run that was saved without it. The stack is
            # rebuilt from the pretrained model instead.
            self.strict_loading = False
            logging.info("Checkpoint has no transducer-eval tensors; loading non-strictly")

        super().on_load_checkpoint(checkpoint)

    def on_train_batch_start(self, batch, batch_idx):
        """Anneal the sink gate from its init toward 0 over `sink_gate_warmup_steps`.

        Left to itself the gate does not open: the sinks are only useful once read, and only get
        gradient once read, so a learnable gate settles shut (measured: mean drifting from -8.0
        to -8.3 over 4000 steps). A schedule forces the model to take them, while still starting
        from a warm-start-safe value. Overrides whatever the parameter holds, so the gate should
        be frozen (`sink_gate_lr_multiplier: 0`) when this is on.
        """
        warmup = int(self._cfg.get('sink_gate_warmup_steps', 0) or 0)
        if warmup <= 0:
            return
        init = float(self._cfg.encoder.get('sink_gate_init') or 0.0)
        progress = min(1.0, self.trainer.global_step / warmup)
        set_sink_gate(self.encoder, init * (1.0 - progress))

    def train(self, mode: bool = True):
        """Keep the frozen modules in eval mode, which `Module.train()` would otherwise undo."""
        super().train(mode)
        if self._cfg.get('freeze_decoder', False):
            self.decoder.eval()
        if getattr(self, 'tdt_prediction', None) is not None:
            self.tdt_prediction.eval()
        return self

    # PTL-specific methods

    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        signal, signal_len, transcript, transcript_len, stno_mask, stno_mask_len, _, _ = batch[:8]
        num_speakers = batch[8] if len(batch) > 8 else None

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

        log_probs = self.decoder(encoder_output=encoded)

        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )
        loss_value = self.add_auxiliary_losses(loss_value)

        speaker_count_loss, speaker_count_acc = self._speaker_count_loss(num_speakers)
        if speaker_count_loss is not None:
            loss_value = loss_value + float(self._cfg.speaker_count_loss_weight) * speaker_count_loss

        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        logs_dict = {
            'train_loss': loss_value,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
        }

        # Whether the sinks are being let in: a gate that never moves looks exactly like one
        # doing its job, so log it rather than assume.
        gates = [layer.self_attn.sink_gate for layer in self.encoder.layers
                 if hasattr(getattr(layer, 'self_attn', None), 'sink_gate')]
        if gates:
            logs_dict['sink_gate'] = torch.stack(gates).mean()
            logs_dict['sink_gate_max'] = torch.stack(gates).max()
        if speaker_count_loss is not None:
            logs_dict['speaker_count_loss'] = speaker_count_loss
            logs_dict['speaker_count_acc'] = speaker_count_acc

        # Off by default: a decode every logged step, and checkpoints are selected on cpWER.
        if self.cfg.get('log_training_wer', False) and (sample_id + 1) % log_every_n_steps == 0:
            self.wer.update(
                predictions=log_probs,
                predictions_lengths=encoded_len,
                targets=transcript,
                targets_lengths=transcript_len,
            )
            wer, _, _ = self.wer.compute()
            self.wer.reset()
            logs_dict.update({'training_batch_wer': wer})

        self.log_dict(logs_dict)

        return {'loss': loss_value}

    # Hooks from `STNOModelMixin`.

    def _eval_predictions(self, encoded, encoded_len):
        """CTC decodes log-probabilities, so the head runs here rather than in the metric."""
        return self.decoder(encoder_output=encoded), encoded_len

    def _eval_loss(self, encoded, encoded_len, predictions, transcript, transcript_len) -> Dict[str, torch.Tensor]:
        """The CTC loss over the log-probabilities already computed for scoring.

        Unlike the transducer's, this one is cheap enough to leave on at session length -- the
        alignment lattice is `(B, T, U)` rather than `(B, T, U, V)`.
        """
        if not self.compute_eval_loss:
            return {}
        return {
            'val_loss': self.loss(
                log_probs=predictions,
                targets=transcript,
                input_lengths=encoded_len,
                target_lengths=transcript_len,
            )
        }

    def _decode_hypotheses(self, encoded, encoded_len, return_hypotheses: bool = True) -> List[Hypothesis]:
        return emission_word_offsets(self.decoding.ctc_decoder_predictions_tensor(
            decoder_outputs=self.decoder(encoder_output=encoded),
            decoder_lengths=encoded_len,
            return_hypotheses=return_hypotheses,
        ))

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """No pretrained models are published under this class."""
        return []
