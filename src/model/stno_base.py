"""Shared machinery for the diarization-conditioned (STNO) target-speaker ASR models.

`STNOModelMixin` holds everything the RNNT/TDT and CTC variants do identically: the
mask-threading encoder `forward`, the multi-talker cpWER/tcpWER evaluation loop with its
per-dataset dataloaders and STM output, the long-form memory workarounds (local-attention
fallback, optimizer offload), and the FDDT parameter group.

What differs between the two is confined to three hooks, which the concrete model implements:

  - `_eval_predictions` -- what the hypothesis accumulator is fed. RNNT decodes from the
    encoder output; CTC decodes from log-probabilities, so it has to run its head first.
  - `_eval_loss` -- the optional validation loss, whose inputs are architecture-specific.
  - `_decode_hypotheses` -- greedy decoding for `predict_step` / `transcribe_stno`.

`training_step` is not shared: it is where the two architectures differ most, and expressing
both through hooks would obscure rather than shorten it.

The mixin is listed *before* the NeMo base class, so its `forward` and the evaluation hooks
take precedence while `super()` still reaches NeMo's implementations.
"""

import copy
import gc
import os
from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig

from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.common.parts.optional_cuda_graphs import WithOptionalCudaGraphs
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import LengthsType, MaskType, NeuralType
from nemo.utils import logging

from src.data.lhotse_utils import named_manifests
from src.metrics.meeteval_mt_wer import pool_results

__all__ = ['STNOModelMixin', 'EVAL_STAGES']


# Where each evaluation stage keeps its state on the model. The dataloaders, NeMo's own display
# names and its "which loader owns the unprefixed metrics" index are NeMo's attributes; the
# dataset names and the per-dataloader metrics are this model's.
EVAL_STAGES = {
    'validation': {
        'prefix': 'val',
        'dataloaders': '_validation_dl',
        'nemo_names': '_validation_names',
        'dl_idx': '_val_dl_idx',
        'names': '_validation_dataset_names',
        'metrics': '_validation_metrics',
    },
    'test': {
        'prefix': 'test',
        'dataloaders': '_test_dl',
        'nemo_names': '_test_names',
        'dl_idx': '_test_dl_idx',
        'names': '_test_dataset_names',
        'metrics': '_test_metrics',
    },
}


class STNOModelMixin:
    """Diarization conditioning and multi-talker evaluation, shared by the RNNT and CTC models."""

    # Which key mismatches `train.py` accepts when warm-starting from a pretrained Parakeet.
    # The FDDT blocks exist in no released checkpoint, so they are always missing; a head that
    # the pretrained model does not share adds to this (see `EncDecCTCModelSTNO`).
    PRETRAINED_EXPECTED_MISSING = ('.fddts.',)
    PRETRAINED_EXPECTED_UNEXPECTED = ()

    # What shape the dataloaders emit. The default is the target-speaker layout: one item per
    # (session, speaker), conditioned on that speaker's mask. The multi-talker model overrides
    # both, taking whole sessions with one transcript per speaker slot and no conditioning at
    # all (not in this repository).
    multi_speaker = False
    num_speaker_slots = 1
    slot_order = 'first_occurrence'

    # The Pandemonium model overrides these, taking whole sessions with one target-speaker STNO
    # mask + transcript per speaker, batched together so its `ConformerEncoderCrossSpeaker` can attend
    # across speakers and refine the mask; see `src/model/cross_speaker_ctc_bpe_model.py`. Mutually
    # exclusive with `multi_speaker` and `sot`.
    group_speakers = False

    # The SOT (Serialized Output Training) model overrides all five, taking whole sessions with
    # one *serialized* transcript covering every speaker instead of one per slot (not in this
    # repository). Mutually exclusive with `multi_speaker`.
    sot = False
    sot_order = 'speaking_time'
    sot_bos_id = None
    sot_eos_id = None
    sot_speaker_change_id = None

    def adapt_pretrained_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Rename a pretrained checkpoint's keys to this model's layout, before loading it.

        The identity here: a transducer model's modules sit where this one's do. Overridden where
        that stops being true (see `EncDecCTCJointModelSTNO`).
        """
        return state_dict
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        types = dict(super().input_types)
        types["stno_mask"] = NeuralType(('B', 'S', 'T'), MaskType(), optional=True)
        types["stno_mask_length"] = NeuralType(tuple('B'), LengthsType(), optional=True)
        return types

    @typecheck()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        stno_mask=None,
        stno_mask_length=None,
    ):
        """Encoder forward pass, conditioned on the diarization-derived STNO mask.

        As in the base class this is only the first of the three RNNT forward steps; the
        prediction and joint networks are applied in `training_step` / `validation_pass`.

        Args:
            input_signal: Raw audio batch of shape (B, T).
            input_signal_length: Per-example sample counts, shape (B,).
            processed_signal: Pre-computed features of shape (B, D, T); mutually exclusive
                with `input_signal`.
            processed_signal_length: Per-example feature frame counts.
            stno_mask: (B, 4, T_enc) silence/target/non-target/overlap indicators at the
                encoder frame rate. `None` disables conditioning entirely.
            stno_mask_length: Per-example unpadded mask lengths, shape (B,).

        Returns:
            `(encoded, encoded_len)` with `encoded` of shape (B, D, T_enc).
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(
            audio_signal=processed_signal,
            length=processed_signal_length,
            stno_mask=stno_mask,
            stno_mask_length=stno_mask_length,
        )
        return encoded, encoded_len

    # Hooks the concrete model implements; see the module docstring.

    def _eval_predictions(self, encoded, encoded_len):
        """What `MeetevalMTWER` decodes: `(predictions, lengths)`.

        The metric dispatches on the type of the decoding object it was built with, and the two
        want different tensors -- an RNNT decoder consumes the encoder output directly, a CTC one
        consumes log-probabilities.
        """
        raise NotImplementedError

    def _eval_loss(self, encoded, encoded_len, predictions, transcript, transcript_len) -> Dict[str, torch.Tensor]:
        """The validation loss, as a dict to merge into the logs, or `{}` when not computed.

        `predictions` is what `_eval_predictions` produced, so a CTC model can reuse its
        log-probabilities rather than running its head a second time.
        """
        raise NotImplementedError

    def _decode_hypotheses(self, encoded, encoded_len, return_hypotheses: bool = True) -> List[Hypothesis]:
        """Greedy decoding of an encoded batch, with word timestamps when available."""
        raise NotImplementedError

    # PTL-specific methods

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, _, _, stno_mask, stno_mask_len, utt_ids, spk_ids = batch[:8]

        encoded, encoded_len = self.forward(
            input_signal=signal,
            input_signal_length=signal_len,
            stno_mask=stno_mask,
            stno_mask_length=stno_mask_len,
        )
        del signal

        best_hyp_text = self._decode_hypotheses(encoded, encoded_len, return_hypotheses=True)
        return list(zip(utt_ids.cpu().tolist(), spk_ids.cpu().tolist(), best_hyp_text))

    def _eval_encode_with_fallback(self, signal, signal_len, stno_mask, stno_mask_len, processed, batch_idx):
        """Encoder forward for evaluation, retrying under local attention if it will not fit.

        Factored out of `validation_pass` so a model that overrides that -- the multi-talker one
        decodes eight speaker streams and cannot use the single-hypothesis path -- keeps the
        long-form memory behaviour rather than reimplementing it. See `_set_local_attention`.
        """
        retry_with_local_attn = False
        try:
            encoded, encoded_len = self._eval_encoder_forward(
                signal, signal_len, stno_mask, stno_mask_len, processed
            )
        except torch.OutOfMemoryError:
            if not self._local_attn_fallback_available():
                raise
            retry_with_local_attn = True

        if retry_with_local_attn:
            # Reclaiming happens here, outside the handler, and the distinction matters. While an
            # `except` block runs, the exception it is handling still references the frames of the
            # failed forward, and those frames still reference every intermediate activation - so
            # `empty_cache()` in there returns only the unused blocks and leaves tens of GiB live.
            # The retry then dies on its first megabyte, in the preprocessor, nowhere near the
            # allocation that actually overflowed. Once the handler has exited the traceback is
            # dropped, and collecting is what genuinely frees the memory.
            gc.collect()
            torch.cuda.empty_cache()
            logging.warning(
                "Out of memory on %s batch %d under full-context attention; retrying this and the "
                "rest of the evaluation with local attention (context %s). Scores from this epoch "
                "reflect local-attention decoding.",
                self._eval_stage(),
                batch_idx,
                list(self.cfg.get('local_attn_eval_context') or [256, 256]),
            )
            self._set_local_attention(True)
            encoded, encoded_len = self._eval_encoder_forward(
                signal, signal_len, stno_mask, stno_mask_len, processed
            )
        return encoded, encoded_len

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len, stno_mask, stno_mask_len, utt_ids, spk_ids = batch[:8]

        # forward() only performs encoder forward
        processed = isinstance(batch, DALIOutputs) and batch.has_processed_signal
        encoded, encoded_len = self._eval_encode_with_fallback(
            signal, signal_len, stno_mask, stno_mask_len, processed, batch_idx
        )
        del signal

        predictions, prediction_lengths = self._eval_predictions(encoded, encoded_len)
        logs_dict = self._eval_loss(encoded, encoded_len, predictions, transcript, transcript_len)

        # Multi-talker scoring replaces token WER: hypotheses are accumulated per
        # (session, speaker) here and scored once per epoch in `_eval_epoch_end`.
        self._eval_metric(dataloader_idx).update(
            predictions=predictions,
            predictions_lengths=prediction_lengths,
            utt_ids=utt_ids,
            spk_ids=spk_ids,
        )

        # The second decoding consumes the encoder output directly, as a transducer does; one
        # encoder forward serves both.
        secondary = self._secondary_metrics(self._eval_stage())
        if secondary is not None:
            secondary[dataloader_idx].update(
                predictions=encoded,
                predictions_lengths=encoded_len,
                utt_ids=utt_ids,
                spk_ids=spk_ids,
            )

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        return logs_dict

    def _stm_output_dir(self, name: Optional[str] = None) -> str:
        """Where one dataset's `ref.stm` / `hyp.stm` go, in `run_inference.sh`'s `{name}/` layout."""
        log_dir = self.trainer.log_dir if self.trainer is not None else '.'
        parts = [log_dir or '.', f'preds_{self.current_epoch}_{self.trainer.global_step}']
        if name:
            parts.append(name)
        path = os.path.join(*parts)
        os.makedirs(path, exist_ok=True)
        return path

    def _dataloader_collection(self, dataloader, dataloader_idx: int):
        """The reference collection backing a (possibly multi-) dataloader.

        Both manifest formats expose the fields the multi-talker metric needs, under different
        names: the Lhotse dataset builds them on the fly from its cuts, the NeMo one keeps them
        on its manifest processor.
        """
        if isinstance(dataloader, (list, tuple)):
            dataloader = dataloader[dataloader_idx]
        dataset = dataloader.dataset

        if hasattr(dataset, 'segments_collection'):
            return dataset.segments_collection
        if hasattr(dataset, 'manifest_processor'):
            return dataset.manifest_processor.collection
        raise AttributeError(
            f"{type(dataset).__name__} exposes neither `segments_collection` nor "
            f"`manifest_processor`, so multi-talker scoring cannot find its references."
        )

    def _score_eval_dataloader(self, outputs, stage: str, dataloader_idx: int, name: Optional[str]):
        """Score one dataset's accumulated hypotheses and dump its STMs.

        Returns the log entries for this dataset, keyed `val/<name>/cp_wer` and so on, together
        with the raw cpWER and tcpWER counts, which `_eval_epoch_end` sums into the pooled number
        checkpoints are selected on.
        """
        stage_attrs = EVAL_STAGES[stage]
        prefix = stage_attrs['prefix']
        scope = f'{prefix}/{name}' if name else prefix

        save_stm_path = self._stm_output_dir(name)
        logging.info("Saving %s predictions to %s", scope, save_stm_path)

        metric = self._eval_metrics(stage)[dataloader_idx]
        cp_res, tcp_res = metric.compute(
            self._dataloader_collection(getattr(self, stage_attrs['dataloaders']), dataloader_idx),
            save_stm_path=save_stm_path,
        )
        metric.reset()

        logs = {}
        if self.compute_eval_loss:
            logs[f'{scope}/loss'] = torch.stack([x[f'{prefix}_loss'] for x in outputs]).mean()
        for metric_name, res in (('cp', cp_res), ('tcp', tcp_res)):
            if res is None:  # tcpWER disabled on this metric (see MeetevalMTWER.compute_tcpwer)
                continue
            for key in ('wer', 'ins', 'del', 'sub', 'len'):
                logs[f'{scope}/{metric_name}_{key}'] = float(res[key])

        secondary_metrics = self._secondary_metrics(stage)
        secondary_cp = None
        if secondary_metrics is not None:
            tag = self.secondary_metric_tag
            # Its own STM directory, or the two decodings overwrite each other's hyp.stm.
            secondary_metric = secondary_metrics[dataloader_idx]
            secondary_stm_path = os.path.join(save_stm_path, tag)
            os.makedirs(secondary_stm_path, exist_ok=True)
            secondary_cp, secondary_tcp = secondary_metric.compute(
                self._dataloader_collection(getattr(self, stage_attrs['dataloaders']), dataloader_idx),
                save_stm_path=secondary_stm_path,
            )
            secondary_metric.reset()
            for metric_name, res in (('cp', secondary_cp), ('tcp', secondary_tcp)):
                for key in ('wer', 'ins', 'del', 'sub', 'len'):
                    logs[f'{scope}/{tag}_{metric_name}_{key}'] = float(res[key])

        return logs, cp_res, tcp_res, secondary_cp

    def _eval_epoch_end(self, stage: str):
        """Score every dataset of `stage` separately, then log them and their pooled aggregate.

        Replaces `ModelPT.on_validation_epoch_end`, which prefixes the second and later
        dataloaders' metrics as `<name>_val/cp_wer` and has nowhere to put a number pooled across
        them. `exp_manager` selects checkpoints on `val/cp_wer`, so that number has to exist
        however many datasets are configured — with one it is simply that dataset's own.
        """
        stage_attrs = EVAL_STAGES[stage]
        prefix = stage_attrs['prefix']
        outputs = self.validation_step_outputs if stage == 'validation' else self.test_step_outputs
        if not outputs:
            return {}

        # A single dataloader reports a flat list of step outputs, several report one list each.
        per_dataloader = [outputs] if isinstance(outputs[0], dict) else list(outputs)
        names = getattr(self, stage_attrs['names'], None) or [None] * len(per_dataloader)

        logs = {}
        if self.compute_eval_loss:
            losses = [step[f'{prefix}_loss'] for outs in per_dataloader for step in outs]
            logs[f'{prefix}_loss'] = torch.stack(losses).mean()

        results = []
        secondary_results = []
        for dataloader_idx, dataloader_outputs in enumerate(per_dataloader):
            dataloader_logs, cp_res, tcp_res, secondary_cp = self._score_eval_dataloader(
                dataloader_outputs, stage, dataloader_idx, names[dataloader_idx]
            )
            logs.update(dataloader_logs)
            results.append((cp_res, tcp_res))
            if secondary_cp is not None:
                secondary_results.append(secondary_cp)
            dataloader_outputs.clear()  # free memory

        if any(names):
            pooled_metrics = [('cp', pool_results([cp_res for cp_res, _ in results]))]
            # tcpWER may be disabled (MeetevalMTWER.compute_tcpwer); skip pooling it if so.
            if all(tcp_res is not None for _, tcp_res in results):
                pooled_metrics.append(('tcp', pool_results([tcp_res for _, tcp_res in results])))
            for metric_name, pooled in pooled_metrics:
                for key in ('wer', 'ins', 'del', 'sub', 'len'):
                    logs[f'{prefix}/{metric_name}_{key}'] = float(pooled[key])

        # Pooled unconditionally, unlike the primary: this is only read by a human comparing the
        # two decodings, so it should exist with one dataset as well as several.
        if secondary_results:
            pooled = pool_results(secondary_results)
            for key in ('wer', 'ins', 'del', 'sub', 'len'):
                logs[f'{prefix}/{self.secondary_metric_tag}_cp_{key}'] = float(pooled[key])

        # `sync_dist` matches the `sync_metrics=True` NeMo passes here; the scores are already
        # reduced across ranks by the metric itself, so it only averages equal values.
        self.log_dict(logs, on_epoch=True, sync_dist=True)
        return {'log': logs}

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        """NeMo's per-dataloader hook, kept to its contract.

        `on_validation_epoch_end` scores directly instead, because it also needs the raw counts
        each dataset contributes to the pooled aggregate.
        """
        names = getattr(self, EVAL_STAGES['validation']['names'], None)
        logs, _, _ = self._score_eval_dataloader(
            outputs, 'validation', dataloader_idx, names[dataloader_idx] if names else None
        )
        return {'log': logs}

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        """See `multi_validation_epoch_end`."""
        names = getattr(self, EVAL_STAGES['test']['names'], None)
        logs, _, _ = self._score_eval_dataloader(
            outputs, 'test', dataloader_idx, names[dataloader_idx] if names else None
        )
        return {'log': logs}

    def on_validation_epoch_end(self):
        """See `_eval_epoch_end`."""
        # All `ASRModel.on_validation_epoch_end` does before delegating to the logging this
        # replaces: validation runs inside training, which continues with the graphs disabled.
        # Only the RNNT model has them -- CTC greedy decoding captures no graphs.
        if isinstance(self, WithOptionalCudaGraphs):
            self.disable_cuda_graphs()
        # Training runs on full context whatever the fallback did to get through evaluation.
        self._set_local_attention(False)
        self._offload_training_state(to_cpu=False)
        return self._eval_epoch_end('validation')

    def on_test_epoch_end(self):
        """See `_eval_epoch_end`. Unlike validation, nothing resumes after testing."""
        return self._eval_epoch_end('test')

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        # Long-form sessions leave the allocator badly fragmented between phases.
        torch.cuda.empty_cache()

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        # Greedy decoding of variable-length long-form audio is incompatible with the captured
        # cuda graphs, which assume a fixed shape.
        WithOptionalCudaGraphs.disable_cuda_graphs_recursive(self, attribute_path="decoding.decoding")
        self._offload_training_state(to_cpu=True)
        torch.cuda.empty_cache()
        self._build_eval_metrics('validation')

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self._build_eval_metrics('test')

    # Local-attention fallback for long-form evaluation.
    #
    # Full-context `rel_pos` attention materializes an (H, T, T) relative-position bias, and
    # `rel_shift` pads that into a second copy. Evaluation decodes whole sessions one at a time,
    # so T is the entire recording: a 50-minute AMI dev session is 37k encoder frames at 12.5 Hz,
    # which puts that pair past 40 GiB and cannot be made to fit by lowering the batch size,
    # because it is already one session per batch.
    #
    # Off by default (`model.local_attn_oom_fallback=false`), because it changes what the score
    # means: a cut decoded this way saw sliding-window attention, not the full-context model
    # being trained. Enabled, it costs one wasted forward pass on the cut that overflows, then
    # holds local attention for the rest of the evaluation and restores full context before
    # training resumes -- so training itself is never affected. Under DDP each rank decides
    # independently, since only the rank that overflows needs to switch and evaluation does no
    # gradient synchronisation; the restore below runs on every rank regardless.
    def _eval_encoder_forward(self, signal, signal_len, stno_mask, stno_mask_len, processed: bool):
        if processed:
            return self.forward(
                processed_signal=signal,
                processed_signal_length=signal_len,
                stno_mask=stno_mask,
                stno_mask_length=stno_mask_len,
            )
        return self.forward(
            input_signal=signal,
            input_signal_length=signal_len,
            stno_mask=stno_mask,
            stno_mask_length=stno_mask_len,
        )

    # Optimizer offload for the duration of an evaluation.
    #
    # Evaluation decodes whole sessions, and the relative-position bias of full-context attention
    # is quadratic in their length: the 49.5-minute AMI dev session needs a 41.08 GiB fp32 tensor.
    # The run that overflowed was 5.83 GiB short of it, while AdamW's two moments and the
    # gradients -- about 7 GiB together at 0.6B parameters -- sat on the card doing nothing for
    # the duration of the decode. Moving them to host memory buys back more than the shortfall,
    # which is what lets a long session be scored at full context instead of falling back to a
    # windowed approximation of it.
    #
    # Both are moved rather than dropped, so a partially accumulated gradient survives, and the
    # optimizer looks its state up per key each step, which makes replacing the tensors safe. The
    # transfer costs a few seconds each way per validation epoch. The margin this leaves is thin
    # by construction -- a long enough session still will not fit, which is what the local
    # attention fallback is for.
    def _offload_training_state(self, to_cpu: bool) -> None:
        if not self.cfg.get('offload_optimizer_on_eval', True):
            return
        if self._trainer is None or not self.trainer.optimizers:
            return
        device = torch.device('cpu') if to_cpu else self.device
        moved = 0
        for optimizer in self.trainer.optimizers:
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value) and value.device.type != device.type:
                        moved += value.numel() * value.element_size()
                        state[key] = value.to(device)
        for parameter in self.parameters():
            if parameter.grad is not None and parameter.grad.device.type != device.type:
                moved += parameter.grad.numel() * parameter.grad.element_size()
                parameter.grad = parameter.grad.to(device)
        if moved:
            # Returns the emptied blocks to the driver: the bias tensor needs one large
            # contiguous segment, which the cached blocks would not satisfy on their own.
            torch.cuda.empty_cache()
            logging.info(
                "Optimizer state and gradients moved %s (%.2f GiB)",
                "to host memory for evaluation" if to_cpu else "back to the device",
                moved / 2**30,
            )

    def _local_attn_fallback_available(self) -> bool:
        """Whether an OOM should be retried with local attention, rather than raised."""
        if not self.cfg.get('local_attn_oom_fallback', False):
            return False
        # Already local: the retry would repeat the configuration that just failed.
        return not getattr(self, '_local_attn_active', False)

    def _set_local_attention(self, local: bool) -> None:
        if getattr(self, '_local_attn_active', False) == local:
            return
        encoder = self.encoder

        if local:
            # The originals are put aside as objects rather than as a description to rebuild
            # from, because `change_attention_model` constructs new modules: restoring that way
            # would hand training a fresh set of Parameters while the optimizer and DDP's
            # gradient reducer stay bound to the ones they were built with, and the attention
            # weights would then quietly stop being updated for the rest of the run. Reinstating
            # the original objects also allocates nothing, which matters because the restore runs
            # inside the evaluation loop's `inference_mode` -- tensors created there are inference
            # tensors, and DDP refuses to broadcast those.
            self._saved_attention = (
                encoder.pos_enc,
                [(layer, layer.self_attn, layer.self_attention_model) for layer in encoder.layers],
                encoder.self_attention_model,
                list(encoder.att_context_size),
                encoder.use_pytorch_sdpa,
            )
            context = list(self.cfg.get('local_attn_eval_context') or [256, 256])
            # NeMo's Longformer attention has no scaled-dot-product path and raises rather than
            # ignoring the flag. Nothing is lost by turning it off: SDPA is what keeps the
            # full-context bias tensor from being materialized, and the sliding-window
            # implementation never builds one to begin with. It is read off the encoder while the
            # replacement modules are constructed, so it has to be set before the call.
            encoder.use_pytorch_sdpa = False
            # `update_config=False`: the config keeps describing the trained topology.
            encoder.change_attention_model(
                self_attention_model='rel_pos_local_attn',
                att_context_size=context,
                update_config=False,
                device=self.device,
            )
            logging.info("Attention model set to rel_pos_local_attn with context %s", context)
        else:
            pos_enc, layers, model, context, sdpa = self._saved_attention
            encoder.pos_enc = pos_enc
            for layer, attn, attn_model in layers:
                layer.self_attn = attn
                layer.self_attention_model = attn_model
            encoder.self_attention_model = model
            encoder.att_context_size = context
            encoder.use_pytorch_sdpa = sdpa
            del self._saved_attention
            logging.info("Attention model restored to %s with context %s", model, context)

        self._local_attn_active = local

    def _eval_stage(self) -> str:
        """Which evaluation stage the trainer is currently running."""
        return 'test' if self._trainer is not None and self.trainer.testing else 'validation'

    def _eval_metric(self, dataloader_idx: int):
        """The hypothesis accumulator for one dataloader of the running evaluation stage.

        `validation_pass` serves both `validation_step` and `test_step`, which is why the stage is
        read off the trainer rather than passed in.
        """
        return self._eval_metrics(self._eval_stage())[dataloader_idx]

    def _eval_metrics(self, stage: str):
        """One `MeetevalMTWER` per dataloader of `stage`, built on first use.

        Lightning runs the dataloaders one after another but calls the epoch end only once they
        have all finished, so a single shared accumulator would pool every dataset's hypotheses
        and then score that pool against each dataset's references in turn. One accumulator per
        dataloader keeps them apart.
        """
        metrics = getattr(self, EVAL_STAGES[stage]['metrics'], None)
        return metrics if metrics is not None else self._build_eval_metrics(stage)

    def _build_eval_metrics(self, stage: str):
        """Build `stage`'s accumulators, one per dataloader.

        Not built in `setup_multiple_validation_data`: NeMo calls that from `ModelPT.__init__`,
        before `self.decoding` — which the metric decodes through — exists. Rebuilding them at
        every epoch start also keeps them in step with `change_decoding_strategy`.
        """
        stage_attrs = EVAL_STAGES[stage]
        names = getattr(self, stage_attrs['names'], None)
        dataloaders = getattr(self, stage_attrs['dataloaders'], None) or []
        count = len(names) if names else max(len(dataloaders), 1)

        # A plain list, not a `ModuleList`: these hold no parameters and their metric states are
        # non-persistent, so registering them would only add and remove state_dict entries as the
        # configured datasets change.
        metrics = [self._build_meeteval_metric().to(self.device) for _ in range(count)]
        setattr(self, stage_attrs['metrics'], metrics)

        # An optional second decoding of the same encoder output, logged under
        # `<scope>/<tag>_cp_wer`. The CTC model uses it to report the transducer's score.
        secondary = [self._build_secondary_metric() for _ in range(count)]
        setattr(
            self,
            self._secondary_attr(stage),
            [metric.to(self.device) for metric in secondary] if secondary[0] is not None else None,
        )
        return metrics

    @staticmethod
    def _secondary_attr(stage: str) -> str:
        return EVAL_STAGES[stage]['metrics'] + '_secondary'

    def _build_secondary_metric(self):
        """A second `MeetevalMTWER` decoding the encoder output another way, or `None`."""
        return None

    @property
    def secondary_metric_tag(self) -> str:
        """Prefix for the secondary decoding's metric keys."""
        return 'alt'

    def _secondary_metrics(self, stage: str):
        return getattr(self, self._secondary_attr(stage), None)

    def _setup_named_eval_dataloaders(self, config: Optional[Union[DictConfig, Dict]], stage: str) -> None:
        """Build one dataloader per named evaluation dataset.

        Replaces `nemo.utils.model_utils.resolve_{validation,test}_dataloaders`. That also turns a
        list of manifests into one dataloader each, but it names them itself and NeMo then logs
        `<stem>_val/cp_wer` where this model logs `val/<stem>/cp_wer`; and it has no mapping form,
        which is the only way to name a dataset or to pool several manifests under one name.
        """
        stage_attrs = EVAL_STAGES[stage]
        setattr(self, stage_attrs['dataloaders'], None)
        setattr(self, stage_attrs['names'], None)
        setattr(self, stage_attrs['metrics'], None)
        setattr(self, stage_attrs['nemo_names'], None)
        setattr(self, stage_attrs['dl_idx'], 0)

        if config is None or config.get('manifest_filepath') is None:
            return

        # Preserved as written, before the per-dataset copies below narrow it to one manifest.
        self._update_dataset_config(dataset_name=stage, config=config)

        pairs = named_manifests(config['manifest_filepath'])
        setup = self.setup_validation_data if stage == 'validation' else self.setup_test_data

        dataloaders = []
        try:
            # `_update_dataset_config` is a no-op in this mode, which is what stops the per-dataset
            # copies from replacing the config just preserved.
            self._multi_dataset_mode = True
            for _, manifest in pairs:
                dataset_config = copy.deepcopy(config)
                dataset_config['manifest_filepath'] = manifest
                setup(dataset_config)
                dataloaders.append(getattr(self, stage_attrs['dataloaders']))
        finally:
            self._multi_dataset_mode = False

        names = [name for name, _ in pairs]
        logging.info("Evaluating %s on %d dataset(s): %s", stage, len(names), ', '.join(names))
        setattr(self, stage_attrs['dataloaders'], dataloaders)
        setattr(self, stage_attrs['names'], names)
        # NeMo's own convention for the same thing, in case anything reads it: its display names
        # end in an underscore, since it uses them as a metric-key prefix.
        setattr(self, stage_attrs['nemo_names'], [f'{name}_' for name in names])

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        """Build the validation loader with one item per (session, speaker).

        `val=True` makes evaluation exhaustive and deterministic, which is what the
        multi-talker metrics need; training instead samples one target speaker per session.
        """
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        self._update_dataset_config(dataset_name='validation', config=val_data_config)
        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config, val=True)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        """Build the test loader with one item per (session, speaker). See `setup_validation_data`."""
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        self._update_dataset_config(dataset_name='test', config=test_data_config)
        self._test_dl = self._setup_dataloader_from_config(config=test_data_config, val=True)

    def setup_multiple_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        """See `_setup_named_eval_dataloaders`."""
        self._setup_named_eval_dataloaders(val_data_config, 'validation')

    def setup_multiple_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        """See `_setup_named_eval_dataloaders`."""
        self._setup_named_eval_dataloaders(test_data_config, 'test')

    # Parameters that get their own learning rate, as (config key, name predicate, description).
    # Order matters only in that a parameter joins the first group that claims it.
    LR_GROUPS = (
        ('fddt_lr_multiplier', lambda name: '.fddts.' in name, 'FDDT'),
        # Claimed before the projections below, which is why it is listed first.
        (
            'sink_gate_lr_multiplier',
            lambda name: '.sink_gate' in name,
            'attention-sink gate',
        ),
        (
            'sink_lr_multiplier',
            lambda name: any(part in name for part in ('.global_q.', '.global_k.', '.global_v.')),
            'attention-sink projections',
        ),
        # The CTC head, for a rate between frozen and the encoder's. `tdt_joint.*` needs no
        # mention: those are these same tensors, and `named_parameters` yields them once.
        ('head_lr_multiplier', lambda name: name.startswith('decoder.'), 'CTC head'),
        # The multi-talker model's two new blocks. Both start from an initialization that makes
        # them close to inert, so unlike the encoder they have little to preserve and can be
        # moved faster than a pretrained stack tolerates.
        #
        # The frame gate gets its own group, claimed before the separator's, for the same reason
        # the attention-sink gate does: it is a handful of tensors that have to travel a long way
        # before they do anything, and at the base rate they do not get there. The measured
        # failure is on record -- 9151 steps left the gate at its initialization.
        # `gate_temperature` is deliberately not here. It is a calibration scalar, and AdamW moves
        # a parameter by about the learning rate per step whatever its gradient, so at the rate
        # the projections need it would run to a saturated gate -- where the sigmoid gradient
        # vanishes and it cannot come back. It trains with the rest of the separator instead.
        (
            'separator_gate_lr_multiplier',
            lambda name: name.startswith(('separator.gate_query', 'separator.gate_key', 'separator.gate_bias')),
            'separator frame gate',
        ),
        ('separator_lr_multiplier', lambda name: name.startswith('separator.'), 'speaker separator'),
        (
            'postnet_lr_multiplier',
            lambda name: name.startswith('speaker_postnet.') or name.startswith('postnet_gate'),
            'speaker post-net',
        ),
        # The shuffle head's speaker branch: the only block in that model with no pretrained
        # function to preserve, so the only one that needs a rate the rest would not tolerate.
        # `speaker_context` is the depthwise temporal conv in front of the MLP -- same from-scratch
        # branch, same rate; only the shuffle head defines it, so this is inert for the others.
        (
            'speaker_head_lr_multiplier',
            lambda name: name.startswith(('speaker_head.', 'speaker_context.')),
            'speaker head',
        ),
        # Pandemonium's two new encoder blocks, both from scratch: cross-speaker attention and
        # the STNO-mask refiner. Live under `encoder.cross_speaker.*` / `encoder.mask_refiner.*`.
        (
            'cross_speaker_lr_multiplier',
            lambda name: '.cross_speaker.' in name,
            'cross-speaker attention',
        ),
        (
            'mask_refiner_lr_multiplier',
            lambda name: '.mask_refiner.' in name,
            'STNO-mask refiner',
        ),
    )

    def setup_optimizer_param_groups(self):
        """Give some parameters their own learning rate; see `LR_GROUPS`.

        With every multiplier at 1.0 this defers to the base class so a *single* group is
        emitted, which keeps `exp_manager.resume_if_exists` able to load optimizer state from
        runs that predate this. (The FDDT multiplier was silently a no-op in the original NeMo
        code, so 1.0 also reproduces the released checkpoints.)
        """
        active = [
            (key, predicate, label, float(self.cfg.get(key, 1.0)))
            for key, predicate, label in self.LR_GROUPS
            if float(self.cfg.get(key, 1.0)) != 1.0
        ]
        if not active:
            super().setup_optimizer_param_groups()
            return

        base_lr = self.cfg.optim.lr
        groups, claimed = [], set()
        for _, predicate, label, multiplier in active:
            params = [p for name, p in self.named_parameters() if predicate(name) and name not in claimed]
            claimed.update(name for name, _ in self.named_parameters() if predicate(name))
            logging.info("Parameter group '%s': %d tensors at lr=%s", label, len(params), base_lr * multiplier)
            # A multiplier can be set for a block this model does not have -- `LSTMSeparator` has
            # no frame-gate projections under the names `separator_gate_lr_multiplier` looks for,
            # for one. Skip rather than hand the optimizer an empty group.
            if params:
                groups.append({'params': params, 'lr': base_lr * multiplier})

        rest = [p for name, p in self.named_parameters() if name not in claimed]
        logging.info("Parameter group 'base': %d tensors at lr=%s", len(rest), base_lr)

        # The unscaled group goes first, so `param_groups[0]['lr']` -- which the training step
        # logs as `learning_rate` -- keeps meaning the base rate.
        self._optimizer_param_groups = [{'params': rest}, *groups]

    """ Transcription related methods """

    def transcribe(self, *args, **kwargs):
        """Disabled: NeMo's transcription path cannot supply the STNO mask.

        Without a mask the encoder runs unconditioned and returns a fluent transcript of
        whoever is loudest, which looks correct but is not target-speaker output. Use
        `infer.py` (RTTM + audio directory -> STM) or `transcribe_stno` instead.
        """
        raise NotImplementedError(
            f"{type(self).__name__} requires diarization conditioning, which transcribe() cannot "
            "provide. Use infer.py for RTTM-driven inference, or transcribe_stno() directly."
        )

    @torch.inference_mode()
    def transcribe_stno(
        self,
        stno_mask: torch.Tensor,
        input_signal: Optional[torch.Tensor] = None,
        input_signal_length: Optional[torch.Tensor] = None,
        processed_signal: Optional[torch.Tensor] = None,
        processed_signal_length: Optional[torch.Tensor] = None,
        return_hypotheses: bool = True,
    ) -> List[Hypothesis]:
        """Decode a batch of target speakers given their STNO masks.

        Either `input_signal`/`input_signal_length` or `processed_signal`/
        `processed_signal_length` must be given. Callers decoding several target speakers of
        the same session should pass `processed_signal`, since the features do not depend on
        the target speaker and recomputing them per speaker is wasteful.

        Args:
            stno_mask: (B, 4, T_enc) mask, one row group per target speaker.
            return_hypotheses: Return `Hypothesis` objects, which carry the word timestamps
                that `infer.py` needs. With `False`, only the text is returned.

        Returns:
            One hypothesis per batch element, in input order.
        """
        was_training = self.training
        self.eval()
        try:
            encoded, encoded_len = self.forward(
                input_signal=input_signal,
                input_signal_length=input_signal_length,
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
                stno_mask=stno_mask,
            )
            return self._decode_hypotheses(encoded, encoded_len, return_hypotheses=return_hypotheses)
        finally:
            if was_training:
                self.train()
