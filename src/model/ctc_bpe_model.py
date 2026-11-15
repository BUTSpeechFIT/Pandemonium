"""Sub-word (BPE) variant of the diarization-conditioned CTC model.

Inherits `(EncDecCTCModelSTNO, ASRBPEMixin)` rather than NeMo's `EncDecCTCModelBPE`, for the
same reason as the transducer variant: `EncDecCTCModelBPE.__init__` hard-requires a
`cfg.tokenizer` section, while this project injects the tokenizer object taken from the
pretrained Parakeet, and a diamond base would put the tokenizer-requiring constructor first in
the C3 order. See `src/model/asr_bpe_model.py`.

The head is sized here, before the modules are built: `cfg.decoder.num_classes` is the
tokenizer's vocabulary, and `ConvASRDecoder` adds the CTC blank itself
(`num_classes_with_blank`). Starting from `nvidia/parakeet-tdt-0.6b-v3` that is 8192 + 1.
"""

import copy
from typing import Dict, Optional

import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.parts.mixins import ASRBPEMixin
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCBPEDecoding, CTCBPEDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecoding
from nemo.collections.common.parts.optional_cuda_graphs import WithOptionalCudaGraphs
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging, model_utils

from src.data.dataloader import get_stno_dataloader
from src.metrics.meeteval_mt_wer import MeetevalMTWER
from src.model.ctc_model import EncDecCTCModelSTNO

__all__ = ['EncDecCTCBPEModelSTNO']


class EncDecCTCBPEModelSTNO(EncDecCTCModelSTNO, ASRBPEMixin):
    """Diarization-conditioned CTC model with sub-word tokenization."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None, tokenizer=None):
        """
        Args:
            cfg: The `model` section of the config.
            trainer: The Lightning trainer, when training.
            tokenizer: A pre-built tokenizer, normally taken from the pretrained Parakeet
                checkpoint. When omitted, `cfg.tokenizer` is used to build one.
        """
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        if 'tokenizer' not in cfg and tokenizer is None:
            raise ValueError("`cfg` must have a `tokenizer` config, or a `tokenizer` must be passed.")

        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)

        if tokenizer is None:
            self._setup_tokenizer(cfg.tokenizer)
        else:
            self.tokenizer = tokenizer

        vocabulary = self.tokenizer.tokenizer.get_vocab()

        # Size the head to the tokenizer before the modules are instantiated.
        with open_dict(cfg):
            cfg.labels = ListConfig(list(vocabulary))

        with open_dict(cfg.decoder):
            cfg.decoder.vocabulary = ListConfig(list(vocabulary))
            cfg.decoder.num_classes = len(vocabulary)

        super().__init__(cfg=cfg, trainer=trainer)

        decoding_cfg = self.cfg.get('decoding', None)
        if decoding_cfg is None:
            decoding_cfg = OmegaConf.structured(CTCBPEDecodingConfig)
            with open_dict(self.cfg):
                self.cfg.decoding = decoding_cfg
        self.decoding = CTCBPEDecoding(self.cfg.decoding, tokenizer=self.tokenizer)

        # Samples of audio per encoder frame: 16000 * 0.01 * 8 = 1280, i.e. 12.5 Hz / 80 ms.
        # This sets both the STNO mask frame rate and the scale for decoded word timestamps.
        self.audio_downsampling_factor = int(
            self.cfg.sample_rate * self.cfg.preprocessor.window_stride * self.cfg.encoder.subsampling_factor
        )
        self.embed_duration = self.audio_downsampling_factor / self.cfg.sample_rate  # seconds

        # NeMo's `EncDecCTCModel.__init__` has no `compute_eval_loss`; the transducer one does,
        # and the shared evaluation path reads it. Defaults to on here, unlike the transducer,
        # because the CTC loss has no vocabulary-sized lattice to blow up on long sessions.
        self.compute_eval_loss = self._cfg.get('compute_eval_loss', True)

        self.wer = WER(
            decoding=self.decoding,
            use_cer=self._cfg.get('use_cer', False),
            dist_sync_on_step=True,
            log_prediction=self._cfg.get('log_prediction', True),
        )

        self._setup_tdt_eval()
        self.maybe_freeze_decoder()
        self._setup_speaker_count_head()

        # The gate is on a schedule (`on_train_batch_start`), so gradient must not fight it.
        if int(self._cfg.get('sink_gate_warmup_steps', 0) or 0) > 0:
            for layer in self.encoder.layers:
                gate = getattr(getattr(layer, 'self_attn', None), 'sink_gate', None)
                if gate is not None:
                    gate.requires_grad = False
            logging.info("Sink gate annealed over %d steps; gate parameters frozen",
                         int(self._cfg.sink_gate_warmup_steps))

        # The multi-talker metrics are not built here: there is one per evaluation dataloader and
        # the dataloaders are set up by `ModelPT.__init__` above, before `self.decoding` exists.
        # `STNOModelMixin._build_eval_metrics` builds them at each evaluation epoch start.

    def _setup_tdt_eval(self) -> None:
        """Reassemble the transducer, for validation-time decoding only.

        Borrows the head's three modules into an `RNNTJoint` -- the same objects, not copies, so
        they stay in step while the head trains -- and adds the prediction network routed here by
        `adapt_pretrained_state_dict`. On the same weights this decoding scored 0.360 cpWER
        against CTC's 0.430, which is why it is tracked per epoch.

        Nothing here trains: the prediction network is frozen and in eval mode, which also keeps
        DDP from raising over parameters no training forward touches.
        """
        self.tdt_decoding = None
        if not self._cfg.get('eval_tdt', False):
            return

        cfg = self._cfg.tdt_eval
        vocabulary = self.tokenizer.tokenizer.get_vocab()

        prediction_cfg = copy.deepcopy(cfg.prediction)
        with open_dict(prediction_cfg):
            prediction_cfg.vocab_size = len(vocabulary)
        self.tdt_prediction = self.from_config_dict(prediction_cfg)

        joint_cfg = copy.deepcopy(cfg.joint)
        with open_dict(joint_cfg):
            joint_cfg.num_classes = len(vocabulary)
            joint_cfg.vocabulary = ListConfig(list(vocabulary))
            joint_cfg.jointnet.encoder_hidden = self._cfg.model_defaults.enc_hidden
            joint_cfg.jointnet.pred_hidden = self._cfg.model_defaults.pred_hidden
        self.tdt_joint = self.from_config_dict(joint_cfg)

        # The tie: throw away the modules `RNNTJoint` built and point at the head's, so there is
        # exactly one copy of these weights in the model.
        self.tdt_joint.enc = self.decoder.enc
        self.tdt_joint.pred = self.decoder.pred
        self.tdt_joint.joint_net = self.decoder.joint_net

        for parameter in self.tdt_prediction.parameters():
            parameter.requires_grad = False
        self.tdt_prediction.eval()

        self.tdt_decoding = RNNTBPEDecoding(
            decoding_cfg=cfg.decoding,
            decoder=self.tdt_prediction,
            joint=self.tdt_joint,
            tokenizer=self.tokenizer,
        )
        logging.info("Transducer decoding enabled for validation; scores logged under val/tdt_cp_*")

    def _setup_speaker_count_head(self) -> None:
        """Classifier over speaker counts, reading the sink tokens. See `_speaker_count_loss`."""
        self.speaker_count_head = None
        if float(self._cfg.get('speaker_count_loss_weight', 0.0) or 0.0) <= 0.0:
            return
        if not getattr(self.encoder, 'learnable_sink_tokens', False):
            logging.warning("speaker_count_loss_weight is set but there are no sink tokens to read")
            return
        max_speakers = int(self._cfg.get('max_speakers', 8))
        self.speaker_count_head = torch.nn.Linear(self._cfg.model_defaults.enc_hidden, max_speakers + 1)
        logging.info("Speaker-count objective on the sinks: 0-%d speakers, weight %s",
                     max_speakers, self._cfg.speaker_count_loss_weight)

    @property
    def secondary_metric_tag(self) -> str:
        return 'tdt'

    def _build_secondary_metric(self) -> Optional[MeetevalMTWER]:
        # `eval_tdt_metric: false` keeps the transducer stack -- so checkpoints stay compatible,
        # both in their state dict and in the optimizer's parameter groups -- while skipping the
        # decode, which is what actually costs: greedy TDT decoding degenerates on long sessions
        # (16 s for AMI dev one epoch, still running after 17 minutes the next).
        if getattr(self, 'tdt_decoding', None) is None or not self._cfg.get('eval_tdt_metric', True):
            return None
        meeteval_cfg = self._cfg.get('meeteval', {}) or {}
        return MeetevalMTWER(
            decoding=self.tdt_decoding,
            dist_sync_on_step=False,
            log_prediction=self._cfg.get('log_prediction', False),
            embed_duration=self.embed_duration,
            output_per_word_timestamps=meeteval_cfg.get('output_per_word_timestamps', True),
            tcp_collar=meeteval_cfg.get('tcp_collar', 5),
            compute_tcpwer=meeteval_cfg.get('compute_tcpwer', True),
            text_norm_type=self._cfg.get('text_norm', 'whisper_nsf'),
            session_scope=meeteval_cfg.get('session_scope', 'recording'),
        )

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        # Greedy TDT decoding captures CUDA graphs, which assume fixed shapes. The base class
        # disables them for `decoding.decoding`; this stack lives elsewhere.
        if getattr(self, 'tdt_decoding', None) is not None:
            WithOptionalCudaGraphs.disable_cuda_graphs_recursive(self, attribute_path="tdt_decoding.decoding")

    def _build_meeteval_metric(self) -> MeetevalMTWER:
        meeteval_cfg = self._cfg.get('meeteval', {}) or {}
        return MeetevalMTWER(
            decoding=self.decoding,
            dist_sync_on_step=False,
            log_prediction=self._cfg.get('log_prediction', False),
            embed_duration=self.embed_duration,
            output_per_word_timestamps=meeteval_cfg.get('output_per_word_timestamps', True),
            tcp_collar=meeteval_cfg.get('tcp_collar', 5),
            compute_tcpwer=meeteval_cfg.get('compute_tcpwer', True),
            text_norm_type=self._cfg.get('text_norm', 'whisper_nsf'),
            session_scope=meeteval_cfg.get('session_scope', 'recording'),
        )

    def change_decoding_strategy(self, decoding_cfg: DictConfig, verbose: bool = True):
        """Rebuild the decoding object, and the WER metric that holds a reference to it.

        `infer.py` uses this to turn on `compute_timestamps`, which the STM output needs.

        Args:
            decoding_cfg: New decoding config; `None` re-applies the current one.
            verbose: Log the resulting config.
        """
        if decoding_cfg is None:
            logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
            decoding_cfg = self.cfg.decoding

        # Fill in every hyperparameter the decoding config declares.
        decoding_cls = OmegaConf.structured(CTCBPEDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

        self.decoding = CTCBPEDecoding(decoding_cfg=decoding_cfg, tokenizer=self.tokenizer)

        self.wer = WER(
            decoding=self.decoding,
            use_cer=self.wer.use_cer,
            log_prediction=self.wer.log_prediction,
            dist_sync_on_step=True,
        )

        # The multi-talker metrics decode through `self.decoding` too, but they are rebuilt at
        # every evaluation epoch start, so they pick the new strategy up on their own.

        with open_dict(self.cfg.decoding):
            self.cfg.decoding = decoding_cfg

        if verbose:
            logging.info("Changed decoding strategy to \n%s", OmegaConf.to_yaml(self.cfg.decoding))

    def _setup_dataloader_from_config(self, config: Optional[Dict], val: bool = False):
        """Build a `(session, target speaker)` dataloader. See `src/data/dataloader.py`."""
        return get_stno_dataloader(
            config=config,
            tokenizer=self.tokenizer,
            audio_downsampling_factor=int(
                self.cfg.sample_rate * self.cfg.preprocessor.window_stride * self.cfg.encoder.subsampling_factor
            ),
            text_norm_type=self._cfg.get('text_norm', 'whisper_nsf'),
            val=val,
            # With `use_distributed_sampler: false` -- required by the dynamic batch sampler --
            # Lightning stops sharding the evaluation loaders too, so they shard themselves.
            lightning_injects_sampler=getattr(
                getattr(self._trainer, '_accelerator_connector', None), 'use_distributed_sampler', True
            ) if self._trainer is not None else True,
            multi_speaker=self.multi_speaker,
            num_speaker_slots=self.num_speaker_slots,
            slot_order=self.slot_order,
            sot=self.sot,
            sot_order=self.sot_order,
            sot_bos_id=self.sot_bos_id,
            sot_eos_id=self.sot_eos_id,
            sot_speaker_change_id=self.sot_speaker_change_id,
            group_speakers=self.group_speakers,
        )

    def _setup_transcribe_dataloader(self, config: Dict):
        raise NotImplementedError(
            "EncDecCTCBPEModelSTNO requires diarization conditioning. Use infer.py instead."
        )

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """No pretrained models are published under this class yet."""
        return []
