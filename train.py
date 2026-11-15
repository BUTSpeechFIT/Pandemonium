"""Train the diarization-conditioned Parakeet target-speaker ASR model.

    python train.py +exp_dir="exps/"

The tokenizer and the initial weights come from `init_from_pretrained` (a pretrained Parakeet).
Which modules that covers depends on the head the config selects: with `head: tdt` it is the
encoder, prediction network and joint; with `head: ctc` (conf/dicop_ctc.yaml) the joint becomes
the CTC projection and the prediction network is dropped. Either way only the FDDT blocks start
from scratch.

The config file is chosen by `$DICOP_CONFIG_NAME`, not Hydra's `--config-name`:

    DICOP_CONFIG_NAME=dicop_ctc python train.py ...

`--config-name` cannot work here: `hydra_runner`'s argparse takes overrides as a `nargs='*'`
positional, which it will not match across a preceding optional, so the flag must come last --
but under DDP Lightning relaunches this script per rank with its own overrides appended after
it, and every child exits 2 with "unrecognized arguments".

Validation reports multi-talker cpWER/tcpWER and writes `ref.stm` / `hyp.stm` under the
experiment's log directory. For inference on new audio, see `infer.py`.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import lightning.pytorch as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models import ASRModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

from utils.nemo import allow_external_nemo_targets, load_ptl_ckpt

allow_external_nemo_targets()

from src.model.asr_bpe_model import EncDecRNNTBPEModelSTNO
from src.model.cross_speaker_ctc_bpe_model import EncDecCrossSpeakerCTCBPEModelSTNO
from src.model.ctc_bpe_model import EncDecCTCBPEModelSTNO

# `head` in the config picks the architecture. `tdt`/`rnnt` and `ctc` are target-speaker models
# and differ only below the encoder; `pandemonium` (conf/pandemonium.yaml) is the CTC target-speaker
# model with a session's speakers batched together so the encoder can attend across them and refine
# the STNO mask.
HEADS = {
    'tdt': EncDecRNNTBPEModelSTNO,
    'rnnt': EncDecRNNTBPEModelSTNO,
    'ctc': EncDecCTCBPEModelSTNO,
    'pandemonium': EncDecCrossSpeakerCTCBPEModelSTNO,
}


def maybe_load_pretrained_model(model_name: str | None) -> ASRModel | None:
    if not model_name:
        return None
    return ASRModel.from_pretrained(model_name=model_name, map_location="cpu")


def model_class(cfg):
    head = str(cfg.get('head', 'tdt')).lower()
    if head not in HEADS:
        raise ValueError(f"Unknown head '{head}'; expected one of {sorted(HEADS)}.")
    return HEADS[head]


# See the module docstring: an environment variable rather than `--config-name`, because the
# latter breaks every DDP rank but the first.
CONFIG_NAME = os.environ.get("DICOP_CONFIG_NAME", "dicop")


@hydra_runner(config_path="conf", config_name=CONFIG_NAME)
def main(cfg):
    logging.info("Hydra config:\n%s", OmegaConf.to_yaml(cfg))

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    pretrained_model = maybe_load_pretrained_model(cfg.get("init_from_pretrained"))

    exp_manager(trainer, cfg.get("exp_manager", None))
    model_cls = model_class(cfg)
    logging.info("Building %s (head=%s)", model_cls.__name__, cfg.get('head', 'tdt'))
    asr_model = model_cls(
        cfg=cfg.model,
        trainer=trainer,
        tokenizer=pretrained_model.tokenizer if pretrained_model is not None else None,
    )

    if pretrained_model is not None:
        # Non-strict on purpose; which mismatches are expected depends on the head. Anything
        # outside what the class declares means a topology mismatch.
        state_dict = asr_model.adapt_pretrained_state_dict(pretrained_model.state_dict())
        missing, unexpected = asr_model.load_state_dict(state_dict, strict=False)
        expected_missing = asr_model.PRETRAINED_EXPECTED_MISSING
        expected_unexpected = asr_model.PRETRAINED_EXPECTED_UNEXPECTED
        surprising_missing = [k for k in missing if not any(p in k for p in expected_missing)]
        surprising_unexpected = [k for k in unexpected if not any(p in k for p in expected_unexpected)]
        logging.info(
            "Warm start from %s: %d missing keys (%d unexpectedly so), %d unused keys in the "
            "checkpoint (%d unexpectedly so)",
            cfg.get("init_from_pretrained"),
            len(missing),
            len(surprising_missing),
            len(unexpected),
            len(surprising_unexpected),
        )
        if surprising_missing or surprising_unexpected:
            logging.warning(
                "Warm start from %s did not match cleanly. Unexplained missing keys: %s. "
                "Unexplained unused keys: %s",
                cfg.get("init_from_pretrained"),
                surprising_missing,
                surprising_unexpected,
            )
        del pretrained_model

    if cfg.get("init_from_ptl_ckpt"):
        load_ptl_ckpt(asr_model, cfg.init_from_ptl_ckpt, strict=True)

    if cfg.get("evaluate_at_start", False):
        trainer.validate(asr_model)

    trainer.fit(asr_model)

    if cfg.model.get("test_ds") is not None and cfg.model.test_ds.get("manifest_filepath") is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
