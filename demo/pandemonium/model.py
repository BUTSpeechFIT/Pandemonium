"""Load Pandemonium from the Hub and transcribe one session given per-speaker activity."""

import os
import tarfile
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import sentencepiece as spm
import torch
import yaml
from huggingface_hub import hf_hub_download

from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor

from .encoder import PandemoniumEncoder
from .modules import JointCTCDecoder, soft_stno_masks

SAMPLE_RATE = 16_000
FRAME_SHIFT = 0.08  # one encoder frame per 80 ms (10 ms mel x 8 subsampling)

MODEL_REPO = os.environ.get("PANDEMONIUM_REPO", "BUT-FIT/Pandemonium")
CHECKPOINT = os.environ.get("PANDEMONIUM_CKPT", "cp_wer=0.1597-epoch=16.ckpt")
# Pandemonium shares Parakeet-TDT-0.6b-v3's tokenizer; it is read out of that .nemo if the model
# repo does not carry its own copy.
TOKENIZER_FALLBACK_REPO = "nvidia/parakeet-tdt-0.6b-v3"

# Encoder constructor arguments that only mattered for training (or are fixed by the vendored
# modules); everything else in the stored config goes to NeMo's ConformerEncoder unchanged.
_TRAIN_ONLY_ENCODER_KEYS = {
    "_target_", "cross_speaker_dropout", "refine_gate_init", "sink_gate_init", "init_global_attn_from_local",
    "activation_checkpointing", "fddt_is_diagonal", "fddt_bias_only", "fddt_use_silence",
    "fddt_use_target", "fddt_use_non_target", "fddt_use_overlap",
}


@dataclass
class Word:
    text: str
    start: float
    end: float


def _tokenizer_path() -> str:
    try:
        return hf_hub_download(MODEL_REPO, "tokenizer.model")
    except Exception:
        nemo_file = hf_hub_download(TOKENIZER_FALLBACK_REPO, "parakeet-tdt-0.6b-v3.nemo")
        out_dir = os.path.join(os.path.dirname(nemo_file), "_pandemonium_tokenizer")
        out = os.path.join(out_dir, "tokenizer.model")
        if not os.path.exists(out):
            os.makedirs(out_dir, exist_ok=True)
            with tarfile.open(nemo_file) as tar:
                member = next(m for m in tar.getmembers() if m.name.endswith("tokenizer.model"))
                with tar.extractfile(member) as src, open(out, "wb") as dst:
                    dst.write(src.read())
        return out


class Pandemonium(torch.nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        pre = {k: v for k, v in cfg["preprocessor"].items() if k != "_target_"}
        self.preprocessor = AudioToMelSpectrogramPreprocessor(**pre)

        enc = {k: v for k, v in cfg["encoder"].items() if k not in _TRAIN_ONLY_ENCODER_KEYS}
        # 1 = chunk the subsampling convolutions only when the input needs it (an hour does).
        enc["subsampling_conv_chunking_factor"] = 1
        self.encoder = PandemoniumEncoder(**enc)

        dec = {k: v for k, v in cfg["decoder"].items() if k not in ("_target_", "vocabulary")}
        self.decoder = JointCTCDecoder(**dec)
        self.blank_id = self.decoder.num_classes_with_blank - 1

    @classmethod
    def from_pretrained(cls, device: str = "cuda", checkpoint_path: Optional[str] = None) -> "Pandemonium":
        path = checkpoint_path or hf_hub_download(MODEL_REPO, CHECKPOINT)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        hp = ckpt["hyper_parameters"]
        cfg = hp["cfg"] if "cfg" in hp else hp
        cfg = yaml.safe_load(yaml.safe_dump(_to_plain(cfg)))

        model = cls(cfg)
        state = {k: v for k, v in ckpt["state_dict"].items() if k.split(".", 1)[0] in ("preprocessor", "encoder", "decoder")}
        model.load_state_dict(state, strict=True)
        model.tokenizer = spm.SentencePieceProcessor(model_file=_tokenizer_path())
        return model.eval().to(device)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    def num_frames(num_samples: int) -> int:
        """Encoder frames for `num_samples` samples (rounded up, as in training)."""
        return -(-num_samples // int(SAMPLE_RATE * FRAME_SHIFT))

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, activity: np.ndarray, dtype: torch.dtype = torch.bfloat16,
                   row_chunk: Optional[int] = None) -> List[List[Word]]:
        """Transcribe every speaker of one session.

        Args:
            audio: mono 16 kHz waveform.
            activity: `(G, T_enc)` per-speaker activity at 80 ms, soft (probabilities) or hard (0/1).
            dtype: compute precision on GPU.
            row_chunk: speaker rows per Conformer-layer call (`None` = sized to the audio length).

        Returns:
            One list of time-stamped words per speaker row of `activity`.
        """
        device = self.device
        signal = torch.as_tensor(audio, dtype=torch.float32, device=device)[None]
        signal_len = torch.tensor([signal.shape[1]], device=device)
        act = torch.as_tensor(activity, dtype=torch.float32, device=device).clamp(0, 1)
        g = act.shape[0]
        if g == 1:
            # A lone speaker makes cross-speaker attention attend to itself only, which training
            # never produced and which silences the stream (100% WER, empty output). A never-active
            # partner row restores the trained regime: its mask (from the formula below) hears the
            # real speaker as non-target, while the real speaker's own mask is unchanged. 12.9% vs
            # 13.1% on two-speaker windows (mixer6 30 s). Its output is discarded below.
            act = torch.cat([act, torch.zeros_like(act)])
        stno = soft_stno_masks(act[None], torch.ones(1, act.shape[0], dtype=torch.bool, device=device))[0]

        features, feature_len = self.preprocessor(input_signal=signal, length=signal_len)
        use_amp = device.type == "cuda" and dtype != torch.float32
        with torch.autocast(device.type, dtype=dtype, enabled=use_amp):
            encoded, _ = self.encoder.encode(features, feature_len, stno, row_chunk=row_chunk)
            # One speaker at a time: the full (G, T, 8193) log-softmax of an hour is ~12 GB, and greedy
            # decoding only needs its argmax.
            best = [self.decoder(encoded[i:i + 1]).argmax(-1)[0].cpu().numpy() for i in range(g)]  # real rows only
        return [self._greedy_words(row) for row in best]

    def _greedy_words(self, best: np.ndarray) -> List[Word]:
        """Collapse a greedy CTC path into words with 80 ms timestamps."""
        words: List[Word] = []
        pieces, first, last = [], 0, 0
        prev = self.blank_id

        def flush():
            if pieces:
                text = self.tokenizer.decode_ids(pieces).strip()
                if text:
                    words.append(Word(text, first * FRAME_SHIFT, (last + 1) * FRAME_SHIFT))

        for frame, token in enumerate(best.tolist()):
            if token != self.blank_id and token != prev:
                if self.tokenizer.id_to_piece(token).startswith("▁"):
                    flush()
                    pieces, first = [], frame
                if not pieces:
                    first = frame
                pieces.append(token)
                last = frame
            prev = token
        flush()
        return words


def _to_plain(cfg):
    """OmegaConf / dict-like -> plain python containers."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass
    return dict(cfg)
