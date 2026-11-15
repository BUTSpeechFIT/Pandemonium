# Pandemonium

**Who said what, in a whole meeting, in one pass.** Pandemonium is a long-form, diarization-conditioned multi-talker
ASR model. Given a recording and a diarization of it, it returns a time-stamped transcript for every speaker, and it
decodes the speakers of a conversation together, so each speaker's transcript is informed by the others.

It is described in **Pandemonium: Longform Target-Speaker Speech Recognition with Cross-Speaker Interactions**
([paper](paper/Pandemonium_ICASSP2027.pdf), submitted to ICASSP 2027).

- **Try it:** [pcspeech-demo.fit.vut.cz/pandemonium](https://pcspeech-demo.fit.vut.cz/pandemonium)
- **Model:** [BUT-FIT/Pandemonium](https://huggingface.co/BUT-FIT/Pandemonium)
- **Backbone without cross-speaker interaction:** [BUT-FIT/DiCoP-v2](https://huggingface.co/BUT-FIT/DiCoP-v2) (TDT and CTC)
- **Pseudo-labeled training data (recipe):** [BUT-FIT/YODAS-2-MT](https://huggingface.co/datasets/BUT-FIT/YODAS-2-MT)

> *Pandemonium*: the collective noun for a flock of parakeets, or chaos.

## How it works

Pandemonium builds on diarization-conditioned target-speaker ASR ([DiCoW](https://github.com/BUTSpeechFIT/DiCoW)):
for each speaker the diarizer finds, every frame is labeled silence, target, other speaker or overlap (an STNO mask),
and each encoder layer applies a small learned transform per label. On top of that:

1. **Long-form backbone (DiCoP).** The conditioning moves from Whisper's 30 s windows into
   [Parakeet-TDT-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), whose local attention takes a whole
   meeting at once.
2. **Cross-speaker attention.** A conversation's speakers are encoded as one batch, and at four encoder layers each
   frame attends across the speakers.
3. **STNO mask refiner.** At three layers the model re-estimates the diarization mask from its own representations,
   which repairs diarization errors.

Training goes synthetic mixtures → pseudo-labeled YODAS2 conversations → real meetings (NOTSOFAR-1, AMI).

## Results

cpWER [%], whole recordings, greedy decoding (Table 1 of the paper):

| system | NSF | AMI | ICSI | Mx6 | SBC | DiPCo | Avg |
|---|---|---|---|---|---|---|---|
| *oracle diarization* | | | | | | | |
| DiCoW v3.3 | 15.5 | 14.5 | 14.1 | 11.4 | 15.0 | 27.1 | 16.3 |
| DiCoP | 17.3 | 14.5 | 14.4 | 11.9 | 16.0 | 26.1 | 16.7 |
| **Pandemonium** | **14.1** | **13.4** | **13.3** | **11.1** | **13.8** | **22.4** | **14.7** |
| *pyannoteAI Precision-3 diarization* | | | | | | | |
| DiCoW v3.3 | 23.0 | 17.3 | 18.6 | 11.5 | 19.3 | 30.0 | 20.0 |
| DiCoP | 23.2 | 16.9 | 18.0 | 12.1 | 19.4 | 28.9 | 19.8 |
| **Pandemonium** | **22.4** | **16.8** | 18.1 | 11.7 | **18.6** | **27.4** | **19.2** |

NSF: NOTSOFAR-1; Mx6: Mixer 6; SBC: Santa Barbara Corpus. With the released pipeline (Nemotron-3-Diarization +
Pandemonium) a meeting is transcribed at 426× real time on one GPU, diarization included. [`benchmark/`](benchmark)
compares it with 16 other systems, on-device and commercial, and reproduces every number on the model card.

## Released models

| model | what it is | use it for |
|---|---|---|
| [Pandemonium](https://huggingface.co/BUT-FIT/Pandemonium) | DiCoP CTC + cross-speaker attention + mask refiner, trained synthetic → pseudo → real | the best accuracy |
| [DiCoP v2, `tdt/`](https://huggingface.co/BUT-FIT/DiCoP-v2) | long-form target-speaker Parakeet, TDT head, SE-DiCoW training data | the paper's DiCoP baseline |
| [DiCoP v2, `ctc/`](https://huggingface.co/BUT-FIT/DiCoP-v2) | the same with a frame-synchronous CTC head | 1.5× faster decoding |
| [DiCoP v0.1](https://huggingface.co/BUT-FIT/DiCoP_v0.1) | an earlier DiCoP on Parakeet-TDT-0.6b-v2 (English) | |

All are CC-BY-4.0.

## Quick start

The inference package in [`demo/`](demo) runs Nemotron-3-Diarization and Pandemonium together; the models download
from the Hugging Face Hub on first use.

```bash
git clone --recurse-submodules https://github.com/BUTSpeechFIT/Pandemonium.git && cd Pandemonium
conda create -n pandemonium python=3.12 -y && conda activate pandemonium
pip install Cython packaging
pip install -r demo/requirements.txt
```

```python
# PYTHONPATH=demo python transcribe.py
import librosa
from pipeline import DemoPipeline, utterances

audio, _ = librosa.load("meeting.wav", sr=16000, mono=True)
pipe = DemoPipeline()
diarization = pipe.diarize(audio)            # Nemotron-3-Diarization, up to 8 speakers
words = pipe.transcribe(audio, diarization)  # Pandemonium: a list of words per speaker
for u in utterances(words):
    print(f"{u['start']:7.2f} {u['end']:7.2f}  speaker {u['speaker']}: {u['text']}")
```

A one-hour, 8-speaker recording needs about 12 GiB of GPU memory. [`demo/README.md`](demo/README.md) describes the
pipeline's settings and how to run the web demo. [`benchmark/systems/pandemonium/run.py`](benchmark/systems/pandemonium/run.py)
decodes whole evaluation sets, with Nemotron-3-Diarization, reference diarization or any RTTM.

## Repository

```
demo/              inference package (pandemonium/, pipeline.py) and the web demo
src/, conf/        research code: models, data loading, metrics; Hydra configs
train.py, infer.py training; target-speaker decoding from RTTMs or Lhotse cuts
scripts/           data conversion, batch inference and scoring
benchmark/         the comparison on the model card, reproducible end to end
data/YODAS-2-MT/   recipe for the pseudo-labeled YODAS2 data (submodule, from the Hugging Face Hub)
paper/             the paper
```

## Research code

### Setup

```bash
conda create -n dicop python=3.11 -y && conda activate dicop
pip install -r requirements.txt
python -c "from numba import cuda; print(cuda.is_available())"   # must print True for training
```

The TDT loss uses a numba CUDA kernel, so training needs a working numba CUDA setup (`conda install
nvidia::cuda-toolkit` if the check fails); inference does not.

### Data

Training and evaluation read [Lhotse](https://github.com/lhotse-speech/lhotse) CutSets (`.jsonl.gz`) whose
supervisions carry speaker and text. [BUTSpeechFIT/mt-asr-data-prep](https://github.com/BUTSpeechFIT/mt-asr-data-prep)
prepares AMI, NOTSOFAR-1, LibriSpeechMix, LibriMix and others; the pseudo-labeled YODAS2 set is built with the
recipe in [`data/YODAS-2-MT`](data/YODAS-2-MT).

### Configs

[Hydra](https://hydra.cc/) configs, as in [NeMo](https://github.com/NVIDIA-NeMo/NeMo). Keys marked `???` must be set,
in the YAML or on the command line.

| config | model | in the paper |
|---|---|---|
| [`pandemonium.yaml`](conf/pandemonium.yaml) | Pandemonium: CTC, speakers batched per session, cross-speaker attention, mask refiner | Pandemonium |
| [`dicop.yaml`](conf/dicop.yaml), [`dicop_ctc.yaml`](conf/dicop_ctc.yaml) | DiCoP base configs, TDT and CTC head | |
| [`dicop_l1_tdt_global.yaml`](conf/dicop_l1_tdt_global.yaml) | 30 s, global attention, TDT | Fig. 2, glob/TDT |
| [`dicop_l2_tdt_local.yaml`](conf/dicop_l2_tdt_local.yaml) | 30 s, local attention, TDT | Fig. 2, loc/TDT; Table 1 DiCoP; DiCoP v2 `tdt/` |
| [`dicop_l3_ctc_local.yaml`](conf/dicop_l3_ctc_local.yaml) | 30 s, local attention, CTC | Fig. 2, loc/CTC; DiCoP v2 `ctc/` |
| `dicop_l4`–`l7` | the same with 60 s, 120 s, 300 s and 600 s training chunks | longer chunks (not shown) |

The `dicop_l*` configs share their data recipe through [`dicop_ladder_data.yaml`](conf/dicop_ladder_data.yaml), and
each inherits from the previous one. Main sections of a config:

| key | what it does |
|---|---|
| `init_from_pretrained` | the Parakeet checkpoint to start from (also supplies the tokenizer) |
| `head` | `tdt`, `ctc` or `pandemonium` |
| `model.train_ds` / `validation_ds` / `test_ds` | manifests, batching and loader settings |
| `model.encoder` | architecture, including the `fddt_*` conditioning and, for Pandemonium, `cross_speaker_layers` / `refine_layers` |
| `model.optim` | optimizer and schedule |
| `trainer`, `exp_manager` | PyTorch Lightning settings; where checkpoints and logs go |

### Training

With your own manifests, on a base config:

```bash
python train.py --config-name dicop init_from_pretrained=nvidia/parakeet-tdt-0.6b-v3 \
    model.train_ds.manifest_filepath=[train_a.jsonl.gz,train_b.jsonl.gz] \
    model.validation_ds.manifest_filepath="{ami:ami_dev.jsonl.gz,nsf:nsf_dev.jsonl.gz}" \
    exp_manager.exp_dir=exps/
```

A list of training manifests is pooled into one set. A dictionary of validation manifests is scored per dataset
(`val/ami/cp_wer`, ...) and pooled (`val/cp_wer`, which checkpoints are selected on).

The paper's `dicop_l*` configs name their own manifests, in the layout mt-asr-data-prep writes under
`$MANIFEST_DIR`. Their test sets include cutsets that mt-asr-data-prep does not build, so defer them:

```bash
MANIFEST_DIR=/data/manifests python train.py --config-name dicop_l2_tdt_local \
    exp_manager.exp_dir=exps/ +model.test_ds.defer_setup=true
```

`++init_from_ptl_ckpt` continues from a checkpoint's weights, as each of Pandemonium's three stages does from the
previous one. Quote the path for Hydra if it contains `=`, as the released file names do:

```bash
python train.py --config-name pandemonium \
    model.train_ds.manifest_filepath=[train.jsonl.gz] model.validation_ds.manifest_filepath=dev.jsonl.gz \
    exp_manager.exp_dir=exps/ "++init_from_ptl_ckpt='Pandemonium/cp_wer=0.1597-epoch=16.ckpt'"
```

`scripts/run_train.sh` wraps training for the mt-asr-data-prep corpora.

### Inference (DiCoP)

```bash
python infer.py --rttm rttms/ --audio-dir audio/ --output hyp.stm \
    --checkpoint DiCoP-v2/tdt/cp_wer=0.1939-epoch=4.ckpt \
    --config conf/dicop.yaml -O init_from_pretrained=nvidia/parakeet-tdt-0.6b-v3
```

`--checkpoint` takes a local `.ckpt` or `.nemo`, or a Hub id; use `conf/dicop_ctc.yaml` for the CTC variant.
`--cuts cuts.jsonl.gz` takes a Lhotse CutSet instead of RTTMs and audio (an RTTM's session id must match its audio
file name). Useful options: `--stm-granularity word|segment`, `--batch-size` (speakers per forward pass),
`--chunk-seconds` (decode in windows), `-O KEY=VALUE` (any config override). Pandemonium decodes a session's
speakers together, so it runs through the `demo/` package above rather than `infer.py`.

### Scoring

cpWER and tcpWER with [MeetEval](https://github.com/fgnt/meeteval), after Whisper/CHiME-8 text normalization:

```bash
python scripts/manifest_to_rttm.py --manifest test.jsonl.gz --output rttms/
python scripts/manifest_to_stm.py  --manifest test.jsonl.gz --output ref.stm
meeteval-wer cpwer  -r ref.stm -h hyp.stm
meeteval-wer tcpwer -r ref.stm -h hyp.stm --collar 5
```

`scripts/run_inference.sh` and `scripts/run_scoring.sh` do this over a set of prepared corpora.

## Citation

```bibtex
@inproceedings{polok2027pandemonium,
  title     = {Pandemonium: Longform Target-Speaker Speech Recognition with Cross-Speaker Interactions},
  author    = {Polok, Alexander and Klement, Dominik and Medennikov, Ivan and Laurent, Antoine and
               P{\'a}lka, Petr and Cornell, Samuele and Watanabe, Shinji and Khudanpur, Sanjeev and
               Balam, Jagadeesh and {\v{C}}ernock{\'y}, Jan and Burget, Luk{\'a}{\v{s}} and Wiesner, Matthew},
  booktitle = {Submitted to ICASSP},
  year      = {2027}
}
```

## Acknowledgements

This work was supported by JSALT 2026 at Johns Hopkins University (NSF CCRI Grant No. 2120435, Google DeepMind, JHU
HLTCOE, JHU AI2AI, ACL), by the Ministry of Education, Youth and Sports of the Czech Republic through the OP JAK
project "Linguistics, Artificial Intelligence and Language and Speech Technologies: from Research to Applications"
(CZ.02.01.01/00/23_020/0008518), and by the Brno Ph.D. Talent Scholarship Programme. Computational resources were
provided by e-INFRA CZ (ID: 90254) and by IDRIS under GENCI allocations AD011012527, A0201012177 and AD011016176.

## License

Code: [Apache-2.0](LICENSE). Models and the dataset recipe: CC-BY-4.0.

## Contact

- Alexander Polok, [ipoloka@fit.vut.cz](mailto:ipoloka@fit.vut.cz)
- Dominik Klement, [iklement@fit.vut.cz](mailto:iklement@fit.vut.cz)
