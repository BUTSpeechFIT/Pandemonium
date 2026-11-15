# Benchmark

Everything needed to reproduce the comparison tables of the [Pandemonium model card](https://huggingface.co/BUT-FIT/Pandemonium):
meeting transcription on six corpora, headset mixes, short fully-overlapped mixtures and single-speaker speech, for
Pandemonium and the systems it is compared with. Every number on the card is in
[`results/published.json`](results/published.json), and [`compare.py`](compare.py) checks a run against it.
Four commercial APIs are published anonymously, as `api_a` to `api_d` (API A–D on the card); their runners are
included like the others, but their results are not attributed to a provider.

What has been verified:

- from the eval sets below, `data/` rebuilds byte-identical audio and references for all 15 sets;
- `score.py` reproduces every published number exactly from the systems' stored outputs;
- rerun through this code, Pandemonium reproduces its published meeting row exactly from the released checkpoint;
- on a smoke set, DiariZen, CocktailASR and MOSS reproduce their stored outputs exactly. SE-DiCoW and DiCoW do on
  the mixtures; on a meeting decoded alone they differ by the odd word, because batched half-precision decoding
  depends on which windows share a batch, so decode a whole corpus in one run, as the published runs did.
  VibeVoice's greedy decoding is not deterministic run to run; MT-Parakeet's smoke outputs also differed slightly
  (NeMo batches the whole manifest, so this may be the same batch effect).

## How it fits together

```
data/prepare_*.py  ─►  $BENCH_DATA/<corpus>/{manifest.json, ref.stm, cutset.jsonl.gz, audio/}
systems/*/run*     ─►  $BENCH_OUT/<system>/<corpus>/<session>.json     (one format for every system)
score.py           ─►  $BENCH_OUT/<system>/scores.json                 (cpWER, tcpWER with a 5 s collar)
tables.py / compare.py / rtf.py
```

`BENCH_DATA` and `BENCH_OUT` default to `benchmark/work/data` and `benchmark/work/out`. A system's output is one
JSON per recording with time-stamped, speaker-labelled segments; `score.py` normalizes their text with
`whisper_nsf` (the normalizer of the paper) and scores them against `ref.stm`. Corpus names can be given one by one
or as the groups `meetings`, `ihm_mix` and `mixtures`.

## 1. Environments and models

Each system runs in its own environment, at the package versions the published numbers were produced with
([`envs/`](envs/)) and with external code checked out at a fixed commit:

```bash
benchmark/setup.sh scoring demo            # enough for data preparation, Pandemonium and scoring
benchmark/setup.sh diarizen ts-asr-whisper cocktail moss vibevoice nemotron35 api
```

| environment | used by | source |
|---|---|---|
| `scoring` | `data/`, `score.py`, `tables.py`, DiCoW collection | this repository |
| `demo` | Pandemonium, MT-Parakeet, single-speaker | [`demo/requirements.txt`](../demo/requirements.txt), NeMo @ `16b0ff2` |
| `diarizen` | DiariZen | BUTSpeechFIT/DiariZen @ `844f555` + [`diarizen.patch`](systems/diarizen/diarizen.patch) (torchaudio ≥ 2.9 import shim) |
| `ts-asr-whisper` | SE-DiCoW, DiCoW v3.3 | BUTSpeechFIT/TS-ASR-Whisper `more_data` @ `0966c5a` + [`ts-asr-whisper.patch`](systems/dicow/ts-asr-whisper.patch) |
| `cocktail` | Xiaomi CocktailASR | model code ships with the checkpoint |
| `moss` | MOSS-Transcribe-Diarize | OpenMOSS/MOSS-Transcribe-Diarize @ `79c675f` |
| `vibevoice` | VibeVoice-ASR | microsoft/VibeVoice @ `16fb2cb` |
| `nemotron35` | Nemotron-3.5-Transcribe (vLLM server) | NVIDIA-NeMo/Speech @ `2ed70ab` + [`nemo-speech.patch`](systems/nemotron35/nemo-speech.patch) |
| `api` | commercial APIs | requests, provider SDKs |

The TS-ASR-Whisper patch is not optional: it changes the decode (supervisions whose text normalizes to nothing no
longer enter the speaker masks, and enrollment windows fit recordings shorter than 30 s).

Models are pinned too. [`models.py`](models.py) lists the Hugging Face revision of each; fetch them once and run
offline, so that every loader resolves to exactly those weights:

```bash
python benchmark/models.py --fetch
export HF_HUB_OFFLINE=1
```

All the models are public; no Hugging Face token is needed.

## 2. Eval sets

### Meetings

One Lhotse cutset per corpus (whole recordings, `cut.start == 0`, supervisions with speaker and text), either a
Shar set or a file-backed cutset:

```bash
python benchmark/data/prepare_meetings.py notsofar /path/to/notsofar1-sdm_eval.tar
```

The corpora are licensed separately and are not distributed here. The sets are:

| name | corpus and split | recordings | hours | speakers | reference |
|---|---|---|---|---|---|
| `notsofar` | NOTSOFAR-1 eval (`240629.1_eval_small_with_GT`), single device; 80 meetings × 2 devices | 160 | 16.67 | 3–7 | official |
| `ami` | AMI test, single distant microphone | 16 | 9.06 | 3–4 | official |
| `icsi` | ICSI test meetings Bmr013, Bmr018, Bro021, one distant microphone | 3 | 2.77 | 7 | ICSI's own `.mrt` transcripts, utterance-level ¹ |
| `mixer6` | Mixer 6 eval interviews, one distant microphone | 23 | 6.02 | 2 | official |
| `sbc` | Santa Barbara Corpus, the 44 of 60 sessions with audio (22.05 kHz) | 44 | 17.33 | 2–7 | official |
| `dipco` | DiPCo eval sessions S25, S27, S30, S31, S32, array U04 channel 4 | 5 | 2.60 | 4 | official |

¹ Turns run from their first to their last word and keep the pauses inside, about 20 % more speech than the
reference shipped with the set's older version.

Session ids are the cut ids (`sdm_MTG_32000_sc_meetup_0-0`, `S25-ch24`, ...). `score.py` strips the trailing
partition tag (`-0`, `-sdm`, `-ch24`) on both sides, so a system may name a recording either way.

### Headset mixes

AMI's official `Mix-Headset` recordings, and the NOTSOFAR-1 close-talk channels summed. Run after `notsofar` and
`ami`, whose references they reuse:

```bash
python benchmark/data/prepare_ihm_mix.py \
    --nsf-meetings /path/to/eval_set/240629.1_eval_small_with_GT/MTG --ami-ihm-mix /path/to/ami/ihm-mix/wav_db
```

### Short fully-overlapped mixtures

LibriSpeechMix (test-clean, 1–3 speakers) and LibriMix (test-clean, 2–3 speakers, clean and with WHAM! noise), from
the cutsets [BUTSpeechFIT/mt-asr-data-prep](https://github.com/BUTSpeechFIT/mt-asr-data-prep) builds:

```bash
(cd mt-asr-data-prep && ./prepare.sh --datasets librispeech,librispeechmix,librimix --root-dir /data)
python benchmark/data/prepare_mixtures.py /data/manifests
```

### Single-speaker

```bash
hf download hf-audio/esb-datasets-test-only-sorted --repo-type dataset --local-dir $BENCH_DATA/esb
```

## 3. Systems

Run each in its environment (`benchmark/work/envs/<name>/bin/python`, written `python` below). Every runner skips
recordings already done, so an interrupted run resumes.

```bash
# Pandemonium: the released pipeline (Nemotron-3-Diarization, soft), reference activity, external RTTMs
python benchmark/systems/pandemonium/run.py nemotron meetings ihm_mix mixtures
python benchmark/systems/pandemonium/run.py oracle   meetings ihm_mix mixtures
python benchmark/systems/pandemonium/run.py rttm notsofar --rttm NOTSOFAR...0.5_conf-refined.rttm \
    --out $BENCH_OUT/pandemonium_precision          # also ami, icsi, mixer6, sbc; dipco with --session-offset 24

# DiariZen, then the systems conditioned on it
python benchmark/systems/diarizen/diarize.py meetings ihm_mix mixtures                 # diarizen env
python benchmark/systems/diarizen/pred_cutset.py meetings ihm_mix mixtures             # scoring env
TSASR=... TSASR_PYTHON=... benchmark/systems/dicow/run.sh se_dicow   meetings ihm_mix mixtures
TSASR=... TSASR_PYTHON=... benchmark/systems/dicow/run.sh dicow_v3_3 meetings ihm_mix mixtures
python benchmark/systems/cocktail/run.py meetings ihm_mix mixtures

# end-to-end models
PYTHON=... benchmark/systems/mt_parakeet/run.sh meetings ihm_mix mixtures              # demo env
python benchmark/systems/e2e/run.py moss meetings ihm_mix mixtures
python benchmark/systems/e2e/run.py vibevoice meetings ihm_mix
python benchmark/systems/e2e/run.py vibevoice mixtures --max-new-tokens 4096
benchmark/systems/nemotron35/serve.sh 0 8001 &                                         # one server per GPU
python benchmark/systems/nemotron35/run.py meetings ihm_mix mixtures --ports 8001

# commercial APIs (keys in ~/.config/api-asr/<provider>, or $API_KEY_DIR)
python benchmark/systems/api/run_api.py assemblyai meetings      # also deepgram smallest azure mistral elevenlabs openai
GEMINI_WIN_S=900 python benchmark/systems/api/run_api.py gemini meetings
python benchmark/systems/api/run_meta.py meetings
```

The Precision-3 RTTMs behind the published row are not included here. `run.py rttm` takes any diarization in RTTM
form; its recording ids are matched to sessions as described in the script.

Settings that matter, all already in the commands above or the runners' defaults:

- **VibeVoice** sometimes loops until its token cap. On the mixtures, where real outputs stay under 400 tokens, the
  cap is 4096; the meetings keep 32768. A looping output whose JSON never closes is salvaged by `score.py`.
- **Gemini** is run on 15-minute windows (`GEMINI_WIN_S=900`) with its own speakers per window, linked across
  windows through DiariZen, and is scored with cpWER: its timestamps drift on long audio.
- **MAI-Transcribe-2** (`azure`) returns every timestamp as 0 when it detects several languages in a recording;
  those sessions were rerun with `AZURE_LOCALE=en` (one, SBC050, stays at 0).
- **Mistral** refused two recordings whole (ICSI Bro021, DiPCo S25); they were transcribed in two overlapping halves
  with `systems/api/run_api_split.py`.
- **smallest.ai** diarizes on `pulse` only; `pulse-pro` ignores the request.
- Nemotron-3.5's vLLM plugin caps audio at 1 h; the patch raises it to 3700 s for ICSI Bro021 (60.5 min).

## 4. Scoring

```bash
python benchmark/score.py $BENCH_OUT/<system>          # every prepared corpus the system has output for
python benchmark/compare.py                            # against results/published.json
python benchmark/tables.py --card                      # the model card's tables (--card: its naming)
```

A recording without output, or with no words, counts as all deletions. meeteval's 20-speaker limit is lifted for
APIs that over-cluster long meetings. The single-speaker sets use the Open ASR Leaderboard's protocol instead:
`python benchmark/single/run.py {pandemonium,parakeet} SET` and `python benchmark/single/score.py`.

## 5. Speed (RTFx)

RTFx is audio duration divided by processing time, including diarization, on one median-length recording per
meeting set ([`data/rtf_subset.json`](data/rtf_subset.json), 10,022 s), each system alone on one GPU, warm-up
excluded:

```bash
python benchmark/systems/pandemonium/run.py nemotron meetings --sessions benchmark/data/rtf_subset.json \
    --warmup $BENCH_DATA/notsofar/audio/.../sdm_MTG_32009_sc_meetup_0-14.wav --out $BENCH_OUT/rtf/pandemonium
python benchmark/rtf.py $BENCH_OUT/rtf/pandemonium
python benchmark/rtf.py $BENCH_OUT/rtf/cocktail --plus-diarizen $BENCH_OUT/rtf/diarizen
```

For API systems the time is the client's wall clock including upload, which is not directly comparable. The
Precision-3 row is the pyannoteAI API (`systems/api/run_pyannote.py`, 92.7 s) plus Pandemonium on its output.
