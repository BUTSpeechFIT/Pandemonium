#!/bin/bash
# Build the environment(s) a benchmark system runs in, at the exact package versions and source commits the
# published numbers were produced with (envs/<name>.txt). Each environment is a uv virtualenv under $BENCH_ENVS
# (default benchmark/work/envs); external code is cloned under $BENCH_SRC (default benchmark/work/src).
#
#   benchmark/setup.sh scoring demo diarizen ts-asr-whisper moss vibevoice cocktail nemotron35 api
#
#   scoring         repo requirements: data preparation, score.py, tables.py, DiCoW collect
#   demo            Pandemonium (all conditions), MT-Parakeet, single-speaker; demo/requirements.txt
#   diarizen        BUTSpeechFIT/DiariZen@844f555 (+ diarizen.patch, a torchaudio>=2.9 import shim)
#   ts-asr-whisper  BUTSpeechFIT/TS-ASR-Whisper more_data@0966c5a + dicow/ts-asr-whisper.patch (SE-DiCoW, DiCoW)
#   moss            OpenMOSS/MOSS-Transcribe-Diarize@79c675f
#   vibevoice       microsoft/VibeVoice@16fb2cb (export VIBEVOICE_REPO=$BENCH_SRC/VibeVoice for e2e/run.py)
#   cocktail        Xiaomi CocktailASR (model code ships with the HF checkpoint)
#   nemotron35      NVIDIA-NeMo/Speech@2ed70aba11 + nemotron35/nemo-speech.patch (vLLM server; needs CUDA 13)
#   api             the API runners (requests, soundfile, librosa, provider SDKs)
#
# Re-running is safe: an existing environment is left alone (delete it to rebuild).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(dirname "$HERE")
ENVS=${BENCH_ENVS:-$HERE/work/envs}
SRC=${BENCH_SRC:-$HERE/work/src}
mkdir -p "$ENVS" "$SRC"
command -v uv > /dev/null || { echo "needs uv (https://docs.astral.sh/uv/)" >&2; exit 1; }

clone() {  # url commit dir
  [[ -d $3/.git ]] || git clone --quiet "$1" "$3"
  git -C "$3" checkout --quiet "$2"
}

venv() {  # name python-version -> prints the env's python
  [[ -x $ENVS/$1/bin/python ]] || uv venv --quiet --python "$2" "$ENVS/$1"
  echo "$ENVS/$1/bin/python"
}

pin() {  # python lockfile [extra pip args...]
  local py=$1 lock=$2; shift 2
  uv pip install --quiet --python "$py" -r "$lock" "$@"
}

for name in "$@"; do
  echo "== $name"
  case $name in
    scoring)
      py=$(venv scoring 3.11)
      pin "$py" "$HERE/envs/scoring.txt" ;;
    demo)
      py=$(venv demo 3.12)
      uv pip install --quiet --python "$py" Cython packaging
      uv pip install --quiet --python "$py" -r "$REPO/demo/requirements.txt"
      pin "$py" "$HERE/envs/demo.txt" ;;
    diarizen)
      clone https://github.com/BUTSpeechFIT/DiariZen.git 844f5555b0a98acd0931511fc641a8c5b8ba92c7 "$SRC/DiariZen"
      git -C "$SRC/DiariZen" submodule update --init --quiet
      git -C "$SRC/DiariZen" apply --check "$HERE/systems/diarizen/diarizen.patch" 2> /dev/null \
        && git -C "$SRC/DiariZen" apply "$HERE/systems/diarizen/diarizen.patch"
      py=$(venv diarizen 3.10)
      pin "$py" "$HERE/envs/diarizen.txt"
      uv pip install --quiet --python "$py" --no-deps -e "$SRC/DiariZen" -e "$SRC/DiariZen/pyannote-audio" ;;
    ts-asr-whisper)
      clone https://github.com/BUTSpeechFIT/TS-ASR-Whisper.git 0966c5a "$SRC/TS-ASR-Whisper"
      git -C "$SRC/TS-ASR-Whisper" apply --check "$HERE/systems/dicow/ts-asr-whisper.patch" 2> /dev/null \
        && git -C "$SRC/TS-ASR-Whisper" apply "$HERE/systems/dicow/ts-asr-whisper.patch"
      py=$(venv ts-asr-whisper 3.11)
      pin "$py" "$HERE/envs/ts-asr-whisper.txt"
      echo "   export TSASR=$SRC/TS-ASR-Whisper TSASR_PYTHON=$py" ;;
    moss)
      clone https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git 79c675f "$SRC/MOSS-Transcribe-Diarize"
      py=$(venv moss 3.12)
      pin "$py" "$HERE/envs/moss.txt"
      uv pip install --quiet --python "$py" --no-deps -e "$SRC/MOSS-Transcribe-Diarize" ;;
    vibevoice)
      clone https://github.com/microsoft/VibeVoice.git 16fb2cb "$SRC/VibeVoice"
      py=$(venv vibevoice 3.11)
      pin "$py" "$HERE/envs/vibevoice.txt"
      uv pip install --quiet --python "$py" --no-deps -e "$SRC/VibeVoice"
      echo "   export VIBEVOICE_REPO=$SRC/VibeVoice" ;;
    cocktail)
      py=$(venv cocktail 3.11)
      pin "$py" "$HERE/envs/cocktail.txt" ;;
    nemotron35)
      clone https://github.com/NVIDIA-NeMo/Speech.git 2ed70aba11 "$SRC/nemo-speech"
      git -C "$SRC/nemo-speech" apply --check "$HERE/systems/nemotron35/nemo-speech.patch" 2> /dev/null \
        && git -C "$SRC/nemo-speech" apply "$HERE/systems/nemotron35/nemo-speech.patch"
      [[ -x $SRC/nemo-speech/.venv/bin/python ]] || uv venv --quiet --python 3.12 "$SRC/nemo-speech/.venv"
      pin "$SRC/nemo-speech/.venv/bin/python" "$HERE/envs/nemotron35.txt"
      uv pip install --quiet --python "$SRC/nemo-speech/.venv/bin/python" --no-deps -e "$SRC/nemo-speech"
      echo "   export NEMO_SPEECH=$SRC/nemo-speech   # serve.sh; its FlashInfer JIT needs the venv's CUDA 13 toolkit" ;;
    api)
      py=$(venv api 3.11)
      pin "$py" "$HERE/envs/api.txt" ;;
    *) echo "unknown environment: $name" >&2; exit 2 ;;
  esac
done
