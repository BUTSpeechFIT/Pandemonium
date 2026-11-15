#!/bin/bash
# SE-DiCoW or DiCoW v3.3 on DiariZen's diarization, through BUTSpeechFIT/TS-ASR-Whisper.
#
#   TSASR=/path/to/TS-ASR-Whisper TSASR_PYTHON=/path/to/env/bin/python \
#     benchmark/systems/dicow/run.sh {se_dicow|dicow_v3_3} CORPUS...
#
# TSASR must be a checkout of more_data@0966c5a with ts-asr-whisper.patch applied (setup.sh does both). Needs
# diarizen/diarize.py and diarizen/pred_cutset.py to have run for the corpora. One decode per corpus, outputs in
# $BENCH_OUT/tsasr/<experiment>_<corpus>/, then collect.py turns them into $BENCH_OUT/<system>/<corpus>/*.json.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH=$(dirname "$(dirname "$HERE")")
: "${TSASR:?set TSASR to the patched TS-ASR-Whisper checkout}" "${TSASR_PYTHON:?set TSASR_PYTHON}"
SYSTEM=$1; shift
case $SYSTEM in
  se_dicow) EXP=SE-DiCoW_greedy_diarizen_full ;;
  dicow_v3_3) EXP=DiCoW_v3_3_greedy_diarizen_full ;;
  *) echo "unknown system $SYSTEM" >&2; exit 2 ;;
esac
cp "$HERE/bench_$SYSTEM.yaml" "$TSASR/configs/decode/"
DATA=$("$TSASR_PYTHON" -c "import sys; sys.path.insert(0, '$BENCH'); import bench; print(bench.DATA)")
OUT=$("$TSASR_PYTHON" -c "import sys; sys.path.insert(0, '$BENCH'); import bench; print(bench.OUT)")
CORPORA=$("$TSASR_PYTHON" -c "import sys; sys.path.insert(0, '$BENCH'); import bench; print(' '.join(bench.expand(sys.argv[1:])))" "$@")

export SRC_ROOT=$TSASR EXPERIMENT_PATH=$OUT/tsasr PYTHONPATH="$TSASR:$TSASR/src:${PYTHONPATH:-}"
export WANDB_MODE=disabled MANIFEST_DIR=${MANIFEST_DIR:-/nonexistent} PRETRAINED_CTC_MODELS_PATH=${PRETRAINED_CTC_MODELS_PATH:-}
export PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-} MUSAN_ROOT=${MUSAN_ROOT:-}
mkdir -p "$OUT/tsasr"
cd "$TSASR"
for c in $CORPORA; do
  GT=$DATA/$c/cutset.jsonl.gz; PRED=$OUT/diarizen/$c/pred_cutset.jsonl.gz
  [[ -f $PRED ]] || { echo "missing $PRED: run diarizen/diarize.py and pred_cutset.py first" >&2; exit 1; }
  if compgen -G "$OUT/tsasr/exp/${EXP}_$c/test_*/0/all_session_wer.csv" > /dev/null; then echo "$SYSTEM $c: done"; continue; fi
  # train_cutsets: a decode-only run still builds the training set (from manifests named in the base config) without
  # using it; the eval cutset stands in so no training data is needed
  "$TSASR_PYTHON" src/main.py +decode=bench_$SYSTEM experiment=${EXP}_$c "data.train_cutsets=[$GT]" \
    "data.eval_cutsets=[$GT]" "data.eval_diar_cutsets=[$PRED]" "data.dev_cutsets=[$GT]" "data.dev_diar_cutsets=[$PRED]" \
    > "$OUT/tsasr/${EXP}_$c.log" 2>&1
  echo "$SYSTEM $c: decoded"
done
"$TSASR_PYTHON" "$HERE/collect.py" "$SYSTEM" $CORPORA
