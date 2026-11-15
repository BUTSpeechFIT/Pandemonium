#!/bin/bash
# Nemotron-3-Diarization + nvidia/multitalker-parakeet-streaming-0.6b-v1, through NeMo's own multitalker streaming
# example, at the NeMo commit demo/requirements.txt pins (16b0ff258); run in the demo environment.
#
#   benchmark/systems/mt_parakeet/run.sh CORPUS...        (groups allowed: meetings ihm_mix mixtures)
#
# Settings: up to 8 speakers, attention context [70, 13].
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export BENCH=$(dirname "$(dirname "$HERE")")
PY=${PYTHON:-python}
NEMO_COMMIT=16b0ff2584c6355c53cbd6c28f0f431a76b24956
SCRIPT=examples/asr/asr_cache_aware_streaming/speech_to_text_multitalker_streaming_infer.py
OUT=$("$PY" -c "import sys; sys.path.insert(0, '$BENCH'); import bench; print(bench.OUT)")
CORPORA=$("$PY" -c "import sys; sys.path.insert(0, '$BENCH'); import bench; print(' '.join(bench.expand(sys.argv[1:])))" "$@")
W=$OUT/mt_parakeet_raw; mkdir -p "$W"
[[ -f $W/infer.py ]] || curl -fsSL "https://raw.githubusercontent.com/NVIDIA/NeMo/$NEMO_COMMIT/$SCRIPT" -o "$W/infer.py"
for c in $CORPORA; do
  [[ -f $W/$c.json ]] && { echo "mt_parakeet $c: done"; continue; }
  "$PY" - "$c" > "$W/$c.manifest.json" <<'PYEOF'
import json, os, sys
import soundfile as sf
sys.path.insert(0, os.environ["BENCH"])
import bench
for sess, path in bench.load_manifest(sys.argv[1]).items():
    print(json.dumps({"audio_filepath": path, "offset": 0, "duration": sf.info(path).duration}))
PYEOF
  "$PY" "$W/infer.py" asr_model=nvidia/multitalker-parakeet-streaming-0.6b-v1 diar_model=nvidia/Nemotron-3-Diarization \
    max_num_of_spks=8 att_context_size="[70,13]" generate_realtime_scripts=False \
    manifest_file="$W/$c.manifest.json" output_path="$W/$c.json" > "$W/$c.log" 2>&1
  echo "mt_parakeet $c: decoded"
done
"$PY" "$HERE/collect.py" $CORPORA
