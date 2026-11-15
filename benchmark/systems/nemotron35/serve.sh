#!/bin/bash
# benchmark/systems/nemotron35/serve.sh GPU PORT -- the model card's vLLM server command (MTP8, CTC timestamp middleware).
cd "${NEMO_SPEECH:?set NEMO_SPEECH to the nemo-speech checkout with its .venv (setup.sh)}" && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=$1
# FlashInfer JIT-compiles its MoE kernels for sm_120: use the venv's CUDA 13 toolkit, not a conda nvcc on PATH.
export CUDA_HOME=$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export VLLM_PLUGINS=nemo_speechlm
# One ICSI session is 60.5 min, just over the card's 1 h; allow it rather than cut it.
export VLLM_MAX_AUDIO_DECODE_DURATION_S=3700
export VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=180
export VLLM_USE_V2_MODEL_RUNNER=0
exec vllm serve "${N35_MODEL:-nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B}" \
  --served-model-name nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B --port $2 \
  --trust-remote-code --dtype bfloat16 --gpu-memory-utilization 0.94 \
  --max-model-len 65536 --max-num-batched-tokens 4096 --max-num-seqs 192 \
  --enable-chunked-prefill --no-enable-prefix-caching \
  --use-replayssm --mamba-backend flashinfer --mamba-cache-mode none \
  --mm-processor-cache-gb 0 --no-async-scheduling \
  --limit-mm-per-prompt '{"audio":{"count":6,"length":120000000}}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":8}' \
  --reasoning-parser nemotron_v3 \
  --trust-request-mm-kwargs \
  --middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware
