#!/bin/bash
# Started by pandemonium-demo.service; serves the demo on 127.0.0.1:7861 behind nginx at /pandemonium/.

cd "$(dirname "$0")/.." || exit 1

source /home/ipoloka/miniconda3/etc/profile.d/conda.sh
conda activate pandemonium-demo

export GRADIO_SERVER_NAME=127.0.0.1
export GRADIO_SERVER_PORT=7861
export GRADIO_ROOT_PATH=/pandemonium
export DEMO_PUBLIC_URL=https://pcspeech-demo.fit.vut.cz/pandemonium/
export DEMO_ACCESS_CODE_FILE=$HOME/.config/pandemonium-demo/access_code
# Emails go out through pcspeech4, which keeps the SMTP password (deploy/pandemonium-mail.py); this
# machine only holds a key restricted there to running that script.
export DEMO_MAIL_SSH=ipoloka@pcspeech4.fit.vutbr.cz
export MAX_SECONDS=2400
export PYTHONUNBUFFERED=1
# Uploads and results; private to the service (PrivateTmp), and the path names no user or UID.
export GRADIO_TEMP_DIR=/tmp/pandemonium
# The service sees a read-only, nearly empty home (pandemonium-demo.service): models come from the local
# cache without the Hugging Face token, scratch caches go to /tmp, and the GPU-kernel cache (slow to
# rebuild: ~50 s at start-up) is kept in the service's own writable folder.
export HF_HUB_OFFLINE=1 HF_HUB_CACHE=$HOME/.cache/huggingface/hub
export XDG_CACHE_HOME=/tmp/cache MPLCONFIGDIR=/tmp/cache/matplotlib
export TRITON_HOME=$HOME/.config/pandemonium-demo/cache TRITON_CACHE_DIR=$HOME/.config/pandemonium-demo/cache/triton \
       TORCHINDUCTOR_CACHE_DIR=$HOME/.config/pandemonium-demo/cache/inductor

exec python app.py
