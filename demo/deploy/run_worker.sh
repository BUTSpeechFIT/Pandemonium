#!/bin/bash
# The forced command of the demo server's worker key on a remote GPU host (see remote.py): serves jobs on
# stdin / stdout until the demo server disconnects. In ~/.ssh/authorized_keys there:
#   restrict,from="147.229.8.178,2001:67c:1220:808::93e5:8b2",command="/home/ipoloka/Pandemonium/demo/deploy/run_worker.sh" ssh-ed25519 AAAA... pandemonium-demo worker

cd "$(dirname "$0")/.." || exit 1

source /home/ipoloka/miniconda3/etc/profile.d/conda.sh
conda activate pandemonium-demo

# The card with the most free memory, unless one is pinned (PANDEMONIUM_WORKER_GPU in ~/.config/pandemonium-demo/worker.env).
[ -f "$HOME/.config/pandemonium-demo/worker.env" ] && source "$HOME/.config/pandemonium-demo/worker.env"
if [ -n "$PANDEMONIUM_WORKER_GPU" ]; then
    export CUDA_VISIBLE_DEVICES=$PANDEMONIUM_WORKER_GPU
else
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
                                  | sort -t, -k2 -nr | head -1 | cut -d, -f1)
fi
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
# Its own log on this host; the demo server sees only failures.
LOG=$HOME/.config/pandemonium-demo/worker.log
mkdir -p "$(dirname "$LOG")"
exec python worker.py 2>>"$LOG"
