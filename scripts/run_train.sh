#!/usr/bin/env bash
#
# Train DiCoP from scratch on a mixture of prepared corpora, all of them at once.
#
#   scripts/run_train.sh                                   # Libri2Mix + NOTSOFAR + AMI
#   scripts/run_train.sh --datasets l2mix,notsofar,ami,ami-ihm-mix
#   scripts/run_train.sh --dry-run                         # print the train.py line and stop
#   scripts/run_train.sh --val-cuts ami=2,notsofar=10      # cheaper validation
#   scripts/run_train.sh --val-cuts 4 -- trainer.max_steps=20   # smoke test
#
# Everything after '--' is appended verbatim as extra Hydra overrides, so anything in
# conf/dicop.yaml can be reached without a flag here.
#
# The training corpora are the **30 s windowed** cutsets: their supervisions are aligned to the
# window, so one cut is 30 s of audio with every speaker's segments in cut-relative time. The
# exceptions are the long-form rows -- `notsofar-full` (whole ~6-minute sessions) and `ami-600s`
# (~10-minute windows) -- which is what conf/dicop_ctc.yaml exists for; see their rows below. All
# selected cutsets are handed to one dataset (`model.train_ds.manifest_filepath=[a,b,c]`), which
# concatenates them, so an epoch covers every (cut, speaker) pair of every corpus and the corpora
# are interleaved by the shuffling sampler. There is no per-corpus weighting: each contributes in
# proportion to the items it holds, and the resulting split is logged at startup, e.g.
#
#   39.1% of the epoch: 12624 cuts, 75.98 h, 30839 items <- .../ami-sdm_cutset_train_30s.jsonl.gz
#
# To reweight, either edit the selection or pass a corpus twice (Hydra list, after `--`).
#
# "From scratch" means what conf/dicop.yaml means by it: the FDDT blocks start from their
# initialization and everything else warm-starts from `init_from_pretrained`
# (nvidia/parakeet-tdt-0.6b-v2), which is also where the tokenizer comes from. The from-scratch
# schedule (lr=0.5, warmup=2000 under NoamAnnealing) is used as-is; `scripts/run_finetune.sh` is
# the script for continuing an already-trained DiCoP checkpoint instead.
#
# Validation runs on each corpus's WHOLE dev set. `--val-cuts` subsets it (one number, or
# `<name>=<n>` pairs, `0` meaning the full set); the table's fourth column is the fallback for a
# corpus named without a value, and is kept because a cheap subset suits a smoke test.
#
# Subsetting is no longer the default because on a segmented manifest it changes what the metric
# measures, not just its variance: cpWER solves one permutation per *recording*, so the score
# includes whether a speaker lands in the same slot across all that recording's windows -- and
# how many windows share a recording is exactly what subsetting controls. 100 AMI cuts (18
# recordings) gave 5.88 windows per recording against 200 NOTSOFAR cuts (177) giving 1.20, worth
# 0.86 vs 0.33 cpWER on one checkpoint. Full dev sets are 13.23 and 88.06, and comparable.
#
# Scalars go to a TensorBoard event file under the exp dir, written by exp_manager
# (`create_tensorboard_logger`, set explicitly here rather than inherited from the yaml).
# `--tensorboard` additionally serves that directory locally for the duration of the run, so a
# fresh run and every earlier one under the same exp dir show up as separate curves.
#
# `val/cp_wer` -- which checkpoints are selected on -- pools the corpora in the mixture (see
# `hydra_string` below), weighted by the reference words each contributes. Under `--val-cuts` it
# is subset-relative; by default it matches what run_inference.sh + run_scoring.sh report.
#
# Deliberately absent: the *-mdm corpora. Their cutsets are MultiCut over one file per
# microphone, which the dataset rejects rather than silently picking a channel; their first array
# channel is the sdm audio these rows already train on.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MANIFEST_ROOT="${MANIFEST_ROOT:-/home/jovyan/mt-asr-data-prep/manifests}"
PYTHON="${PYTHON:-python}"

# name|train cutset under $MANIFEST_ROOT|dev cutset under $MANIFEST_ROOT|default validation cuts
CORPORA=(
  "l2mix|librimix/librimix_cutset_libri2mix_train-clean-100_30s.jsonl.gz|librimix/librimix_cutset_libri2mix_dev-clean.jsonl.gz|200"
  "l2mix-360|librimix/librimix_cutset_libri2mix_train-clean-360_30s.jsonl.gz|librimix/librimix_cutset_libri2mix_dev-clean.jsonl.gz|200"
  "l2mix-noisy|librimix/librimix_cutset_libri2mix_train-clean-100_noisy_30s.jsonl.gz|librimix/librimix_cutset_libri2mix_dev-clean_noisy.jsonl.gz|200"
  "l3mix|librimix/librimix_cutset_libri3mix_train-clean-100_30s.jsonl.gz|librimix/librimix_cutset_libri3mix_dev-clean.jsonl.gz|200"
  "notsofar|notsofar1/notsofar1_sdm_train_set_240825.1_train_cutset_30s.jsonl.gz|notsofar1/notsofar1_sdm_dev_set_240825.1_dev1_cutset.jsonl.gz|20"
  # Whole sessions rather than 30 s windows: 346 cuts of 6-8 minutes, 1884 (cut, speaker) items,
  # 35.6 h. Train context matches the sessions the model is scored on, at a cost that makes it
  # unusable at the default batch size: the transducer joint is (B, T, U, V), and at T = 4500
  # encoder frames a single median item is a 6.6 GiB fp32 tensor (27.5 GiB for the longest),
  # before the loss and its gradient. Use `--batch-size 1` and buy the batch back with `--accum`,
  # and lower `--warmup-steps` with it -- there are only ~470 optimizer steps per epoch at
  # batch 1 x accum 4 on one GPU, so the 2000-step default would spend four epochs warming up.
  "notsofar-full|notsofar1/notsofar1_sdm_train_set_240825.1_train_cutset.jsonl.gz|notsofar1/notsofar1_sdm_dev_set_240825.1_dev1_cutset.jsonl.gz|20"
  "ami|ami/ami-sdm_cutset_train_30s.jsonl.gz|ami/ami-sdm_cutset_dev.jsonl.gz|4"
  # Long-form AMI: ~10-minute windows cut at silence, so no utterance straddles a boundary
  # (mt-asr-data-prep/src/window_at_silence.py). 1481 windows, 215 h, 5864 (window, speaker)
  # items -- three passes over AMI at different random offsets, so it outweighs notsofar-full
  # roughly 3:1 in items and 6:1 in audio when the two are trained together.
  "ami-600s|ami/ami-sdm_cutset_train_600s.jsonl.gz|ami/ami-sdm_cutset_dev.jsonl.gz|4"
  "ami-ihm-mix|ami/ami-ihm-mix_cutset_train_30s.jsonl.gz|ami/ami-ihm-mix_cutset_dev.jsonl.gz|4"
  # Matched-condition pairs: the same 30 s training windows as `notsofar`/`ami` above, but
  # validated on 30 s dev windows instead of whole 6-minute (NOTSOFAR) or 30-90-minute (AMI)
  # sessions. The plain rows deliberately keep the long-form dev set -- for a target-speaker
  # model, "train short, score the session you actually care about" is the intended setup. These
  # rows are for the case where the *validation loss itself* has to be comparable to the training
  # loss, which a length mismatch of two orders of magnitude makes meaningless -- notably the SOT
  # head, whose decoder is a sequence model whose difficulty scales with target length.
  #
  # Built by `mt-asr-data-prep/src/pre_segment_using_alignments.py --max_len 30` (2026-08-19),
  # deliberately WITHOUT the `--stochastic --num_stochastic_copies 2` the training sets use: the
  # extra randomly-offset passes are oversampling, which is right for training and would just
  # score the same audio three times here. NOTSOFAR: 2341 windows, 18.50 h, from 177 sessions.
  # AMI: 1585 windows, 8.94 h, from 18 sessions.
  #
  # One caveat when reading absolute WER off these. The windower keeps a supervision that
  # straddles a window boundary whole in *both* neighbours, so a few words are scored twice --
  # measured on word-alignment positions, 3.5% of NOTSOFAR words and 8.6% of AMI words appear in
  # exactly two windows. It is the same rule the 30 s training sets are built with (that is what
  # makes the conditions matched) and it is identical across systems, so comparisons are fair;
  # the dev audio is just not a clean partition.
  "notsofar-30s|notsofar1/notsofar1_sdm_train_set_240825.1_train_cutset_30s.jsonl.gz|notsofar1/notsofar1_sdm_dev_set_240825.1_dev1_cutset_30s.jsonl.gz|200"
  "ami-30s|ami/ami-sdm_cutset_train_30s.jsonl.gz|ami/ami-sdm_cutset_dev_30s.jsonl.gz|100"
)

SELECTION="l2mix,notsofar,ami"
VAL_SELECTION=""
VAL_CUTS_SPEC=""
EXP_DIR=""
RUN_NAME=""
LR=0.5
WARMUP_STEPS=2000
MAX_EPOCHS=100
BATCH_SIZE=16
ACCUM=""   # unset: leave `trainer.accumulate_grad_batches` to the config
DEVICES=-1
EVAL_AT_START=0
RESUME=0
WANDB=0
TENSORBOARD=0
TB_PORT=6006
SKIP_CHECKS=0
DRY_RUN=0
LIST_ONLY=0
EXTRA_ARGS=()

usage() {
  cat <<EOF
Train DiCoP from scratch on several prepared corpora simultaneously.

Usage: $(basename "$0") [options] [-- <extra Hydra overrides>]

Data:
  --datasets LIST       Comma-separated corpus names, trained on together.
                        Default: $SELECTION
  --val-datasets LIST   Comma-separated subset of --datasets to validate on. The rest are
                        trained on but never decoded. Default: all of them.
  --val-cuts SPEC       Validation subset size: a single number for every corpus, or
                        comma-separated <name>=<n> pairs. 0 means that corpus's full dev
                        set. Default: the full dev set of every corpus. A corpus named in
                        a pairs spec without a value falls back to the --list default.
  --manifest-root DIR   mt-asr-data-prep manifests. Default: $MANIFEST_ROOT

Run:
  --exp-dir DIR         Default: $REPO_ROOT/exps/multi-corpus
  --name NAME           Run name under the exp dir. Default: dicop-<datasets>
  --resume              Continue an interrupted run in the same exp dir.
  --eval-at-start       Validate once before training. Confirms the warm start.
  --tensorboard         Serve the exp dir with a local TensorBoard for as long as training
                        runs. Scalars are written either way; this only views them.
  --tb-port N           Port for --tensorboard. Default: $TB_PORT
  --wandb               Also log to Weights & Biases.

Optimization (conf/dicop.yaml's from-scratch defaults):
  --lr F                NoamAnnealing base lr. Default: $LR
  --warmup-steps N      Default: $WARMUP_STEPS
  --max-epochs N        Default: $MAX_EPOCHS
  --batch-size N        Training batch size. Default: $BATCH_SIZE
  --accum N             Gradient accumulation steps. Default: whatever the config sets,
                        which for the ladder configs is chosen per chunk length.
  --devices SPEC        Lightning devices; -1 is all visible GPUs. Default: $DEVICES

Other:
  --skip-checks         Skip the preflight checks.
  --dry-run             Print the commands without running them.
  --list                Show the known corpora and the resolved settings, then exit.
  -h, --help            This message.

If it runs out of memory, halve --batch-size and double --accum to hold the effective batch.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets) SELECTION="$2"; shift 2 ;;
    --val-datasets) VAL_SELECTION="$2"; shift 2 ;;
    --val-cuts) VAL_CUTS_SPEC="$2"; shift 2 ;;
    --manifest-root) MANIFEST_ROOT="$2"; shift 2 ;;
    --exp-dir) EXP_DIR="$2"; shift 2 ;;
    --name) RUN_NAME="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --eval-at-start) EVAL_AT_START=1; shift ;;
    --tensorboard) TENSORBOARD=1; shift ;;
    --tb-port) TB_PORT="$2"; TENSORBOARD=1; shift 2 ;;
    --wandb) WANDB=1; shift ;;
    --lr) LR="$2"; shift 2 ;;
    --warmup-steps) WARMUP_STEPS="$2"; shift 2 ;;
    --max-epochs) MAX_EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --accum) ACCUM="$2"; shift 2 ;;
    --devices) DEVICES="$2"; shift 2 ;;
    --skip-checks) SKIP_CHECKS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --list) LIST_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

known_names() {
  local row
  for row in "${CORPORA[@]}"; do printf '%s\n' "${row%%|*}"; done
}

row_for() {
  local name="$1" row
  for row in "${CORPORA[@]}"; do
    [[ "${row%%|*}" == "$name" ]] && { printf '%s\n' "$row"; return 0; }
  done
  return 1
}

# Table order rather than the order they were named, so the same set of corpora always produces
# the same cut order and the same run name.
SELECTED=()
IFS=',' read -r -a requested <<<"$SELECTION"
for name in "${requested[@]}"; do
  [[ -z "$name" ]] && continue
  if ! row_for "$name" >/dev/null; then
    echo "Unknown corpus: $name" >&2
    echo "Known: $(known_names | paste -sd, -)" >&2
    exit 2
  fi
done
for row in "${CORPORA[@]}"; do
  name="${row%%|*}"
  for requested_name in "${requested[@]}"; do
    if [[ "$name" == "$requested_name" ]]; then SELECTED+=("$name"); break; fi
  done
done
if [[ ${#SELECTED[@]} -eq 0 ]]; then
  echo "No corpus selected by --datasets $SELECTION" >&2
  exit 2
fi

# `--val-datasets` marks which selected corpora are decoded at validation. Training is
# unaffected -- a corpus left out still contributes every cut to the epoch, it just gets no dev
# decoding and no `val/<corpus>/*` scalars. Decoding is the expensive half: on the five-corpus
# mixture the LibriMix dev sets are 9000 of 12903 validation windows.
VALIDATES=()
if [[ -z "$VAL_SELECTION" ]]; then
  for _ in "${SELECTED[@]}"; do VALIDATES+=(1); done
else
  IFS=',' read -r -a val_requested <<<"$VAL_SELECTION"
  for name in "${val_requested[@]}"; do
    [[ -z "$name" ]] && continue
    found=0
    for selected_name in "${SELECTED[@]}"; do
      [[ "$name" == "$selected_name" ]] && found=1 && break
    done
    if [[ $found -eq 0 ]]; then
      echo "--val-datasets names $name, which is not in --datasets $SELECTION" >&2
      exit 2
    fi
  done
  for name in "${SELECTED[@]}"; do
    found=0
    for requested_name in "${val_requested[@]}"; do
      [[ "$name" == "$requested_name" ]] && found=1 && break
    done
    VALIDATES+=("$found")
  done
  if [[ ! " ${VALIDATES[*]} " == *" 1 "* ]]; then
    echo "--val-datasets $VAL_SELECTION leaves nothing to validate on" >&2
    exit 2
  fi
fi

# `--val-cuts` is either one number for everything or per-corpus `<name>=<n>` pairs; anything
# unnamed keeps the table default. Without the flag at all, every corpus is validated on its
# WHOLE dev set (0), because a subset silently changes what the metric measures -- see the
# header.
val_cuts_for() {
  local name="$1" default="$2" token
  [[ -z "$VAL_CUTS_SPEC" ]] && { printf '0\n'; return; }
  if [[ "$VAL_CUTS_SPEC" =~ ^[0-9]+$ ]]; then printf '%s\n' "$VAL_CUTS_SPEC"; return; fi
  IFS=',' read -r -a pairs <<<"$VAL_CUTS_SPEC"
  for token in "${pairs[@]}"; do
    [[ -z "$token" ]] && continue
    if [[ "$token" != *=* ]]; then
      echo "--val-cuts takes a number or <name>=<n> pairs, got '$token'" >&2
      exit 2
    fi
    if ! row_for "${token%%=*}" >/dev/null; then
      echo "Unknown corpus in --val-cuts: ${token%%=*}" >&2
      exit 2
    fi
    [[ "${token%%=*}" == "$name" ]] && { printf '%s\n' "${token#*=}"; return; }
  done
  printf '%s\n' "$default"
}

[[ -z "$EXP_DIR" ]] && EXP_DIR="$REPO_ROOT/exps/multi-corpus"
if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="dicop-$(IFS=+; echo "${SELECTED[*]}")"
fi

TRAIN_CUTS=()
DEV_CUTS=()
VAL_CUTS_PATHS=()
VAL_CUTS_NUMS=()
for name in "${SELECTED[@]}"; do
  IFS='|' read -r _ train dev default_val <<<"$(row_for "$name")"
  num="$(val_cuts_for "$name" "$default_val")"
  TRAIN_CUTS+=("$MANIFEST_ROOT/$train")
  DEV_CUTS+=("$MANIFEST_ROOT/$dev")
  VAL_CUTS_NUMS+=("$num")
  # Keyed by corpus and size, so changing --val-cuts builds a new subset rather than silently
  # reusing the previous one under a stale name.
  if [[ "$num" == "0" ]]; then
    VAL_CUTS_PATHS+=("$MANIFEST_ROOT/$dev")
  else
    VAL_CUTS_PATHS+=("$EXP_DIR/val_${name}_${num}.jsonl.gz")
  fi
done

run() {
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '  '
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

# Hydra list override: quoted elements, so a path is never mistaken for part of the grammar.
hydra_list() {
  local out="" path
  for path in "$@"; do
    [[ -n "$out" ]] && out+=","
    out+="'$path'"
  done
  printf '[%s]' "$out"
}

# Hydra mapping override, taking name/path pairs. Validation uses this rather than a bare list so
# each corpus is scored and logged under its own name (val/ami/cp_wer, ...) instead of after the
# subset file it happens to be read from; `val/cp_wer`, the monitored metric, is logged alongside
# them, pooled over the mixture.
hydra_dict() {
  local out=""
  while [[ $# -gt 0 ]]; do
    [[ -n "$out" ]] && out+=","
    out+="$1:'$2'"
    shift 2
  done
  printf '{%s}' "$out"
}

settings() {
  echo "Corpora    : ${SELECTED[*]}"
  local i
  for i in "${!SELECTED[@]}"; do
    printf '  %-12s train %s\n' "${SELECTED[$i]}" "${TRAIN_CUTS[$i]}"
    if [[ "${VALIDATES[$i]}" != "1" ]]; then
      printf '  %-12s val   (not validated; --val-datasets %s)\n' "" "$VAL_SELECTION"
    elif [[ "${VAL_CUTS_NUMS[$i]}" == "0" ]]; then
      printf '  %-12s val   %s (full dev set)\n' "" "${DEV_CUTS[$i]}"
    else
      printf '  %-12s val   %s (%s cuts of %s)\n' "" "${VAL_CUTS_PATHS[$i]}" \
        "${VAL_CUTS_NUMS[$i]}" "${DEV_CUTS[$i]}"
    fi
  done
  echo "Exp dir    : $EXP_DIR/$RUN_NAME"
  echo "Optim      : lr=$LR warmup=$WARMUP_STEPS epochs=$MAX_EPOCHS batch=$BATCH_SIZE accum=${ACCUM:-<config>} devices=$DEVICES"
  echo "TensorBoard: tensorboard --logdir $EXP_DIR --port $TB_PORT"
}

if [[ $LIST_ONLY -eq 1 ]]; then
  printf '%-14s %-4s %-8s %s\n' NAME VAL TRAIN 'TRAIN CUTSET'
  for row in "${CORPORA[@]}"; do
    IFS='|' read -r name train dev default_val <<<"$row"
    state=missing
    [[ -s "$MANIFEST_ROOT/$train" && -s "$MANIFEST_ROOT/$dev" ]] && state=ok
    printf '%-14s %-4s %-8s %s\n' "$name" "$default_val" "$state" "$train"
  done
  echo
  echo "VAL is the fallback subset size for a corpus named in --val-cuts; the default is the"
  echo "full dev set. See --val-cuts."
  echo
  settings
  exit 0
fi

if [[ $SKIP_CHECKS -eq 0 ]]; then
  for path in "${TRAIN_CUTS[@]}" "${DEV_CUTS[@]}"; do
    if [[ ! -s "$path" ]]; then
      echo "Cutset missing or empty: $path" >&2
      echo "Check --manifest-root (currently $MANIFEST_ROOT) and --datasets." >&2
      exit 2
    fi
  done
  # The TDT loss is a numba CUDA kernel, so a broken toolchain fails on the first backward pass,
  # well after the model and the cutsets have been loaded. Catch it up front instead.
  #
  # `cuda.is_available()` alone does not catch it. Under numba 0.66 it returns True and the
  # forward kernels compile, then the *gradient* kernel dies with "Signature mismatch: 2 argument
  # types given, but function takes 1 arguments": 0.66 implements the `min`/`max` builtins as a
  # vararg overload (`def impl(*x)`) that the CUDA device-function compiler cannot handle, and
  # NeMo clamps with `min(g, clamp)` / `max(g, -clamp)` in compute_tdt_grad_kernel. So the probe
  # compiles and launches that exact construct rather than just asking whether CUDA is there.
  # Skipped under --dry-run, which only prints a command and needs no working runtime.
  if [[ $DRY_RUN -eq 0 ]]; then
    probe_status=0
    probe="$("$PYTHON" -W ignore - <<'PY' 2>&1
import sys

try:
    import numpy as np
    from numba import cuda
except Exception as exc:
    sys.exit(f"{type(exc).__name__}: {exc}")

if not cuda.is_available():
    sys.exit("numba reports no usable CUDA device or toolchain")

@cuda.jit
def clamp_probe(out, clamp):
    i = cuda.grid(1)
    g = out[i]
    g = min(g, clamp)
    g = max(g, -clamp)
    out[i] = g

try:
    buf = cuda.to_device(np.zeros(1, dtype=np.float32))
    clamp_probe[1, 1](buf, 1.0)
    cuda.synchronize()
except Exception as exc:
    sys.exit(f"{type(exc).__name__}: {exc}")
PY
)" || probe_status=$?
    if [[ $probe_status -ne 0 ]]; then
      echo "numba cannot compile the TDT gradient kernel, and training needs it:" >&2
      echo "  ${probe##*$'\n'}" >&2
      echo "A 'Signature mismatch' here means numba 0.66; pin numba==0.65.1." >&2
      echo "A missing libnvvm.so means no CUDA toolkit; conda install -c nvidia cuda-nvcc." >&2
      echo "See the Setup section of README.md. Pass --skip-checks to run anyway." >&2
      exit 2
    fi
  fi
fi

[[ $DRY_RUN -eq 0 ]] && mkdir -p "$EXP_DIR"

for i in "${!SELECTED[@]}"; do
  [[ "${VALIDATES[$i]}" != "1" ]] && continue
  [[ "${VAL_CUTS_NUMS[$i]}" == "0" ]] && continue
  if [[ -s "${VAL_CUTS_PATHS[$i]}" ]]; then
    echo "Reusing validation subset ${VAL_CUTS_PATHS[$i]}"
  else
    echo "Building the ${VAL_CUTS_NUMS[$i]}-cut ${SELECTED[$i]} validation subset"
    run "$PYTHON" "$REPO_ROOT/scripts/subset_cutset.py" \
      --cuts "${DEV_CUTS[$i]}" --output "${VAL_CUTS_PATHS[$i]}" --num "${VAL_CUTS_NUMS[$i]}"
  fi
done

val_pairs=()
for i in "${!SELECTED[@]}"; do
  [[ "${VALIDATES[$i]}" != "1" ]] && continue
  val_pairs+=("${SELECTED[$i]}" "${VAL_CUTS_PATHS[$i]}")
done

cmd=("$PYTHON" "$REPO_ROOT/train.py")
cmd+=("model.train_ds.manifest_filepath=$(hydra_list "${TRAIN_CUTS[@]}")")
cmd+=("model.validation_ds.manifest_filepath=$(hydra_dict "${val_pairs[@]}")")
cmd+=("model.train_ds.batch_size=$BATCH_SIZE")
cmd+=("model.optim.lr=$LR")
cmd+=("model.optim.sched.warmup_steps=$WARMUP_STEPS")
cmd+=("trainer.max_epochs=$MAX_EPOCHS")
# Only when asked for: otherwise a config that sets its own accumulation -- as the ladder
# configs do, to hold the effective batch up as the chunk length grows -- would be silently
# overridden back to 1 by this line.
if [[ -n "$ACCUM" ]]; then cmd+=("trainer.accumulate_grad_batches=$ACCUM"); fi
cmd+=("trainer.devices=$DEVICES")
cmd+=("exp_manager.exp_dir=$EXP_DIR")
cmd+=("exp_manager.name=$RUN_NAME")
# Set rather than assumed: this script's runs are watched on TensorBoard, and the yaml default
# is not this script's to rely on. exp_manager writes the event file next to the run's logs.
cmd+=("exp_manager.create_tensorboard_logger=true")
[[ $EVAL_AT_START -eq 1 ]] && cmd+=("evaluate_at_start=true")
[[ $WANDB -eq 1 ]] && cmd+=("exp_manager.create_wandb_logger=true")
if [[ $RESUME -eq 1 ]]; then
  cmd+=("exp_manager.resume_if_exists=true" "exp_manager.resume_ignore_no_checkpoint=true")
fi
cmd+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

settings
echo

if [[ $DRY_RUN -eq 1 ]]; then
  [[ $TENSORBOARD -eq 1 ]] && run tensorboard --logdir "$EXP_DIR" --port "$TB_PORT"
  run "${cmd[@]}"
  exit 0
fi

# The event files are written whether or not this runs; --tensorboard only serves them, for as
# long as training lasts. Its failure (a busy port, usually) must not take the run down with it,
# so it is started detached, checked once, and reported -- never fatal.
TB_PID=""
if [[ $TENSORBOARD -eq 1 ]]; then
  tb_log="$EXP_DIR/tensorboard.log"
  tensorboard --logdir "$EXP_DIR" --port "$TB_PORT" >"$tb_log" 2>&1 &
  TB_PID=$!
  trap '[[ -n "$TB_PID" ]] && kill "$TB_PID" 2>/dev/null; true' EXIT
  sleep 3
  if kill -0 "$TB_PID" 2>/dev/null; then
    echo "TensorBoard on http://localhost:$TB_PORT (pid $TB_PID, log $tb_log)"
  else
    TB_PID=""
    echo "TensorBoard did not start; training continues. Tail of $tb_log:" >&2
    tail -5 "$tb_log" >&2
    echo "The event files are written regardless. Serve them from anywhere with a working" >&2
    echo "TensorBoard: tensorboard --logdir $EXP_DIR --port <free port>" >&2
  fi
  echo
fi

# tee so a long run is watchable and still leaves a log; pipefail would report tee's status.
log="$EXP_DIR/train.log"
started=$SECONDS
set +e
"${cmd[@]}" 2>&1 | tee -a "$log"
status=${PIPESTATUS[0]}
set -e
elapsed=$((SECONDS - started))

if [[ $status -ne 0 ]]; then
  echo "FAILED after ${elapsed}s (exit $status); see $log" >&2
  exit $status
fi

# Two things push the checkpoints below $EXP_DIR/$RUN_NAME: exp_manager nests each run under a
# timestamped version directory, and the monitored metric `val/cp_wer` has a '/' in it, so
# Lightning's filename template turns the part before it into another directory
# (checkpoints/dicop-...--val/cp_wer=...-last.ckpt). Locate the newest .ckpt and report the
# directory it is actually in rather than reconstructing a path that does not exist.
ckpt_dir="$(find "$EXP_DIR/$RUN_NAME" -name '*.ckpt' -printf '%T@ %h\n' 2>/dev/null |
            sort -rn | head -1 | cut -d' ' -f2-)"
[[ -z "$ckpt_dir" ]] && ckpt_dir="$EXP_DIR/$RUN_NAME/<version>/checkpoints"

echo
echo "done in ${elapsed}s -> $EXP_DIR/$RUN_NAME"
echo "scalars: tensorboard --logdir $EXP_DIR --port $TB_PORT"
if [[ -n "$VAL_CUTS_SPEC" ]]; then
  echo "val/<corpus>/cp_wer above is subset-relative (--val-cuts $VAL_CUTS_SPEC); score the"
else
  echo "val/<corpus>/cp_wer above is on the full dev sets, and val/cp_wer pools them by reference"
  echo "word count; re-score the"
fi
echo "best checkpoint with:"
echo "  scripts/run_inference.sh --checkpoint $ckpt_dir/<best>.ckpt --output-dir $REPO_ROOT/exps/decode"
echo "  scripts/run_scoring.sh --decode-dir $REPO_ROOT/exps/decode"
