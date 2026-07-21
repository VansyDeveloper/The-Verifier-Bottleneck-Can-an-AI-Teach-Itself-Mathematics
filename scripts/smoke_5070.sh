#!/bin/bash
# Fast end-to-end check for Windows 10 + WSL2 + RTX 5070 12 GB.
# Runtime is measured and logged; no artificial timeout interrupts the work.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate

MODEL=${MODEL:-Qwen/Qwen3-0.6B-Base}
MODEL_TAG=${MODEL//\//_}
PROFILE_TAG=g2_e32_lora8_c128
if [ "${SLEEP_MODE:-0}" = 1 ]; then PROFILE_TAG=${PROFILE_TAG}_sleep; fi
TIMING_FILE=${TIMING_FILE:-logs/rtx5070_${MODEL_TAG}_${PROFILE_TAG}_s1.txt}
CALIBRATION_STEPS=${CALIBRATION_STEPS:-1}
RUN_EPSILON=${RUN_EPSILON:-1}
RUN_TESTS=${RUN_TESTS:-1}
EPSILON_OUT=results/data/exploration_smoke_5070.json
FIGURES_OUT=results/figures/smoke_5070

train_args=(
  --lora --lora-r 8 --lora-alpha 16
  --per-device-batch 2 --grad-accum 1
  --max-completion-length 128 --vllm-gpu-memory-utilization 0.20
)
if [ "${SLEEP_MODE:-0}" = 1 ]; then
  train_args+=(--vllm-enable-sleep-mode)
fi

if [ "${DRY_RUN:-0}" = 1 ]; then
  NPROC=1 ALPHAS="1.0" BETAS="0.0" SEEDS="314159" \
    STEPS="$CALIBRATION_STEPS" GENERATIONS=2 TRAIN_SIZE=2000 \
    EVAL_SIZE=32 EVAL_STEPS=999999 CONFIG_TAG=rtx5070_smoke \
    DRY_RUN=1 bash scripts/sweep.sh "${train_args[@]}"
  echo "DRY RUN: epsilon smoke would run with n=4, K=4, M=16"
  exit 0
fi

mkdir -p logs
if [ "$RUN_TESTS" = 1 ]; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sed -n '1p')
  gpu_memory=${gpu_memory//[[:space:]]/}
  if [ "$gpu_memory" -lt 11000 ]; then
    echo "The documented profile requires the 12 GB desktop RTX 5070; found ${gpu_memory} MiB" >&2
    exit 1
  fi
  python -m unittest discover -s tests -v
  grid_count=$(DRY_RUN=1 bash scripts/sweep.sh | grep -c '^START h2_' || true)
  if [ "$grid_count" -ne 75 ]; then
    echo "Expected 75 paper-grid runs, got $grid_count" >&2
    exit 1
  fi
fi

if [ ! -s "$TIMING_FILE" ] || [ "${FORCE_CALIBRATION:-0}" = 1 ]; then
  python -c \
    'import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])' \
    "$MODEL"
  started=$(date +%s)
  NPROC=1 bash scripts/run_train.sh 1.0 0.0 \
    --model "$MODEL" --run-name "calibration_5070_s${CALIBRATION_STEPS}_${started}" \
    --seed 314159 --noise-mode iid --temperature 1.0 --num-generations 2 \
    --prime-max 29 --k-min 2 --k-max 5 --style compact --eval-split eval \
    --max-steps "$CALIBRATION_STEPS" --eval-steps 999999 \
    --train-size 2000 --eval-size 32 --no-save "${train_args[@]}"
  elapsed=$(( $(date +%s) - started ))
  if [ "$elapsed" -le 0 ]; then
    elapsed=1
  fi
  printf '%s\n' "$elapsed" > "$TIMING_FILE"
  echo "Calibration: $CALIBRATION_STEPS steps in ${elapsed}s"
else
  echo "Reusing calibration from $TIMING_FILE (set FORCE_CALIBRATION=1 to refresh)"
fi

if [ "$RUN_EPSILON" = 1 ]; then
  python eval/eval_exploration.py \
    --model "$MODEL" --n-problems 4 --k 4 --reference-samples 16 \
    --resolution-m 4 8 --temperatures 0.7 1.0 1.3 \
    --proposal-temperature 1.3 --mixture-weights 0 0.5 1 \
    --max-tokens 128 --gpu-memory-utilization 0.70 \
    --out "$EPSILON_OUT"
  python analysis/plot_core_results.py \
    --exploration "$EPSILON_OUT" --out "$FIGURES_OUT"
fi

echo "RTX 5070 smoke test complete"
