#!/bin/bash
# Native Apple-Silicon smoke test: MPS GRPO + Serguei epsilon + one figure.
# Runtime is measured and logged; no artificial timeout interrupts the work.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SCRIPT_PATH="$SCRIPT_DIR/$(basename "$0")"
cd "$SCRIPT_DIR/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate

MODEL=${MODEL:-Qwen/Qwen3-0.6B-Base}
MODEL_TAG=${MODEL//\//_}
TIMING_FILE=${TIMING_FILE:-logs/m1_${MODEL_TAG}_lora8_g2_c64_s1.txt}
RUN_EPSILON=${RUN_EPSILON:-1}
RUN_TESTS=${RUN_TESTS:-1}
EPSILON_OUT=results/data/exploration_smoke_m1.json
FIGURES_OUT=results/figures/smoke_m1

train_args=(
  --lora --lora-r 8 --lora-alpha 16
  --per-device-batch 2 --grad-accum 1
  --max-completion-length 64 --generation-backend transformers --precision fp16
)

if [ "${DRY_RUN:-0}" = 1 ]; then
  MODEL="$MODEL" NPROC=1 PRECISION=fp16 ALPHAS="1.0" BETAS="0.0" SEEDS="314159" \
    STEPS=1 GENERATIONS=2 TRAIN_SIZE=512 EVAL_SIZE=8 EVAL_STEPS=999999 \
    CONFIG_TAG=m1_smoke DRY_RUN=1 bash scripts/sweep.sh "${train_args[@]}"
  echo "DRY RUN: MPS epsilon smoke would use n=2, K=2, M=8"
  exit 0
fi

if [ "$(uname -s)" = Darwin ] && [ "${M1_CAFFEINATED:-0}" != 1 ] \
  && command -v caffeinate >/dev/null 2>&1; then
  export M1_CAFFEINATED=1 MODEL TIMING_FILE RUN_EPSILON RUN_TESTS
  exec caffeinate -i bash "$SCRIPT_PATH" "$@"
fi

mkdir -p logs
if [ "$RUN_TESTS" = 1 ]; then
  python -c 'import platform, torch; print(platform.platform()); print("MPS:", torch.backends.mps.is_available()); assert torch.backends.mps.is_available()'
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
  NPROC=1 PRECISION=fp16 bash scripts/run_train.sh 1.0 0.0 \
    --model "$MODEL" --run-name "calibration_m1_s1_${started}" --seed 314159 \
    --noise-mode iid --temperature 1.0 --num-generations 2 \
    --prime-max 29 --k-min 2 --k-max 5 --style compact --eval-split eval \
    --max-steps 1 --eval-steps 999999 --train-size 512 --eval-size 8 \
    --no-save "${train_args[@]}"
  elapsed=$(( $(date +%s) - started ))
  if [ "$elapsed" -le 0 ]; then elapsed=1; fi
  printf '%s\n' "$elapsed" > "$TIMING_FILE"
  echo "M1 calibration: one complete cell in ${elapsed}s"
else
  echo "Reusing calibration from $TIMING_FILE (set FORCE_CALIBRATION=1 to refresh)"
fi

if [ "$RUN_EPSILON" = 1 ]; then
  python eval/eval_exploration.py \
    --model "$MODEL" --backend transformers --device mps --generation-batch-size 2 \
    --n-problems 2 --k 2 --reference-samples 8 --resolution-m 2 4 \
    --temperatures 0.7 1.0 1.3 --proposal-temperature 1.3 \
    --mixture-weights 0 0.5 1 --max-tokens 64 --out "$EPSILON_OUT"
  python analysis/plot_core_results.py \
    --exploration "$EPSILON_OUT" --out "$FIGURES_OUT"
fi

echo "M1 Pro smoke test complete"
