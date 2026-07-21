#!/bin/bash
# Calibrated native-MPS pilot: five-hour planning target, no runtime kill timer.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SCRIPT_PATH="$SCRIPT_DIR/$(basename "$0")"
cd "$SCRIPT_DIR/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate

MODEL=${MODEL:-Qwen/Qwen3-0.6B-Base}
MODEL_TAG=${MODEL//\//_}
TARGET_SECONDS=${TARGET_SECONDS:-${MAX_SECONDS:-18000}}
MAX_PILOT_STEPS=${MAX_PILOT_STEPS:-10}
RESERVE_SECONDS=${RESERVE_SECONDS:-3600}
RUN_EPSILON=${RUN_EPSILON:-1}
TIMING_FILE=${TIMING_FILE:-logs/m1_${MODEL_TAG}_lora8_g2_c64_s1.txt}
SESSION_TAG=${SESSION_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
H2_DATA_OUT="results/data/pilot5h_m1_${SESSION_TAG}"
FIGURES_OUT="results/figures/pilot5h_m1_${SESSION_TAG}"
EPSILON_OUT="results/data/exploration_5h_m1_${SESSION_TAG}.json"

case "$TARGET_SECONDS:$MAX_PILOT_STEPS:$RESERVE_SECONDS" in
  *[!0-9:]*|:*|*:) echo "time and step limits must be positive integers" >&2; exit 2 ;;
esac
case "$RUN_EPSILON" in 0|1) ;; *) echo "RUN_EPSILON must be 0 or 1" >&2; exit 2 ;; esac
if [ "$TARGET_SECONDS" -le 120 ] || [ "$MAX_PILOT_STEPS" -le 0 ] \
  || [ "$RESERVE_SECONDS" -le 0 ] \
  || [ "$RESERVE_SECONDS" -ge $((TARGET_SECONDS - 120)) ]; then
  echo "require 0 < RESERVE_SECONDS < TARGET_SECONDS-120 and MAX_PILOT_STEPS > 0" >&2
  exit 2
fi

train_args=(
  --lora --lora-r 8 --lora-alpha 16
  --per-device-batch 2 --grad-accum 1
  --max-completion-length 64 --generation-backend transformers --precision fp16
)

if [ "${DRY_RUN:-0}" = 1 ]; then
  MODEL="$MODEL" NPROC=1 PRECISION=fp16 SEEDS="0" STEPS="${STEPS:-1}" \
    GENERATIONS=2 TRAIN_SIZE=512 EVAL_SIZE=8 EVAL_STEPS=999999 \
    CONFIG_TAG=m1_5h_dry DRY_RUN=1 bash scripts/sweep.sh "${train_args[@]}"
  echo "DRY RUN: runtime calibration chooses full 25-cell or 3-cell diagnostic mode"
  echo "DRY RUN: MPS epsilon sweep and three core figures follow H2"
  exit 0
fi

if [ "$(uname -s)" = Darwin ] && [ "${M1_CAFFEINATED:-0}" != 1 ] \
  && command -v caffeinate >/dev/null 2>&1; then
  export M1_CAFFEINATED=1 MODEL TARGET_SECONDS MAX_PILOT_STEPS RESERVE_SECONDS
  export RUN_EPSILON TIMING_FILE SESSION_TAG
  exec caffeinate -i bash "$SCRIPT_PATH" "$@"
fi

started=$(date +%s)
target_deadline=$((started + TARGET_SECONDS))
mkdir -p logs runs/incomplete
python -c 'import torch; assert torch.backends.mps.is_available(), "MPS is unavailable"'

run_epsilon() {
  local profile=${1:-full}
  local epsilon_args=(
    --model "$MODEL" --backend transformers --device mps --generation-batch-size 8
    --n-problems 8 --k 8 --reference-samples 32 --resolution-m 4 8 16
    --temperatures 0.5 0.8 1.0 1.3 --proposal-temperature 1.3
    --mixture-weights 0 0.25 0.5 0.75 1 --max-tokens 96 --out "$EPSILON_OUT"
  )
  if [ "$profile" = small ]; then
    epsilon_args=(
      --model "$MODEL" --backend transformers --device mps --generation-batch-size 2
      --n-problems 2 --k 2 --reference-samples 8 --resolution-m 2 4
      --temperatures 0.8 1.0 1.2 --proposal-temperature 1.2
      --mixture-weights 0 1 --max-tokens 64 --out "$EPSILON_OUT"
    )
  fi
  python eval/eval_exploration.py "${epsilon_args[@]}"
}

plot_results() {
  local plot_args=(--out "$FIGURES_OUT")
  if [ -s "$H2_DATA_OUT/checker_grid_summary.json" ]; then
    plot_args+=(--h2-summary "$H2_DATA_OUT/checker_grid_summary.json")
  fi
  if [ -s "$EPSILON_OUT" ]; then
    plot_args+=(--exploration "$EPSILON_OUT")
  fi
  if [ "${#plot_args[@]}" -eq 2 ]; then
    echo "no collected JSON available for plotting" >&2
    return 0
  fi
  python analysis/plot_core_results.py "${plot_args[@]}"
}

calibration_status=0
if [ ! -s "$TIMING_FILE" ] || [ "${FORCE_CALIBRATION:-0}" = 1 ]; then
  set +e
  MODEL="$MODEL" RUN_EPSILON=0 RUN_TESTS=1 TIMING_FILE="$TIMING_FILE" \
    FORCE_CALIBRATION="${FORCE_CALIBRATION:-0}" \
    bash scripts/smoke_m1.sh
  calibration_status=$?
  set -e
fi
if [ "$calibration_status" -ne 0 ]; then
  echo "MPS calibration failed; attempting only the smallest epsilon run" >&2
  set +e
  run_epsilon small
  epsilon_status=$?
  plot_results
  plot_status=$?
  set -e
  echo "M1 pilot stopped after calibration (calibration=$calibration_status, epsilon=$epsilon_status, plots=$plot_status)"
  exit "$calibration_status"
fi

read -r time_s1 < "$TIMING_FILE"
case "$time_s1" in
  ''|*[!0-9]*) echo "invalid calibration file: $TIMING_FILE" >&2; exit 2 ;;
esac
if [ "$time_s1" -le 0 ]; then echo "calibration time must be positive" >&2; exit 2; fi

effective_reserve=$((RUN_EPSILON * RESERVE_SECONDS))
planning_grid_deadline=$((target_deadline - effective_reserve))
planning_grid_seconds=$((planning_grid_deadline - $(date +%s)))
if [ "$planning_grid_seconds" -le 60 ]; then
  echo "calibration consumed the H2 budget" >&2
  planning_grid_seconds=0
fi

pairs=()
guarded_full_seconds=$((time_s1 * 25 * 14 / 10))
if [ "$guarded_full_seconds" -le "$planning_grid_seconds" ]; then
  mode=full-grid
  for alpha in 1.0 0.8 0.6 0.4 0.2; do
    for beta in 0.0 0.2 0.4 0.6 0.8; do
      pairs+=("$alpha:$beta")
    done
  done
else
  mode=three-cell-diagnostic
  pairs=("1.0:0.0" "0.6:0.6" "0.2:0.8")
fi

if [ -n "${STEPS:-}" ]; then
  case "$STEPS" in ''|*[!0-9]*) echo "STEPS must be a positive integer" >&2; exit 2 ;; esac
  pilot_steps=$STEPS
elif [ "$planning_grid_seconds" -le 0 ]; then
  pilot_steps=1
else
  pilot_steps=$((planning_grid_seconds * 10 / (14 * ${#pairs[@]} * time_s1)))
fi
if [ "$pilot_steps" -lt 1 ]; then pilot_steps=1; fi
if [ "$pilot_steps" -gt "$MAX_PILOT_STEPS" ]; then pilot_steps=$MAX_PILOT_STEPS; fi

echo "M1 five-hour pilot: mode=$mode, cells=${#pairs[@]}, seed=0, steps=$pilot_steps"
echo "Calibration: one complete cell=${time_s1}s; full-grid guarded estimate=${guarded_full_seconds}s"
echo "Scheduling target=${TARGET_SECONDS}s; no artificial runtime stop is enabled"
echo "This is a short-horizon pilot, not the 3-seed x 600-step paper protocol."

config_file="logs/m1_5h_config_id_${SESSION_TAG}.txt"
failure_file="logs/m1_5h_failures_${SESSION_TAG}.tsv"
: > "$config_file"
: > "$failure_file"
completed=0
failures=0

for pair in "${pairs[@]}"; do
  alpha=${pair%%:*}
  beta=${pair##*:}
  set +e
  env MODEL="$MODEL" NPROC=1 PRECISION=fp16 ALPHAS="$alpha" BETAS="$beta" SEEDS=0 \
    STEPS="$pilot_steps" GENERATIONS=2 TRAIN_SIZE=512 EVAL_SIZE=8 \
    EVAL_STEPS=999999 CONFIG_TAG="m1_5h_${SESSION_TAG}" CONFIG_OUT="$config_file" \
    bash scripts/sweep.sh "${train_args[@]}"
  cell_status=$?
  set -e
  if [ "$cell_status" -eq 0 ]; then
    completed=$((completed + 1))
  else
    failures=$((failures + 1))
    printf '%s\t%s\t%s\n' "$alpha" "$beta" "$cell_status" >> "$failure_file"
    if [ -s "$config_file" ]; then
      read -r config_id < "$config_file"
      run_name="h2_c${config_id}_n${pilot_steps}_a${alpha}_b${beta}_t1.0_g2_s0"
      if [ -d "runs/$run_name" ]; then
        mv "runs/$run_name" "runs/incomplete/${run_name}_${SESSION_TAG}_status${cell_status}"
      fi
    fi
  fi
done

expected=${#pairs[@]}
if [ "$completed" -eq "$expected" ]; then
  grid_status=0
else
  grid_status=1
fi
echo "H2 cells completed: $completed/$expected; failed: $failures; mode=$mode"

if [ -s "$config_file" ] && [ "$completed" -gt 0 ]; then
  read -r config_id < "$config_file"
  collector_args=(
    --run-prefix "h2_c${config_id}_" --expected-seeds 1 --allow-short-horizon
    --data-out "$H2_DATA_OUT"
  )
  if [ "$completed" -lt 25 ]; then collector_args+=(--allow-partial-grid); fi
  set +e
  python analysis/h2_checker_noise/collect_results.py "${collector_args[@]}"
  collector_status=$?
  set -e
  if [ "$collector_status" -ne 0 ] && [ "$grid_status" -eq 0 ]; then
    grid_status=$collector_status
  fi
fi

epsilon_status=0
if [ "$RUN_EPSILON" = 1 ]; then
  set +e
  run_epsilon full
  epsilon_status=$?
  set -e
fi

set +e
plot_results
plot_status=$?
set -e

elapsed=$(( $(date +%s) - started ))
echo "M1 pilot finished in ${elapsed}s (mode=$mode, grid=$grid_status, epsilon=$epsilon_status, plots=$plot_status)"
if [ "$grid_status" -ne 0 ]; then exit "$grid_status"; fi
if [ "$epsilon_status" -ne 0 ]; then exit "$epsilon_status"; fi
if [ "$plot_status" -ne 0 ]; then exit "$plot_status"; fi
