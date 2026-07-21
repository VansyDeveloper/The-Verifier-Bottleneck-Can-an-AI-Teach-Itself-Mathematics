#!/bin/bash
# Time-planned pilot: every H2 alpha-beta cell once, then epsilon measurement.
# This is a 25-cell/one-seed pilot, not the 75-run paper protocol.
# The five-hour budget sizes the run but never kills an active process.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate

MODEL=${MODEL:-Qwen/Qwen3-0.6B-Base}
MAX_SECONDS=${MAX_SECONDS:-18000}
MAX_PILOT_STEPS=${MAX_PILOT_STEPS:-25}
RESERVE_SECONDS=${RESERVE_SECONDS:-3300}
RUN_EPSILON=${RUN_EPSILON:-1}
MODEL_TAG=${MODEL//\//_}
PROFILE_TAG=g2_e32_lora8_c128
if [ "${SLEEP_MODE:-0}" = 1 ]; then PROFILE_TAG=${PROFILE_TAG}_sleep; fi
TIMING_S1=${TIMING_S1:-logs/rtx5070_${MODEL_TAG}_${PROFILE_TAG}_s1.txt}
TIMING_S5=${TIMING_S5:-logs/rtx5070_${MODEL_TAG}_${PROFILE_TAG}_s5.txt}
SESSION_TAG=${SESSION_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
H2_DATA_OUT="results/data/pilot5h_${SESSION_TAG}"
FIGURES_OUT="results/figures/pilot5h_${SESSION_TAG}"
EPSILON_OUT="results/data/exploration_5h_5070_${SESSION_TAG}.json"

case "$MAX_SECONDS:$MAX_PILOT_STEPS:$RESERVE_SECONDS" in
  *[!0-9:]*|:*|*:) echo "time and step limits must be positive integers" >&2; exit 2 ;;
esac
case "$RUN_EPSILON" in 0|1) ;; *) echo "RUN_EPSILON must be 0 or 1" >&2; exit 2 ;; esac
if [ "$MAX_SECONDS" -le 120 ] || [ "$MAX_PILOT_STEPS" -le 0 ] \
  || [ "$RESERVE_SECONDS" -le 0 ] \
  || [ "$RESERVE_SECONDS" -ge $((MAX_SECONDS - 120)) ]; then
  echo "require 0 < RESERVE_SECONDS < MAX_SECONDS-120 and MAX_PILOT_STEPS > 0" >&2
  exit 2
fi

train_args=(
  --lora --lora-r 8 --lora-alpha 16
  --per-device-batch 2 --grad-accum 1
  --max-completion-length 128 --vllm-gpu-memory-utilization 0.20
)
if [ "${SLEEP_MODE:-0}" = 1 ]; then
  train_args+=(--vllm-enable-sleep-mode)
fi

if [ "${DRY_RUN:-0}" = 1 ]; then
  MODEL="$MODEL" NPROC=1 SEEDS="0" STEPS="${STEPS:-5}" GENERATIONS=2 \
    TRAIN_SIZE=2000 EVAL_SIZE=32 EVAL_STEPS=999999 \
    CONFIG_TAG=rtx5070_5h_dry DRY_RUN=1 \
    bash scripts/sweep.sh "${train_args[@]}"
  echo "DRY RUN: epsilon sweep would run after all 25 H2 cells"
  exit 0
fi

started=$(date +%s)
target_deadline=$((started + MAX_SECONDS))
mkdir -p logs runs/incomplete

run_epsilon() {
  local profile=${1:-full}
  local epsilon_args=(
    --model "$MODEL" --n-problems 8 --k 8 --reference-samples 32
    --resolution-m 4 8 16 --temperatures 0.5 0.8 1.0 1.3
    --proposal-temperature 1.3 --mixture-weights 0 0.25 0.5 0.75 1
    --max-tokens 128 --gpu-memory-utilization 0.70
    --out "$EPSILON_OUT"
  )
  if [ "$profile" = small ]; then
    epsilon_args=(
      --model "$MODEL" --n-problems 4 --k 4 --reference-samples 16
      --resolution-m 4 8 --temperatures 0.7 1.0 1.3
      --proposal-temperature 1.3 --mixture-weights 0 0.5 1
      --max-tokens 128 --gpu-memory-utilization 0.70
      --out "$EPSILON_OUT"
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
if [ ! -s "$TIMING_S1" ] || [ "${FORCE_CALIBRATION:-0}" = 1 ]; then
  set +e
  MODEL="$MODEL" RUN_EPSILON=0 TIMING_FILE="$TIMING_S1" CALIBRATION_STEPS=1 \
    FORCE_CALIBRATION="${FORCE_CALIBRATION:-0}" \
    bash scripts/smoke_5070.sh
  calibration_status=$?
  set -e
fi
if [ "$calibration_status" -eq 0 ] \
  && { [ ! -s "$TIMING_S5" ] || [ "${FORCE_CALIBRATION:-0}" = 1 ]; }; then
  set +e
  MODEL="$MODEL" RUN_EPSILON=0 RUN_TESTS=0 TIMING_FILE="$TIMING_S5" \
    CALIBRATION_STEPS=5 FORCE_CALIBRATION="${FORCE_CALIBRATION:-0}" \
    bash scripts/smoke_5070.sh
  calibration_status=$?
  set -e
fi
if [ "$calibration_status" -ne 0 ]; then
  echo "calibration failed; H2 is infeasible, attempting the smallest epsilon run" >&2
  set +e
  run_epsilon small
  epsilon_status=$?
  set -e
  set +e
  plot_results
  plot_status=$?
  set -e
  echo "Five-hour pilot stopped after calibration (calibration=$calibration_status, epsilon=$epsilon_status, plots=$plot_status)"
  exit "$calibration_status"
fi

read -r time_s1 < "$TIMING_S1"
read -r time_s5 < "$TIMING_S5"
case "$time_s1:$time_s5" in
  *[!0-9:]*|:*|*:) echo "invalid calibration files" >&2; exit 2 ;;
esac
if [ "$time_s1" -le 0 ] || [ "$time_s5" -le 0 ]; then
  echo "calibration times must be positive" >&2
  exit 2
fi

# t(s) ~= fixed + slope*s. The factor 1.4 is a guard for cell-to-cell variance.
delta=$((time_s5 - time_s1))
if [ "$delta" -le 0 ]; then
  seconds_per_step=$(((time_s5 + 4) / 5))
  fixed_seconds=0
else
  seconds_per_step=$(((delta + 3) / 4))
  fixed_seconds=$((time_s1 - seconds_per_step))
  if [ "$fixed_seconds" -lt 0 ]; then fixed_seconds=0; fi
fi
effective_reserve=$((RUN_EPSILON * RESERVE_SECONDS))
grid_seconds=$((target_deadline - effective_reserve - $(date +%s)))
if [ "$grid_seconds" -le 60 ]; then
  echo "calibration consumed the planned grid budget; using a one-minute sizing floor" >&2
  grid_seconds=60
fi
safe_seconds_per_cell=$((grid_seconds * 10 / 14 / 25))

if [ -n "${STEPS:-}" ]; then
  case "$STEPS" in *[!0-9]*|'') echo "STEPS must be a positive integer" >&2; exit 2 ;; esac
  pilot_steps=$STEPS
elif [ "$safe_seconds_per_cell" -le "$fixed_seconds" ]; then
  pilot_steps=1
else
  pilot_steps=$(((safe_seconds_per_cell - fixed_seconds) / seconds_per_step))
fi
if [ "$pilot_steps" -lt 1 ]; then pilot_steps=1; fi
if [ "$pilot_steps" -gt "$MAX_PILOT_STEPS" ]; then pilot_steps=$MAX_PILOT_STEPS; fi

echo "Five-hour-target pilot: 25 cells, seed=0, steps=$pilot_steps; no runtime kill timer"
echo "Calibration: s1=${time_s1}s, s5=${time_s5}s, fixed~${fixed_seconds}s, step~${seconds_per_step}s"
if [ "$pilot_steps" -lt 3 ]; then
  echo "WARNING: fewer than 3 steps only validate the pipeline, not an H2 effect" >&2
fi

alphas=(1.0 0.8 0.6 0.4 0.2)
betas=(0.0 0.2 0.4 0.6 0.8)
config_file=logs/rtx5070_5h_config_id.txt
failure_file="logs/rtx5070_5h_failures_${SESSION_TAG}.tsv"
: > "$config_file"
: > "$failure_file"
cell_index=0
completed=0
failures=0

for alpha in "${alphas[@]}"; do
  for beta in "${betas[@]}"; do
    set +e
    env MODEL="$MODEL" NPROC=1 ALPHAS="$alpha" BETAS="$beta" SEEDS="0" \
      STEPS="$pilot_steps" GENERATIONS=2 TRAIN_SIZE=2000 EVAL_SIZE=32 \
      EVAL_STEPS=999999 CONFIG_TAG="rtx5070_5h_${SESSION_TAG}" \
      CONFIG_OUT="$config_file" \
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
    cell_index=$((cell_index + 1))
  done
done

if [ "$completed" -eq 25 ]; then
  grid_status=0
else
  grid_status=1
fi
echo "H2 cells completed: $completed/25; failed: $failures; attempted: $cell_index/25"

if [ -s "$config_file" ] && [ "$completed" -gt 0 ]; then
  read -r config_id < "$config_file"
  collector_args=(
    --run-prefix "h2_c${config_id}_" --expected-seeds 1
    --allow-short-horizon --allow-missing-forgetting
    --data-out "$H2_DATA_OUT"
  )
  if [ "$completed" -lt 25 ]; then collector_args+=(--allow-partial-grid); fi
  set +e
  python analysis/h2_checker_noise/collect_results.py "${collector_args[@]}"
  collector_status=$?
  set -e
  if [ "$collector_status" -ne 0 ]; then
    echo "collector failed with status $collector_status" >&2
    if [ "$grid_status" -eq 0 ]; then grid_status=$collector_status; fi
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
echo "Five-hour pilot finished in ${elapsed}s (grid=$grid_status, epsilon=$epsilon_status, plots=$plot_status)"
if [ "$grid_status" -ne 0 ]; then exit "$grid_status"; fi
if [ "$epsilon_status" -ne 0 ]; then exit "$epsilon_status"; fi
if [ "$plot_status" -ne 0 ]; then exit "$plot_status"; fi
