#!/bin/bash
# Substantive M1 diagnostic: three checker regimes, three seeds, full curves.
set -euo pipefail

export PILOT_MODE=${PILOT_MODE:-three-cell-diagnostic}
export SEEDS=${SEEDS:-"0 1 2"}
export STEPS=${STEPS:-60}
export MAX_PILOT_STEPS=${MAX_PILOT_STEPS:-$STEPS}
export GENERATIONS=${GENERATIONS:-8}
export PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-8}
export MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-128}
export PRIME_MAX=${PRIME_MAX:-13}
export K_MIN=${K_MIN:-2}
export K_MAX=${K_MAX:-2}
export EVAL_STEPS=${EVAL_STEPS:-15}
export CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-5}
export SAVE_FINAL=${SAVE_FINAL:-1}
export RUN_EPSILON=${RUN_EPSILON:-0}
export TARGET_SECONDS=${TARGET_SECONDS:-43200}

exec bash "$(dirname "$0")/run_5h_m1.sh" "$@"
