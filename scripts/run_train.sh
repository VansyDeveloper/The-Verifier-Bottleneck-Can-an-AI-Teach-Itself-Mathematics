#!/bin/bash
# Usage: bash scripts/run_train.sh <alpha> <beta> [extra train_grpo.py args...]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate

# Optional location of CUDA compatibility libraries for older host drivers.
if [ -n "${CUDA_COMPAT_PATH:-}" ]; then
  export LD_LIBRARY_PATH="${CUDA_COMPAT_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

ALPHA=${1:-1.0}
BETA=${2:-0.0}
shift 2 || true

accelerate launch --num_processes ${NPROC:-8} --mixed_precision bf16 training/train_grpo.py \
  --alpha "$ALPHA" --beta "$BETA" "$@"
