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

if [ "$(uname -s)" = Darwin ]; then
  if [ "${NPROC:-1}" != 1 ]; then
    echo "Apple MPS supports only NPROC=1 in this launcher" >&2
    exit 2
  fi
  export PYTORCH_ENABLE_MPS_FALLBACK=${PYTORCH_ENABLE_MPS_FALLBACK:-1}
  python training/train_grpo.py --alpha "$ALPHA" --beta "$BETA" \
    --generation-backend transformers --precision "${PRECISION:-auto}" "$@"
else
  accelerate launch --num_processes "${NPROC:-8}" \
    --mixed_precision "${MIXED_PRECISION:-bf16}" training/train_grpo.py \
    --alpha "$ALPHA" --beta "$BETA" --precision "${PRECISION:-auto}" "$@"
fi
