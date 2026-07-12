#!/bin/bash
# Sweep the checker dials (alpha, beta) sequentially for the H2 / G2 phase diagram.
# Each run trains from the base model with a checker of the given (alpha, beta);
# true_accuracy (reward weight 0) is logged every eval so we can chart
# {collapse, sharpen} vs checker informativeness. Usage: bash sweep.sh
set -uo pipefail
cd "$(dirname "$0")/.."

STEPS=250

# (alpha beta): perfect -> noisier-but-informative -> break-even (I(c;V)=0) -> inverted.
# At alpha==beta the checker carries no information -> expect stagnation/collapse (H2).
PAIRS=(
  "1.0 0.0"
  "0.9 0.1"
  "0.8 0.2"
  "0.7 0.3"
  "0.6 0.4"
  "0.5 0.5"
  "0.4 0.6"
)

for pair in "${PAIRS[@]}"; do
  set -- $pair
  a=$1; b=$2
  echo "==================  alpha=$a beta=$b  =================="
  bash scripts/run_train.sh "$a" "$b" \
    --prime-max 29 --no-save \
    --max-steps "$STEPS" --eval-steps 25 --save-steps 1000000 \
    --train-size 20000 --eval-size 128 \
    2>&1 | tee "logs/sweep_a${a}_b${b}.log"
done
echo "SWEEP DONE"
