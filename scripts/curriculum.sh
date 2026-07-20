#!/bin/bash
# Curriculum with a PERFECT judge (alpha=1, beta=0): start on small moduli the
# base model can already partly solve, then raise the modulus ceiling stage by
# stage, each stage resuming the WEIGHTS of the previous one. The point is to
# reach moduli that RL-from-base cannot crack cold (base pass@1 ~= 0), and to
# show how good true_accuracy gets when the checker is ideal. Usage: bash curriculum.sh
#
# Disk-frugal: --save-steps is set beyond --max-steps so no optimizer checkpoint
# is written mid-run; only trainer.save_model(final) lands (~3.4GB weights). The
# shared fuse mount runs near-full, and a fat 26GB run (weights+optimizer) is what
# killed an earlier attempt mid-write. Fail-fast: a stage that fails to produce
# its `final` weights aborts the ladder instead of cascading HF repo-id errors.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
mkdir -p logs

CEILINGS=(13 23 37 59 97)   # difficulty ladder: max prime per stage
STEPS=150
SEED=0
BASE="Qwen/Qwen3-1.7B-Base"

prev_final=""
for pmax in "${CEILINGS[@]}"; do
  run="curric_p${pmax}_s${SEED}"
  final="${ROOT}/runs/${run}/final"
  # Source weights: previous stage's final (absolute), or the base model cold.
  src="$BASE"
  [ -n "$prev_final" ] && src="$prev_final"

  if [ -d "$final" ]; then
    echo "==================  curriculum stage prime_max=$pmax  (SKIP: $final exists)  =================="
    prev_final="$final"
    continue
  fi

  echo "==================  curriculum stage prime_max=$pmax  (from $src)  =================="
  bash scripts/run_train.sh 1.0 0.0 \
    --model "$src" --run-name "$run" --prime-max "$pmax" \
    --max-steps "$STEPS" --eval-steps 25 --save-steps 1000000 \
    --train-size 20000 --eval-size 128 \
    2>&1 | tee "logs/${run}.log"

  if [ ! -d "$final" ]; then
    echo "CURRICULUM ABORT: stage prime_max=$pmax did not produce $final (see logs/${run}.log)"
    exit 1
  fi
  prev_final="$final"
done
echo "CURRICULUM DONE"
