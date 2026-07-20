#!/bin/bash
# Phase-3 cell runner, n~U[2,5], p<=29, fast config (<=6h total).
# Usage: bash phase3_cells.sh <cells-file>
# cells-file lines: "tag temp gens alpha beta"  (grad-accum fixed at 1:
# equal compute/step, budget = attempts-per-problem breadth at 128 gens/step)
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CELLS_FILE=${1:?"usage: bash phase3_cells.sh configs/cells_<name>.txt"}
COMMON="--model Qwen/Qwen3-0.6B-Base --style compact --prime-max 29 --k-min 2 --k-max 5 --eval-split eval \
  --kl-beta 0.0 --per-device-batch 16 --max-completion-length 384 \
  --max-steps 150 --eval-steps 50 --eval-size 128 --train-size 20000"

steps_of() { ls runs/$1/completions/completions_0*.parquet 2>/dev/null | wc -l; }

while read -r tag temp gens a b extra; do
  [ -z "$tag" ] && continue
  name="ph3n25p29_${tag}_a${a}_b${b}"
  attempt=0
  while [ "$(steps_of $name)" -lt 145 ] && [ $attempt -lt 4 ]; do
    attempt=$((attempt+1))
    [ -d "runs/$name" ] && rm -rf "runs/$name"
    echo "$(date +%H:%M) START $name attempt=$attempt"
    bash scripts/run_train.sh $a $b --temperature $temp \
      --num-generations $gens --grad-accum 1 $COMMON $extra \
      --run-name "$name" > "logs_$name.txt" 2>&1
    echo "$(date +%H:%M) END $name -> $(steps_of $name) steps"
  done
done < "$CELLS_FILE"
echo "=== TRAINING CELLS DONE ($CELLS_FILE) ==="

mkdir -p results_remote/ph3n25p29
export CUDA_VISIBLE_DEVICES=0
EVAL="--k 64 --n-problems 200 --temperature 1.0 --k-min 2 --k-max 5 --prime-max 29 --style compact --max-tokens 384"

# baseline eval: server B (cells_b) does 'eval' split, server A does 'train'
case "$CELLS_FILE" in
  *cells_b*) BSPLIT=eval ;;
  *) BSPLIT=train ;;
esac
if [ ! -f "results_remote/ph3n25p29/base_0.6B_${BSPLIT}.json" ]; then
  echo "$(date +%H:%M) EVAL base split=$BSPLIT"
  python eval/eval_passk.py --model Qwen/Qwen3-0.6B-Base $EVAL --split $BSPLIT \
    --out "results_remote/ph3n25p29/base_0.6B_${BSPLIT}.json" > "logs_eval_base_${BSPLIT}.txt" 2>&1
fi

while read -r tag temp gens a b extra; do
  [ -z "$tag" ] && continue
  name="ph3n25p29_${tag}_a${a}_b${b}"
  for split in eval train; do
    if [ -d "runs/$name/final" ] && [ ! -f "results_remote/ph3n25p29/${name}_${split}.json" ]; then
      echo "$(date +%H:%M) EVAL $name split=$split"
      python eval/eval_passk.py --model "runs/$name/final" $EVAL --split $split \
        --out "results_remote/ph3n25p29/${name}_${split}.json" \
        > "logs_eval_${name}_${split}.txt" 2>&1
    fi
  done
done < "$CELLS_FILE"
echo "=== ALL DONE ($CELLS_FILE) ==="
