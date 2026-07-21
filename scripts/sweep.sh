#!/bin/bash
# Full H2 checker grid. Exploration is deliberately fixed across all cells.
# Override ALPHAS/BETAS/SEEDS to run a subset, e.g.
# ALPHAS=1.0 BETAS=0.0 SEEDS=0 STEPS=10 DRY_RUN=1 bash scripts/sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."

read -r -a ALPHA_GRID <<< "${ALPHAS:-1.0 0.8 0.6 0.4 0.2}"
read -r -a BETA_GRID <<< "${BETAS:-0.0 0.2 0.4 0.6 0.8}"
read -r -a SEED_GRID <<< "${SEEDS:-0 1 2}"
STEPS=${STEPS:-600}
TEMPERATURE=${TEMPERATURE:-1.0}
GENERATIONS=${GENERATIONS:-16}
MODEL=${MODEL:-Qwen/Qwen3-0.6B-Base}
TRAIN_SIZE=${TRAIN_SIZE:-20000}
EVAL_SIZE=${EVAL_SIZE:-128}
EVAL_STEPS=${EVAL_STEPS:-25}
PRIME_MAX=${PRIME_MAX:-29}
K_MIN=${K_MIN:-2}
K_MAX=${K_MAX:-5}
STYLE=${STYLE:-compact}
EVAL_SPLIT=${EVAL_SPLIT:-eval}
CONFIG_TAG=${CONFIG_TAG:-}
export NPROC=${NPROC:-1}
mkdir -p logs

hash256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$@"
  else
    shasum -a 256 "$@"
  fi
}

CONFIG_ID=$(
  {
    printf '%s\0' "$MODEL" "$STEPS" "$TEMPERATURE" "$GENERATIONS" "$NPROC"
    printf '%s\0' "$PRIME_MAX" "$K_MIN" "$K_MAX" "$STYLE" "$EVAL_SPLIT"
    printf '%s\0' "$TRAIN_SIZE" "$EVAL_SIZE" "$EVAL_STEPS" "$CONFIG_TAG"
    printf '%s\0' "${SAVE_FINAL:-0}" "$@"
    hash256 scripts/sweep.sh scripts/run_train.sh training/train_grpo.py \
      modcomp/checker.py modcomp/gen.py modcomp/metrics.py pyproject.toml uv.lock
  } | hash256
)
CONFIG_ID=${CONFIG_ID%% *}
CONFIG_ID=${CONFIG_ID:0:10}
echo "CONFIG h2_c${CONFIG_ID} (model=$MODEL, steps=$STEPS, T=$TEMPERATURE, G=$GENERATIONS)"
if [ -n "${CONFIG_OUT:-}" ]; then
  printf '%s\n' "$CONFIG_ID" > "$CONFIG_OUT"
fi

for a in "${ALPHA_GRID[@]}"; do
  for b in "${BETA_GRID[@]}"; do
    for seed in "${SEED_GRID[@]}"; do
      name="h2_c${CONFIG_ID}_n${STEPS}_a${a}_b${b}_t${TEMPERATURE}_g${GENERATIONS}_s${seed}"
      done_file="logs/${name}.done"
      if [ -f "$done_file" ] && [ "${DRY_RUN:-0}" != 1 ]; then
        echo "SKIP $name (completed)"
        continue
      fi
      cmd=(bash scripts/run_train.sh "$a" "$b"
        --model "$MODEL" --run-name "$name" --seed "$seed"
        --noise-mode iid --temperature "$TEMPERATURE" --num-generations "$GENERATIONS"
        --prime-max "$PRIME_MAX" --k-min "$K_MIN" --k-max "$K_MAX"
        --style "$STYLE" --eval-split "$EVAL_SPLIT"
        --max-steps "$STEPS" --eval-steps "$EVAL_STEPS"
        --train-size "$TRAIN_SIZE" --eval-size "$EVAL_SIZE")
      if [ "${SAVE_FINAL:-0}" != 1 ]; then
        cmd+=(--no-save)
      fi
      cmd+=("$@")
      echo "START $name (NPROC=$NPROC)"
      if [ "${DRY_RUN:-0}" = 1 ]; then
        printf ' %q' "${cmd[@]}"
        printf '\n'
        continue
      fi
      "${cmd[@]}" 2>&1 | tee "logs/${name}.log"
      touch "$done_file"
    done
  done
done
echo "SWEEP DONE"
