#!/usr/bin/env bash
# D-021: composition supervision as a capacity ceiling, plus the 2x2 grid that
# separates program depth from held-out primes.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

START=artifacts/adapters/sft_atomic_r32_pilot_aw4_cont
ARMS=(iid_action@0.7 iid_action@3.0 iid_action@8.0)
SETS=(capacity_d2_train capacity_d3_train capacity_d2_heldout capacity_d3_heldout)

for seed in 0 1 2; do
  name="sft_composition_oracle_seed${seed}"
  if [ -d "artifacts/adapters/${name}" ]; then echo "skip ${name}"; continue; fi
  echo "=== ${name} $(date -u +%FT%TZ) ==="
  .venv/bin/python scripts/train_sft.py --config configs/training/sft_atomic.yaml \
    --tier pilot --seed "${seed}" --max-steps 1875 \
    --lora-config configs/training/lora_r32.yaml \
    --apply-weight 0 --micro-batch-size 2 --allow-composition \
    --base-adapter "${START}" \
    --extra-train-file artifacts/data/pilot/sft_train_composition.jsonl \
    --output-name "${name}" \
    > "artifacts/capacity_${name}.stdout.log" 2> "artifacts/capacity_${name}.stderr.log"
  echo "${name} exit=$?"; tail -c 350 "artifacts/capacity_${name}.stdout.log"
done

run_one () {  # adapter role seed split
  local adapter="$1" role="$2" seed="$3" split="$4"
  [ -d "${adapter}" ] || { echo "MISSING ${adapter}"; return; }
  local stem="artifacts/eval_${split}_${role}_seed${seed}"
  [ -f "${stem}.done" ] && { echo "skip ${stem}"; return; }
  echo "=== ${split} ${role} seed${seed} $(date -u +%FT%TZ) ==="
  .venv/bin/python scripts/run_action_search_screen.py \
    --adapter "${adapter}" --input "artifacts/data/pilot/${split}.jsonl" \
    --k 32 --seed "${seed}" --task-tag "${split}" --run-method-tag "arms-${role}" \
    --methods "${ARMS[@]}" > "${stem}.stdout.log" 2> "${stem}.stderr.log"
  local status=$?
  [ ${status} -eq 0 ] && touch "${stem}.done" || tail -5 "${stem}.stderr.log"
  echo "${stem} exit=${status}"
  return "${status}"
}

for split in "${SETS[@]}"; do
  for seed in 0 1 2; do
    run_one "${START}" atomic "${seed}" "${split}" || exit $?
    run_one "artifacts/adapters/sft_composition_oracle_seed${seed}" oracle "${seed}" "${split}" || exit $?
  done
done

bash scripts/run_composition_forgetting.sh

echo "CAPACITY SERIES DONE $(date -u +%FT%TZ)"
