#!/usr/bin/env bash
# D-018 Phase G: 400-step GRPO, seeds 0/1/2, both branches, identical start point.
# Sequential because the 4 GB card holds exactly one run.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

START=artifacts/adapters/sft_atomic_r32_pilot_aw4_cont
LOG=artifacts/confirmatory_grpo

for seed in 0 1 2; do
  for method in iid_action prefix_balanced_action; do
    name="grpo_${method}_400_seed${seed}"
    if [ -d "artifacts/adapters/${name}" ]; then
      echo "skip ${name} (already present)"
      continue
    fi
    echo "=== ${name} $(date -u +%FT%TZ) ==="
    .venv/bin/python scripts/train_action_grpo.py \
      --method "${method}" --adapter "${START}" \
      --input artifacts/data/pilot/rl_train.jsonl \
      --steps 400 --group-size 8 --seed "${seed}" \
      --save-every 50 --output-name "${name}" \
      > "${LOG}_${name}.stdout.log" 2> "${LOG}_${name}.stderr.log"
    status=$?
    echo "${name} exit=${status}"
    tail -c 400 "${LOG}_${name}.stdout.log"
    [ ${status} -ne 0 ] && tail -5 "${LOG}_${name}.stderr.log"
  done
done
echo "ALL GRPO DONE $(date -u +%FT%TZ)"
