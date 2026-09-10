#!/usr/bin/env bash
# docs/07 section 7.8: atomic forgetting after 400-step GRPO.
# Greedy pass@1 on the frozen atomic validation splits, for the shared starting
# checkpoint and for all six trained adapters.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

evaluate () {  # adapter label
  local adapter="$1" label="$2"
  [ -d "${adapter}" ] || { echo "MISSING ${adapter}"; return; }
  for split in sft_validation_apply_non_sh1 sft_validation_plan; do
    .venv/bin/python scripts/evaluate_model.py \
      --config configs/evaluation/baseline.yaml --adapter "${adapter}" \
      --input "artifacts/data/pilot/${split}.jsonl" --k 1 --seed 0 \
      --temperature 0.0 --max-new-tokens 24 --label "forget-${label}" 2>/dev/null \
      | .venv/bin/python -c "import json,sys; d=json.load(sys.stdin); print(f\"${label}\t${split}\tpass@1={d['pass_at_1']:.4f}\tparse={d['parse_rate']:.3f}\tn={d['tasks']}\")"
  done
}

evaluate artifacts/adapters/sft_atomic_r32_pilot_aw4_cont pre-grpo
for seed in 0 1 2; do
  evaluate "artifacts/adapters/grpo_iid_action_400_seed${seed}" "iid-seed${seed}"
  evaluate "artifacts/adapters/grpo_prefix_balanced_action_400_seed${seed}" "prefix-seed${seed}"
done
