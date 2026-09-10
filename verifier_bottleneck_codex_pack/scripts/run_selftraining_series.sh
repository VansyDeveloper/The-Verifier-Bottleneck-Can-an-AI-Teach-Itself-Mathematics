#!/usr/bin/env bash
# D-020: GRPO with an adequately exploring training sampler (T=8.0), then the
# v2 evaluation grid. Single manipulated variable versus the D-018 cold branch.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

START=artifacts/adapters/sft_atomic_r32_pilot_aw4_cont
ARMS=(iid_action@0.7 iid_action@3.0 iid_action@8.0 iid_action@100.0 prefix_balanced_action@0.7)

for seed in 0 1 2; do
  name="grpo_iid_action_hot_400_seed${seed}"
  if [ -d "artifacts/adapters/${name}" ]; then echo "skip ${name}"; continue; fi
  echo "=== ${name} $(date -u +%FT%TZ) ==="
  .venv/bin/python scripts/train_action_grpo.py \
    --method iid_action --adapter "${START}" \
    --input artifacts/data/pilot/rl_train.jsonl \
    --steps 400 --group-size 8 --seed "${seed}" --temperature 8.0 \
    --save-every 50 --output-name "${name}" \
    > "artifacts/selftraining_${name}.stdout.log" 2> "artifacts/selftraining_${name}.stderr.log"
  echo "${name} exit=$?"; tail -c 400 "artifacts/selftraining_${name}.stdout.log"
done

run_one () {  # adapter role seed
  local adapter="$1" role="$2" seed="$3"
  [ -d "${adapter}" ] || { echo "MISSING ${adapter}"; return; }
  local stem="artifacts/eval_v2_${role}_seed${seed}"
  [ -f "${stem}.done" ] && { echo "skip ${stem}"; return; }
  echo "=== v2 ${role} seed${seed} $(date -u +%FT%TZ) ==="
  .venv/bin/python scripts/run_action_search_screen.py \
    --adapter "${adapter}" --input artifacts/data/pilot/confirmatory_heldout_v2.jsonl \
    --k 32 --seed "${seed}" --task-tag confirmatoryv2 --run-method-tag "arms-${role}" \
    --methods "${ARMS[@]}" > "${stem}.stdout.log" 2> "${stem}.stderr.log"
  local status=$?
  [ ${status} -eq 0 ] && touch "${stem}.done" || tail -5 "${stem}.stderr.log"
  echo "${stem} exit=${status}"; grep '^\[' "${stem}.stdout.log" || true
}

for seed in 0 1 2; do
  run_one "${START}" pre "${seed}"
  run_one "artifacts/adapters/grpo_iid_action_400_seed${seed}" coldgrpo "${seed}"
  run_one "artifacts/adapters/grpo_iid_action_hot_400_seed${seed}" hotgrpo "${seed}"
done

echo "SELFTRAINING SERIES DONE $(date -u +%FT%TZ)"
