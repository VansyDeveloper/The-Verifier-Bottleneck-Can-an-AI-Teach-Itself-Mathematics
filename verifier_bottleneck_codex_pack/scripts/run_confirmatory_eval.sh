#!/usr/bin/env bash
# D-018 evaluation grid.
#
#   confirmatory_heldout.jsonl (300 tasks, never opened): full factorial
#     {pre-GRPO, iid-GRPO, prefix-GRPO} x {A1..A5} x seeds {0,1,2}
#   final_like_heldout.jsonl (200 tasks, already opened in July): continuity
#     only, A1 and A4, reported as re-analysis
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARMS=(iid_action@0.7 iid_action@2.0 iid_action@3.0 prefix_balanced_action@0.7 temp_mix_action@0.2,0.6,1.0,1.4)
CONTINUITY=(iid_action@0.7 prefix_balanced_action@0.7)

run_one () {  # adapter role seed input tag arms...
  local adapter="$1" role="$2" seed="$3" input="$4" tag="$5"; shift 5
  if [ ! -d "${adapter}" ]; then echo "MISSING adapter ${adapter} - skipped"; return; fi
  local stem="artifacts/eval_${tag}_${role}_seed${seed}"
  if [ -f "${stem}.done" ]; then echo "skip ${stem}"; return; fi
  echo "=== ${tag} ${role} seed${seed} $(date -u +%FT%TZ) ==="
  .venv/bin/python scripts/run_action_search_screen.py \
    --adapter "${adapter}" --input "${input}" --k 32 --seed "${seed}" \
    --task-tag "${tag}" --run-method-tag "arms-${role}" \
    --methods "$@" > "${stem}.stdout.log" 2> "${stem}.stderr.log"
  local status=$?
  if [ ${status} -eq 0 ]; then touch "${stem}.done"; else tail -5 "${stem}.stderr.log"; fi
  echo "${stem} exit=${status}"
  grep '^\[' "${stem}.stdout.log" || true
}

for seed in 0 1 2; do
  run_one artifacts/adapters/sft_atomic_r32_pilot_aw4_cont pre "${seed}" \
    artifacts/data/pilot/confirmatory_heldout.jsonl confirmatory "${ARMS[@]}"
  run_one "artifacts/adapters/grpo_iid_action_400_seed${seed}" iidgrpo "${seed}" \
    artifacts/data/pilot/confirmatory_heldout.jsonl confirmatory "${ARMS[@]}"
  run_one "artifacts/adapters/grpo_prefix_balanced_action_400_seed${seed}" prefixgrpo "${seed}" \
    artifacts/data/pilot/confirmatory_heldout.jsonl confirmatory "${ARMS[@]}"
done

for seed in 0 1 2; do
  run_one artifacts/adapters/sft_atomic_r32_pilot_aw4_cont pre "${seed}" \
    artifacts/data/pilot/final_like_heldout.jsonl july "${CONTINUITY[@]}"
  run_one "artifacts/adapters/grpo_iid_action_400_seed${seed}" iidgrpo "${seed}" \
    artifacts/data/pilot/final_like_heldout.jsonl july "${CONTINUITY[@]}"
  run_one "artifacts/adapters/grpo_prefix_balanced_action_400_seed${seed}" prefixgrpo "${seed}" \
    artifacts/data/pilot/final_like_heldout.jsonl july "${CONTINUITY[@]}"
done

echo "ALL EVAL DONE $(date -u +%FT%TZ)"
