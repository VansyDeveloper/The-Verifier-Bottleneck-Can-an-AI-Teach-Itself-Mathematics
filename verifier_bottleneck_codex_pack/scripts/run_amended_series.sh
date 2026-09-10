#!/usr/bin/env bash
# D-021 completion + D-022 motif series, under Amendment 001.
#
# Order matters: the two missing D-021 adapters first (they finish the already
# preregistered grid), then the motif-held adapters, then all exhaustive ranking.
# Every step is skipped if its output already exists, so this is safe to re-run
# after an interruption.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

START=artifacts/adapters/sft_atomic_r32_pilot_aw4_cont

train_oracle () {  # name train_file
  local name="$1" train_file="$2"
  [ -d "artifacts/adapters/${name}" ] && { echo "skip ${name}"; return; }
  local seed="${name##*seed}"
  echo "=== train ${name} $(date -u +%FT%TZ) ==="
  .venv/bin/python scripts/train_sft.py --config configs/training/sft_atomic.yaml \
    --tier pilot --seed "${seed}" --max-steps 1875 \
    --lora-config configs/training/lora_r32.yaml \
    --apply-weight 0 --micro-batch-size 2 --allow-composition \
    --base-adapter "${START}" --extra-train-file "${train_file}" \
    --output-name "${name}" \
    > "artifacts/amended_${name}.stdout.log" 2> "artifacts/amended_${name}.stderr.log"
  echo "${name} exit=$?"; tail -c 300 "artifacts/amended_${name}.stdout.log"
}

for seed in 1 2; do
  train_oracle "sft_composition_oracle_seed${seed}" artifacts/data/pilot/sft_train_composition.jsonl
done
for seed in 0 1 2; do
  train_oracle "sft_composition_nomotif_seed${seed}" artifacts/data/pilot/sft_train_composition_nomotif.jsonl
done

rank () {  # adapter role split
  local adapter="$1" role="$2" split="$3"
  [ -d "${adapter}" ] || { echo "MISSING ${adapter}"; return; }
  local stem="artifacts/rank_${split}_${role}"
  [ -f "${stem}.done" ] && { echo "skip ${stem}"; return; }
  echo "=== rank ${split} ${role} $(date -u +%FT%TZ) ==="
  .venv/bin/python scripts/run_exhaustive_ranking.py \
    --adapter "${adapter}" --input "artifacts/data/pilot/${split}.jsonl" \
    --task-tag "${split}" --run-method-tag "rank-${role}" \
    > "${stem}.stdout.log" 2> "${stem}.stderr.log"
  local status=$?
  [ ${status} -eq 0 ] && touch "${stem}.done" || tail -4 "${stem}.stderr.log"
  echo "${stem} exit=${status}"
}

# D-021 capacity grid, confound-free metric. Depth-2 sets first: they are ~5x
# cheaper, so an early interruption still leaves a complete depth-2 story.
for split in capacity_d2_train capacity_d2_heldout capacity_d3_train capacity_d3_heldout; do
  rank "${START}" atomic "${split}"
  for seed in 0 1 2; do
    rank "artifacts/adapters/sft_composition_oracle_seed${seed}" "oracle${seed}" "${split}"
  done
done

# D-022 motif grid: known/held-out field x seen/held motif.
for split in motif_a_d3 motif_b_d3 motif_c_d3 motif_d_d3; do
  rank "${START}" atomic "${split}"
  for seed in 0 1 2; do
    rank "artifacts/adapters/sft_composition_nomotif_seed${seed}" "nomotif${seed}" "${split}"
  done
done

echo "AMENDED SERIES DONE $(date -u +%FT%TZ)"
