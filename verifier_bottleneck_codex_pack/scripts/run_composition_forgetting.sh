#!/usr/bin/env bash
# Greedy atomic checks for the shared baseline and all D-021/D-022 adapters.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

adapter_for () {
  case "$1" in
    atomic) echo artifacts/adapters/sft_atomic_r32_pilot_aw4_cont ;;
    oracle*) echo "artifacts/adapters/sft_composition_oracle_seed${1#oracle}" ;;
    nomotif*) echo "artifacts/adapters/sft_composition_nomotif_seed${1#nomotif}" ;;
  esac
}

for role in atomic oracle0 oracle1 oracle2 nomotif0 nomotif1 nomotif2; do
  adapter=$(adapter_for "${role}")
  for split in sft_validation_apply_non_sh1 sft_validation_plan; do
    stem="artifacts/eval_composition_forget_${role}_${split}"
    [ -f "${stem}.done" ] && { echo "skip ${stem}"; continue; }
    .venv/bin/python scripts/evaluate_model.py \
      --config configs/evaluation/baseline.yaml \
      --adapter "${adapter}" --input "artifacts/data/pilot/${split}.jsonl" \
      --k 1 --seed 0 --temperature 0.0 --max-new-tokens 24 \
      --label "composition-forget-${role}-${split}" \
      > "${stem}.stdout.log" 2> "${stem}.stderr.log"
    status=$?
    if [ "${status}" -ne 0 ]; then
      tail -20 "${stem}.stderr.log"
      exit "${status}"
    fi
    touch "${stem}.done"
    echo "${stem} done"
  done
done
