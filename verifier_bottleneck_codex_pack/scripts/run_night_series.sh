#!/usr/bin/env bash
# Post-hoc control series for the paper. Unattended, resumable, ordered by value.
#
# Nothing here is confirmatory. Every evaluation set is already open except the
# newly generated capacity_d4_train, so these are disclosed post-hoc controls whose
# job is to remove alternative explanations of results that are already reported.
#
# Stages, in the order they run:
#   S0  back up artifacts/adapters (they are gitignored and were lost once already)
#   S1  mode-aware re-grade of the stored APPLY retention generations   [no GPU]
#   S2  continuous ranking endpoints from data already on disk          [no GPU]
#   S3  oracle adapters on motif_b / motif_d - the missing D-022 positive control
#   S4  equal-budget atomic control - de-confounds C1/C2 from "1875 more steps"
#   S5  composition SFT with APPLY retained - tests the cause of the mode collapse
#   S6  depth-4 exhaustive ranking - the depth axis currently rests on one point
#   S7  degraded verifier on the distillation labels - the project's titular axis
#
# Every step is skipped when its marker exists, so re-running continues where it
# stopped. analyze_night_series.py runs after each GPU stage, so partial results
# are always readable. The frozen registered reports are never regenerated.
set -u
cd "$(dirname "$0")/.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

PY=.venv/bin/python
START=artifacts/adapters/sft_atomic_r32_pilot_aw4_cont
NIGHT=artifacts/night
BACKUP_DIR="${VB_BACKUP_DIR:-$HOME/vb_adapter_backup}"
MIN_FREE_GB="${VB_MIN_FREE_GB:-8}"
mkdir -p "${NIGHT}"
LOG="${NIGHT}/night.log"

say () { echo "[$(date -u +%FT%TZ)] $*" | tee -a "${LOG}"; }

free_gb () { df -BG --output=avail . | tail -1 | tr -dc '0-9'; }

guard_disk () {
  local free; free=$(free_gb)
  if [ "${free}" -lt "${MIN_FREE_GB}" ]; then
    say "STOP: only ${free} GB free, need ${MIN_FREE_GB} GB. Nothing further will run."
    exit 3
  fi
}

# Optimiser checkpoints are 162 MB each and exist only to resume training. The
# adapter is already copied to artifacts/adapters by then, so keep the final one
# and drop the intermediates: 1.1 GB saved per SFT run.
trim_checkpoints () {
  local marker="$1"
  local run_dir
  run_dir=$(grep -o '"run_id": "[^"]*"' "${marker}" 2>/dev/null | head -1 | cut -d'"' -f4)
  [ -n "${run_dir}" ] || return 0
  [ -d "artifacts/runs/${run_dir}/checkpoints" ] || return 0
  find "artifacts/runs/${run_dir}/checkpoints" -name 'step-*.pt' \
    -not -name 'step-1875.pt' -delete 2>/dev/null
  say "trimmed intermediate optimiser checkpoints in ${run_dir}"
}

# ---------------------------------------------------------------- helpers

train_sft () {  # name apply_weight extra_file|"" seed
  local name="$1" aw="$2" extra="$3" seed="$4"
  if [ -d "artifacts/adapters/${name}" ]; then say "skip train ${name} (adapter exists)"; return 0; fi
  guard_disk
  say "train ${name} (apply-weight=${aw}, seed=${seed})"
  local args=(--config configs/training/sft_atomic.yaml --tier pilot --seed "${seed}"
              --max-steps 1875 --lora-config configs/training/lora_r32.yaml
              --apply-weight "${aw}" --micro-batch-size 2
              --base-adapter "${START}" --output-name "${name}")
  if [ -n "${extra}" ]; then
    args+=(--allow-composition --extra-train-file "${extra}")
  fi
  ${PY} scripts/train_sft.py "${args[@]}" \
    > "${NIGHT}/train_${name}.stdout.log" 2> "${NIGHT}/train_${name}.stderr.log"
  local status=$?
  if [ ${status} -ne 0 ]; then
    say "FAILED train ${name} exit=${status}"; tail -6 "${NIGHT}/train_${name}.stderr.log" | tee -a "${LOG}"
    return ${status}
  fi
  trim_checkpoints "${NIGHT}/train_${name}.stdout.log"
  say "done train ${name}"
}

rank () {  # adapter role split
  local adapter="$1" role="$2" split="$3"
  [ -d "${adapter}" ] || { say "MISSING adapter ${adapter} - cannot rank ${split}|${role}"; return 1; }
  local stem="artifacts/rank_${split}_${role}"
  [ -f "${stem}.done" ] && { say "skip rank ${split}|${role}"; return 0; }
  guard_disk
  say "rank ${split}|${role}"
  ${PY} scripts/run_exhaustive_ranking.py \
    --adapter "${adapter}" --input "artifacts/data/pilot/${split}.jsonl" \
    --task-tag "${split}" --run-method-tag "rank-${role}" \
    > "${stem}.stdout.log" 2> "${stem}.stderr.log"
  local status=$?
  if [ ${status} -eq 0 ]; then touch "${stem}.done"; say "done rank ${split}|${role}"
  else say "FAILED rank ${split}|${role} exit=${status}"; tail -4 "${stem}.stderr.log" | tee -a "${LOG}"; fi
  return ${status}
}

retention () {  # adapter role
  local adapter="$1" role="$2"
  [ -d "${adapter}" ] || { say "MISSING adapter ${adapter} - cannot check retention"; return 1; }
  for split in sft_validation_apply_non_sh1 sft_validation_plan; do
    local stem="artifacts/eval_night_retention_${role}_${split}"
    [ -f "${stem}.done" ] && { say "skip retention ${role}|${split}"; continue; }
    say "retention ${role}|${split}"
    ${PY} scripts/evaluate_model.py \
      --config configs/evaluation/baseline.yaml \
      --adapter "${adapter}" --input "artifacts/data/pilot/${split}.jsonl" \
      --k 1 --seed 0 --temperature 0.0 --max-new-tokens 24 \
      --label "night-retention-${role}-${split}" \
      > "${stem}.stdout.log" 2> "${stem}.stderr.log"
    if [ $? -eq 0 ]; then touch "${stem}.done"; else
      say "FAILED retention ${role}|${split}"; tail -4 "${stem}.stderr.log" | tee -a "${LOG}"
    fi
  done
}

reanalyse () {
  ${PY} scripts/analyze_night_series.py > "${NIGHT}/analysis.stdout.log" 2>&1 \
    && say "re-analysed: artifacts/reports/NIGHT_CONTROLS_REPORT.md" \
    || say "analysis step reported a problem, see ${NIGHT}/analysis.stdout.log"
}

stage_done () { [ -f "${NIGHT}/$1.done" ]; }
mark_stage  () { touch "${NIGHT}/$1.done"; say "STAGE $1 COMPLETE"; }

# ================================================================ S0 backup
if ! stage_done S0_backup; then
  say "STAGE S0: backing up artifacts/adapters to ${BACKUP_DIR}"
  mkdir -p "${BACKUP_DIR}"
  if tar -C artifacts -cf "${BACKUP_DIR}/adapters_$(date -u +%Y%m%dT%H%M%SZ).tar" adapters \
       2>> "${NIGHT}/backup.stderr.log"; then
    ( cd "${BACKUP_DIR}" && sha256sum adapters_*.tar > SHA256SUMS.txt )
    say "backup written, $(du -sh "${BACKUP_DIR}" | cut -f1) in ${BACKUP_DIR}"
    say "NOTE: same physical disk. Copy it off the machine for a real backup."
    mark_stage S0_backup
  else
    say "backup FAILED - continuing anyway, but adapters remain unprotected"
  fi
fi

# ================================================== S1 re-grade APPLY (no GPU)
if ! stage_done S1_regrade; then
  say "STAGE S1: mode-aware re-grade of stored APPLY retention generations"
  if ${PY} scripts/regrade_apply_retention.py > "${NIGHT}/regrade.stdout.log" 2>&1; then
    tail -12 "${NIGHT}/regrade.stdout.log" | tee -a "${LOG}"
    mark_stage S1_regrade
  else
    say "S1 FAILED"; tail -8 "${NIGHT}/regrade.stdout.log" | tee -a "${LOG}"
  fi
fi

# ============================================ S2 continuous endpoints (no GPU)
if ! stage_done S2_continuous; then
  say "STAGE S2: continuous ranking endpoints from data already on disk"
  reanalyse
  mark_stage S2_continuous
fi

# ==================================== S3 D-022 positive control (~80 min GPU)
if ! stage_done S3_motif_positive_control; then
  say "STAGE S3: oracle adapters on the held-motif families"
  for split in motif_b_d3 motif_d_d3; do
    for seed in 0 1 2; do
      rank "artifacts/adapters/sft_composition_oracle_seed${seed}" "oracle${seed}" "${split}"
    done
  done
  reanalyse
  mark_stage S3_motif_positive_control
fi

# ================================= S4 equal-budget atomic control (~2.5 h GPU)
if ! stage_done S4_equal_budget_control; then
  say "STAGE S4: equal-budget atomic control"
  say "  5000 atomic examples (4x1000 APPLY + 1000 PLAN) = the composition run's"
  say "  example count, same 1875 steps, same lr, same rank, same start adapter."
  for seed in 0 1 2; do
    train_sft "sft_atomic_control_eqbudget_seed${seed}" 4 "" "${seed}"
  done
  for split in capacity_d3_train capacity_d3_heldout; do
    for seed in 0 1 2; do
      rank "artifacts/adapters/sft_atomic_control_eqbudget_seed${seed}" "atomctl${seed}" "${split}"
    done
  done
  reanalyse
  mark_stage S4_equal_budget_control
fi

# ================================ S5 composition SFT keeping APPLY (~2.5 h GPU)
if ! stage_done S5_applykeep; then
  say "STAGE S5: composition SFT with APPLY retained in the mixture"
  say "  apply-weight 1 => 1000 APPLY + 1000 PLAN + 4000 composition = 6000 examples."
  say "  Optimiser steps stay at 1875, which is the quantity being controlled;"
  say "  the example count differs from the 5000 of the original run by design."
  for seed in 0 1 2; do
    train_sft "sft_composition_applykeep_seed${seed}" 1 \
      artifacts/data/pilot/sft_train_composition.jsonl "${seed}"
  done
  for seed in 0 1 2; do
    retention "artifacts/adapters/sft_composition_applykeep_seed${seed}" "applykeep${seed}"
  done
  if ${PY} scripts/regrade_apply_retention.py > "${NIGHT}/regrade_after_applykeep.stdout.log" 2>&1; then
    say "re-graded the original APPLY runs again for a like-for-like table"
  fi
  for split in capacity_d3_train capacity_d3_heldout; do
    for seed in 0 1 2; do
      rank "artifacts/adapters/sft_composition_applykeep_seed${seed}" "applykeep${seed}" "${split}"
    done
  done
  reanalyse
  mark_stage S5_applykeep
fi

# ================================================ S6 depth-4 ranking (~2.5 h GPU)
if ! stage_done S6_depth4; then
  say "STAGE S6: depth-4 exhaustive ranking, 625 candidates per task"
  if [ ! -f artifacts/data/pilot/capacity_d4_train.jsonl ]; then
    ${PY} scripts/prepare_depth4_data.py --count 125 \
      > "${NIGHT}/depth4_data.stdout.log" 2>&1 \
      && say "generated capacity_d4_train" \
      || say "depth-4 data generation FAILED"
  fi
  rank "${START}" atomic capacity_d4_train
  for seed in 0 1 2; do
    rank "artifacts/adapters/sft_composition_oracle_seed${seed}" "oracle${seed}" capacity_d4_train
  done
  reanalyse
  mark_stage S6_depth4
fi

# ============================================= S7 degraded verifier (~1.7 h GPU)
if ! stage_done S7_noisy_verifier; then
  say "STAGE S7: degraded verifier on the distillation labels"
  say "  alpha = 1.0 throughout; beta in {0.10, 0.25, 0.50} false-accepts."
  say "  Identical recipe to sft_composition_oracle_seed0 - only the labels differ."
  if [ ! -f artifacts/data/pilot/sft_train_composition_noisy_b10.jsonl ]; then
    ${PY} scripts/prepare_noisy_verifier_data.py > "${NIGHT}/noisy_data.stdout.log" 2>&1 \
      && say "generated the three noisy distillation sets" \
      || say "noisy data generation FAILED"
  fi
  for tag in 10 25 50; do
    train_sft "sft_composition_noisy_b${tag}_seed0" 0 \
      "artifacts/data/pilot/sft_train_composition_noisy_b${tag}.jsonl" 0
  done
  for tag in 10 25 50; do
    rank "artifacts/adapters/sft_composition_noisy_b${tag}_seed0" "noisy${tag}" capacity_d3_heldout
  done
  reanalyse
  mark_stage S7_noisy_verifier
fi

say "NIGHT SERIES FINISHED. free disk: $(free_gb) GB"
say "read: artifacts/reports/NIGHT_CONTROLS_REPORT.md"
say "read: artifacts/reports/APPLY_RETENTION_REGRADE.md"
