# Report package contents

Updated 2026-08-02. Superseded July version preserved in git history only if the
directory is tracked; this file is the current index for `artifacts/reports/`.

## Current

| file | role |
|---|---|
| `FINAL_REPORT.md` | **master report** — both preregistered series, all contrasts, limitations, reproduction |
| `confirmatory_statistics.json` | D-018: Q1–Q4, arm table, per-seed detail |
| `selftraining_statistics.json` | D-020: Q5–Q8, arm table, per-seed detail |
| `july_reanalysis_statistics.json` | continuity on the already-opened July set |
| `data_integrity.json` | all ten splits, exact-key leaks quantified |
| `run_index.csv` | 120 runs, 116 DONE and 4 FAILED |
| `confirmatory_diversity_curve.png` | structure on the diversity curve; below it on cost |
| `dev_temperature_sweep.png` | interior iid optimum at T≈8 |
| `selftraining_mechanism.png` | per-candidate accuracy up 2.4x, coverage down 2.1x, pass@32 flat |
| `selftraining_arms.png` | one shared diversity curve across all three adapters |
| `confirmatory_arm_by_role.png` | search arm dominates; training does not |
| `confirmatory_contrasts.png` | D-018 preregistered CIs |
| `selftraining_contrasts.png` | D-020 preregistered CIs |
| `confirmatory_grpo_training.png` | training reward vs group coverage |

## Superseded, preserved

| file | status |
|---|---|
| `final_report.md` | July pilot. Data valid, interpretation replaced by `FINAL_REPORT.md`. |
| `final_report_2026-08-01.md` | D-018 only; folded into the master report. |
| `three_seed_heldout_statistics.json` | July three-seed statistics. |
| `pilot_pre_grpo_paired_statistics.json`, `pilot_post_grpo_paired_statistics.json` | July paired tests. |
| `summary.json` | July machine-readable summary. |
| `heldout_pass_at_32.png` | July figure. |
