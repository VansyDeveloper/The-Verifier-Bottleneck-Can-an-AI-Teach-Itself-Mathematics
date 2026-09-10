# Experiments used in paper10

This publication snapshot contains the existing experiments behind `paper10/main10.tex`.
The two series have different splits, reference checkpoints and replication counts;
their effect sizes must not be pooled. Negative controls used in the paper are retained.

## Primary composition confirmation (six replicates)

`new_stuff/stage4_verifier_bottleneck_audit_v22/` preserves the compact audit release,
including its source-code SHA-256 manifest, protocols, recovery amendment, tests,
statistics, tables and operational provenance. Earlier implementations inside `source/`
are dependencies of the final training and evaluation path.

- Final analysis: `scientific/statistics/FINAL_ANALYSIS.json`.
- Per-replicate results: `post_analysis_release_v1/tables/primary_per_replicate.csv`.
- Entry point: `source/artifacts/stage4_composition_confirm_v1_0p6b/code/confirm.py`.
- Primary result: exact Hit@32 gain of 27.40 percentage points against equal-budget
  atomic control; six positive replicates, 95% t interval [25.403, 29.397].
- Held-motif transfer failures and atomic forgetting remain part of the evidence.

The compact release does not contain frozen model weights, adapters, generated
Stage 4 datasets or full raw ranking shards. Its audit README identifies the original
full archive needed for a byte-identical reconstruction. Source README files and
machine-specific paths inside this historical snapshot are preserved verbatim.

## Search, GRPO and composition controls (three seeds)

`verifier_bottleneck_codex_pack/` contains the polynomial environment, verifier,
action-space search, SFT/GRPO, exhaustive ranking, analysis scripts, tests,
configurations, data and registered protocols used in the second series.

- `artifacts/reports/FINAL_REPORT.md`: consolidated interpretation.
- `artifacts/reports/confirmatory_statistics.json`: diversity-matched search controls.
- `artifacts/reports/selftraining_statistics.json`: 400-step GRPO comparison.
- `artifacts/reports/ranking_statistics.json`: composition capacity and motif transfer.
- `artifacts/reports/apply_retention_regrade.json`: atomic APPLY regrading.
- `artifacts/reports/run_index.csv`: historical run inventory, including pilot runs.

The saved data and aggregate reports are included; model weights, adapters and raw
run directories are not. Analyses that read `artifacts/runs/` require those original
run files. Pilot helpers and datasets are retained where they support the final pipeline.

## Local checks

The packages declare their dependencies in their respective `pyproject.toml` files;
the second series also includes `requirements/lock_linux_cu124.txt`.
Tests for the second series run from `verifier_bottleneck_codex_pack/` with
`python -m pytest -q`. Stage 4 tests reside in the three `source/artifacts/*/tests/`
directories. These checks validate code, not a new GPU reproduction of the paper.

`paper10/` contains the manuscript source, its table fragments, figure PDFs and class
file as supplied. Author placeholders and manuscript wording are unchanged.
