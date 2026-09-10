# Post-hoc control series - results

These are **disclosed post-hoc controls**, not confirmatory tests. Every
evaluation set they use was already opened by D-018 / D-020 / D-021 / D-022,
with the single exception of `capacity_d4_train`, which is newly generated here.
Their purpose is to rule out alternative explanations of results that are already
reported, so they carry no new preregistered decision rule and no Holm family.

The registered `ranking_statistics.json` and `RANKING_FINAL_REPORT.md` are
untouched; statistics come from the same crossed seed-by-task bootstrap.

| contrast | question | delta (pp) | 95% CI | seeds +/- | p |
|---|---|---:|---:|---:|---:|
| `N1a_mass_oracle_vs_atomic_d3_heldout` | probability mass on correct programs, held-out fields | +7.80 | [+6.19, +9.43] | 3/0 | 9.999e-05 |
| `N1b_mass_oracle_vs_atomic_d3_train` | probability mass on correct programs, known fields | +6.22 | [+4.35, +8.07] | 3/0 | 9.999e-05 |
| `N1c_mrr_oracle_vs_atomic_d3_heldout` | reciprocal rank of the best correct program, held-out fields | +23.73 | [+19.38, +28.52] | 3/0 | 9.999e-05 |
| `N1d_mass_nomotif_vs_atomic_motif_a` | probability mass, seen motifs / known fields | +9.87 | [+7.49, +12.54] | 3/0 | 9.999e-05 |
| `N1e_mass_nomotif_vs_atomic_motif_b` | probability mass, held motifs / known fields | +0.24 | [-0.13, +0.72] | 2/1 | 0.2567 |
| `N1f_mass_nomotif_vs_atomic_motif_c` | probability mass, seen motifs / new fields | +11.34 | [+9.15, +13.69] | 3/0 | 9.999e-05 |
| `N1g_mass_nomotif_vs_atomic_motif_d` | probability mass, held motifs / new fields | +0.40 | [+0.12, +0.85] | 3/0 | 0.0378 |
| `N2a_oracle_vs_atomic_motif_b` | POSITIVE CONTROL: are held-motif tasks rankable when the motif IS in training? | +54.67 | [+48.13, +61.20] | 3/0 | 9.999e-05 |
| `N2b_oracle_vs_atomic_motif_d` | POSITIVE CONTROL: same, new fields | +51.73 | [+45.07, +58.40] | 3/0 | 9.999e-05 |
| `N2c_oracle_vs_nomotif_motif_b` | effect of withholding the motif, seed-paired, known fields | +68.13 | [+59.20, +76.80] | 3/0 | 9.999e-05 |
| `N2d_oracle_vs_nomotif_motif_d` | effect of withholding the motif, seed-paired, new fields | +66.80 | [+56.40, +77.60] | 3/0 | 9.999e-05 |
| `N2e_mass_oracle_vs_nomotif_motif_b` | same on probability mass | +10.54 | [+7.99, +13.33] | 3/0 | 9.999e-05 |
| `N3a_atomctl_vs_atomic_d3_train` | does 1875 more steps of atomic training alone move depth-3 ranking? | +0.11 | [-6.44, +6.44] | 2/1 | 0.9886 |
| `N3b_atomctl_vs_atomic_d3_heldout` | same, held-out fields | +1.67 | [-4.67, +7.67] | 2/1 | 0.6126 |
| `N3c_oracle_vs_atomctl_d3_train` | C2 de-confounded: composition data vs equal-budget atomic data | +48.11 | [+38.67, +57.11] | 3/0 | 9.999e-05 |
| `N3d_oracle_vs_atomctl_d3_heldout` | C1 de-confounded: composition data vs equal-budget atomic data | +57.44 | [+49.67, +65.33] | 3/0 | 9.999e-05 |
| `N4a_applykeep_vs_atomic_d3_train` | composition gain when APPLY stays in the mixture, known fields | +46.11 | [+39.56, +52.22] | 3/0 | 9.999e-05 |
| `N4b_applykeep_vs_atomic_d3_heldout` | composition gain when APPLY stays in the mixture, held-out fields | +54.00 | [+47.33, +60.67] | 3/0 | 9.999e-05 |
| `N4c_applykeep_vs_oracle_d3_heldout` | ranking cost of keeping APPLY, seed-paired | -5.11 | [-11.22, +0.33] | 0/3 | 0.08969 |
| `N5a_oracle_vs_atomic_d4_hit32` | does the composition gain survive at depth 4 (625 candidates)? | +23.47 | [+13.07, +34.13] | 3/0 | 9.999e-05 |
| `N5b_oracle_vs_atomic_d4_hit8` | same at Hit@8 | +8.53 | [+3.20, +14.40] | 3/0 | 0.0041 |
| `N5c_mass_oracle_vs_atomic_d4` | same on probability mass | +1.33 | [+0.69, +2.27] | 3/0 | 0.006699 |
| `N6a_noisy10_vs_atomic_d3_heldout` | distillation gain when the verifier false-accepts 10% of trajectories | +49.00 | [+42.00, +55.67] | 1/0 | 9.999e-05 |
| `N6b_noisy25_vs_atomic_d3_heldout` | same at beta = 0.25 | +48.00 | [+41.00, +54.67] | 1/0 | 9.999e-05 |
| `N6c_noisy50_vs_atomic_d3_heldout` | same at beta = 0.50 | +41.00 | [+34.33, +48.00] | 1/0 | 9.999e-05 |
| `N6d_noisy10_vs_exact_labels` | gain lost to a 10% false-accept rate, against the exact-verifier adapter | -8.67 | [-12.67, -5.00] | 0/1 | 0.0002 |
| `N6e_noisy25_vs_exact_labels` | same at beta = 0.25 | -9.67 | [-14.00, -5.33] | 0/1 | 9.999e-05 |
| `N6f_noisy50_vs_exact_labels` | same at beta = 0.50 | -16.67 | [-21.67, -11.67] | 0/1 | 9.999e-05 |
| `N6g_mass_noisy50_vs_exact_labels` | same on probability mass at beta = 0.50 | -5.41 | [-6.65, -4.31] | 0/1 | 9.999e-05 |

## Motif main effect on probability mass

- motif effect **+10.28 pp**, 95% CI [+8.05, +12.54]
- field effect **-0.82 pp**, 95% CI [-2.20, +0.64]
- per-family deltas (pp): A +9.87, B +0.24, C +11.34, D +0.40

## Cells discovered

```
capacity_d2_heldout|atomic
capacity_d2_heldout|oracle0
capacity_d2_heldout|oracle1
capacity_d2_heldout|oracle2
capacity_d2_train|atomic
capacity_d2_train|oracle0
capacity_d2_train|oracle1
capacity_d2_train|oracle2
capacity_d3_heldout|applykeep0
capacity_d3_heldout|applykeep1
capacity_d3_heldout|applykeep2
capacity_d3_heldout|atomctl0
capacity_d3_heldout|atomctl1
capacity_d3_heldout|atomctl2
capacity_d3_heldout|atomic
capacity_d3_heldout|noisy10
capacity_d3_heldout|noisy25
capacity_d3_heldout|noisy50
capacity_d3_heldout|oracle0
capacity_d3_heldout|oracle1
capacity_d3_heldout|oracle2
capacity_d3_train|applykeep0
capacity_d3_train|applykeep1
capacity_d3_train|applykeep2
capacity_d3_train|atomctl0
capacity_d3_train|atomctl1
capacity_d3_train|atomctl2
capacity_d3_train|atomic
capacity_d3_train|oracle0
capacity_d3_train|oracle1
capacity_d3_train|oracle2
capacity_d4_train|atomic
capacity_d4_train|oracle0
capacity_d4_train|oracle1
capacity_d4_train|oracle2
motif_a_d3|atomic
motif_a_d3|nomotif0
motif_a_d3|nomotif1
motif_a_d3|nomotif2
motif_b_d3|atomic
motif_b_d3|nomotif0
motif_b_d3|nomotif1
motif_b_d3|nomotif2
motif_b_d3|oracle0
motif_b_d3|oracle1
motif_b_d3|oracle2
motif_c_d3|atomic
motif_c_d3|nomotif0
motif_c_d3|nomotif1
motif_c_d3|nomotif2
motif_d_d3|atomic
motif_d_d3|nomotif0
motif_d_d3|nomotif1
motif_d_d3|nomotif2
motif_d_d3|oracle0
motif_d_d3|oracle1
motif_d_d3|oracle2
```
