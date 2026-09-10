# D-021/D-022 exhaustive-ranking final report

## Completion and metric

All 32 registered ranking cells are complete: four D-021 splits and four D-022 families,
each evaluated with the atomic baseline and three trained adapters. The 12/300 smoke run
is excluded by exact task-count matching. Depth-2 Hit@32 is not interpreted because all 25
programs fit inside K=32.

Uncertainty uses a 10,000-fold crossed training-seed-by-task bootstrap. Reported bootstrap
p-values are two-sided tests from the centered bootstrap null; Holm correction follows the
registered families. Per-seed exact McNemar tests remain in the JSON as diagnostics only.

## D-021: capacity and field transfer

| Contrast | Delta (pp) | 95% CI (pp) | p / Holm p |
|---|---:|---:|---:|
| C1: supervision transfer to held-out fields, Hit@32 | +59.11 | [+53.00, +65.33] | 9.999e-05 |
| C2: compositions learnable in-domain, Hit@32 | +48.22 | [+41.33, +55.00] | 0.0003 |
| C3: depth-2 minus depth-3, Hit@8 | +42.89 | [+38.00, +47.67] | 0.0003 |
| C4: train minus held-out fields, Hit@32 | -2.22 | [-6.00, +1.56] | 0.2464 |

C1 and C2 are positive with confidence intervals above zero, selecting the preregistered
method-ceiling branch for PLAN ranking. C4 is near zero: field novelty does not measurably
reduce exhaustive Hit@32 after direct composition supervision. The atomic-retention result
below prevents treating this as an unqualified capacity-control success.

## D-022: held-motif 2x2

| Family | Delta Hit@32 (pp) | 95% CI (pp) | p / Holm p |
|---|---:|---:|---:|
| A: known fields, seen motifs | +55.20 | [+48.13, +62.00] | 0.0003 |
| B: known fields, held motifs | -13.47 | [-24.00, -2.67] | 0.0133 |
| C: new fields, seen motifs | +58.80 | [+51.47, +65.73] | 0.0003 |
| D: new fields, held motifs | -15.07 | [-25.60, -4.27] | 0.005799 |

- Motif main effect: **+71.27 pp**, 95% CI **[+58.53, +83.60]**.
- Field main effect: **-1.00 pp**, 95% CI **[-8.27, +6.33]**.

The sign flip reproduces the external qualitative result on independently generated data:
composition supervision helps on seen motifs and harms ranking on held motifs, while field
novelty contributes little.

## Atomic forgetting

| Adapter | APPLY pass@1 | PLAN pass@1 |
|---|---:|---:|
| atomic | 1.000 | 0.980 |
| oracle0 | 0.000 | 0.970 |
| oracle1 | 0.000 | 0.970 |
| oracle2 | 0.000 | 0.910 |
| nomotif0 | 0.000 | 0.970 |
| nomotif1 | 0.000 | 0.980 |
| nomotif2 | 0.000 | 0.960 |

**Retention warning: CATASTROPHIC_APPLY_COLLAPSE.** All six trained adapters drop
from APPLY pass@1 1.00 to 0.00 with parse rate 0.00.
Mean PLAN pass@1 changes by -2.00 pp.
Under section 6 of the frozen D-021 preregistration, the capacity gain is therefore
not counted as a clean success. It remains evidence of PLAN ranking capacity, not of
retained multi-mode competence.

2 complete CPU fallback baseline runs are retained
in the run index but excluded here by the environment-matching rule `cuda_available == true`;
all reported retention cells use the same GPU environment.

## Claim boundary

Direct supervision rules out a hard representational-capacity ceiling for the tested model,
LoRA rank and task family. It does not isolate whether the earlier nulls were caused by search,
policy-gradient credit assignment, supervision density or optimisation budget.

## Measured evaluation cost

The 32 ranking cells used 212,800 model passes
and 6.29 summed GPU-hours.
The 14 retained atomic checks generated 7,452
completion tokens in 5.2 summed GPU-minutes.
Peak allocated CUDA memory was
1.26 GiB.

## Reproduction

```bash
.venv/bin/python scripts/analyze_ranking.py
.venv/bin/python scripts/plot_ranking_results.py
```
