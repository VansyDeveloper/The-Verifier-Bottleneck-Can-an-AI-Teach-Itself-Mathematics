# D-021 sampled-grid report

These preregistered sampled `pass@32` endpoints are retained for continuity. Under
Amendment 001 they are secondary to exhaustive ranking because duplicate samples make
`pass@32` depend on sampler diversity as well as policy preference.

| Contrast | Delta (pp) | 95% CI (pp) | p / Holm p |
|---|---:|---:|---:|
| C1: transfer to held-out fields | +63.33 | [+56.67, +70.33] | 9.999e-05 |
| C2: learnability in-domain | +62.56 | [+56.44, +68.44] | 0.0003 |
| C3: depth-2 minus depth-3 | +26.44 | [+19.67, +33.33] | 0.0003 |
| C4: train minus held-out fields | +1.11 | [-5.33, +7.67] | 0.7246 |
| C1 secondary at iid@8.0 | +12.67 | [+7.00, +18.44] | 0.0003 |

**Registered decision:** METHOD CEILING - capacity and transfer are both fine, so the earlier nulls belong to search and policy-gradient training; first-step credit assignment becomes worth testing.

This decision concerns the sampled endpoint only. The retention rule and final claim
boundary are reported in `RANKING_FINAL_REPORT.md`.

## Measured evaluation cost

The 24 cells used 235,197 model passes, scored
2,116,773 operation tokens and took
6.62 summed GPU-hours. Peak allocated CUDA
memory was 1.26 GiB.

## Reproduction

```bash
.venv/bin/python scripts/analyze_capacity.py
```
