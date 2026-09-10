# Stage 4 composition-only confirmation

**Decision: POSITIVE_COMPOSITION_CONFIRMATION.**

The model is Qwen3-0.6B throughout. The historical pilot remains unchanged and formally negative under its old gates.

## Pre-freeze failure and recovery amendment

Freeze attempt 0 failed before any final file or evaluator access because the legacy non-gating confirm-atomic schedule required field/degree cells with zero remaining disjoint states.
The byte-identical FAILED receipt and logs are preserved. Recovery Amendment 001 was committed while final data, rankings and metrics were absent; it changed only confirm-atomic feasibility handling and retried freeze once.
Final A/B/C/D generation, all seeds and sizes, the primary endpoint and gates, all statistics, and all 12 trained adapters remained byte-identical.

## Primary final-A result

- Mean `composition_distill - atomic_control` Hit@32: **27.400 pp**.
- 95% seed t-CI: **[25.403, 29.397] pp**.
- Two-sided seed t-test: **p=3.4451821e-07**.
- Positive replicates: **6/6**.
- Exact sign-flip sensitivity: **p=0.03125**.
- Crossed seed-by-task bootstrap 95% CI: **[23.117, 31.667] pp**.

## Interpretation

A positive primary decision supports improved ranking of unseen state/task-disjoint family-A depth-3 compositions in known fields. It does not by itself prove held-motif or new-field transfer.
The exploratory dev-B transfer delta observed before this preregistration was -17.6 pp and is disclosed as negative.

SH1, PLAN, APPLY and atomic forgetting are reported descriptively and never veto the composition decision.

## Per-replicate primary deltas

| Replicate | Control hits | Distill hits | Delta (pp) |
|---:|---:|---:|---:|
| 0 | 435 | 737 | 30.200 |
| 1 | 457 | 724 | 26.700 |
| 2 | 454 | 714 | 26.000 |
| 3 | 460 | 752 | 29.200 |
| 4 | 459 | 729 | 27.000 |
| 5 | 454 | 707 | 25.300 |

## Integrity

- Raw ranking shards independently rederived: 2080.
- All programs were exactly enumerated, verifier labels recomputed, and score/tie order checked.
- Within each replicate, control and distill used equal optimizer steps, epochs, effective batch and loss-bearing target tokens.
- Final splits passed state/task disjointness and held-motif audits.
- Recovery amendment receipt SHA-256: `0D7812513251B4B69B90E4B590CDADE768ACC1D94292106F9A165C5B806F53E4`.
- Freeze-attempt-0 FAILED SHA-256: `187C95F2E6E0FE4FAF0B7F176C0B1D166E7D9F44394F78F858BFBEC0A3BFFE3E`.
