# Stage 4 Verifier Bottleneck — final Discover-and-Distill report

**Scientific decision: POSITIVE_COMPOSITION_CONFIRMATION.**

The model is Qwen3-0.6B throughout. The starting atomic checkpoint is the frozen corrective checkpoint with SH1 APPLY 72.778%. No 1.7B model was used.

## Primary preregistered confirmation

- Final-A depth-3 Hit@32 mean delta: **27.400 pp**.
- 95% seed t-CI: **[25.403, 29.397] pp**.
- Two-sided seed t-test: **p=3.4451821e-07**.
- Positive replicates: **6/6**.
- Exact sign-flip sensitivity: **p=0.03125**.
- Crossed seed-by-task bootstrap (20,000): **[23.117, 31.667] pp**.

All four registered primary gates passed. This is evidence that composition_distill improved exact ranking of unseen family-A depth-3 compositions relative to the equal-budget atomic_control.

## Secondary endpoints

| Endpoint | Mean delta (pp) | 95% CI (pp) | raw p | Holm p |
|---|---:|---:|---:|---:|
| final_b_depth3_hit32 | -16.017 | [-19.105, -12.929] | 4.24369e-05 | 8.48739e-05 |
| final_c_depth3_hit32 | 26.133 | [24.912, 27.355] | 3.75834e-08 | 1.87917e-07 |
| final_d_depth3_hit32 | -11.800 | [-13.087, -10.513] | 2.56032e-06 | 1.02413e-05 |
| transfer_bcd_depth2_hit32 | 0.000 | [0.000, 0.000] | 1 | 1 |
| transfer_bcd_depth4_hit32 | 5.200 | [4.454, 5.946] | 9.92033e-06 | 2.9761e-05 |

## Atomic skills and forgetting

All 13 atomic raw files (13,000 task rows) were independently rederived. The descriptive ≤2 pp forgetting check passed for 2/12 trained branch/replicate pairs.
Per the registered composition-only confirmation, atomic forgetting is reported descriptively and does not override the primary composition decision.

## Integrity

- Full exact-ranking validation: **PASS**, 2080 shards.
- Every depth-2/3/4 task ranks the complete 25/125/625 unique program set; verifier labels and score order were recomputed.
- Leakage audit count: **0**.
- Training budgets are equal within every replicate.
- All primary evidence was produced locally under the frozen environment.

## Operational acceleration addendum (post hoc, non-gating)

Secondary rankings were accelerated on ai01 with 3×RTX 5080. A depth-4 equivalence smoke retained all correct-program ranks and every target rank metric; two near-tied incorrect positions out of 625 swapped under torch 2.11, with the difference disclosed in ACCELERATION_PROVENANCE.json.
All imported remote raw rankings were rederived into Windows metrics/receipts under the frozen evaluator before final analysis. Failed transfer/preflight attempts produced no accepted scientific shards and are preserved as operational provenance.

## Scope

This positive result supports exact compositional ranking transfer for the registered final-A endpoint. Secondary B/C/D and depth-4 results are reported separately and must not be conflated with the primary claim.
The earlier exploratory pilot remains historically unchanged; the six-replicate follow-up is the confirmatory evidence reported here.

Archive hashes and member counts are recorded in manifests/ARCHIVE_MANIFEST.json.
