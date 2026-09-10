# Stage 4 SH1 v5 — continued Qwen3-0.6B curriculum

Frozen before v5 data generation, training, calibration, or final evaluation.

- Model size is fixed to Qwen3-0.6B. Changing model size is forbidden without new direct user approval.
- Parent checkpoint is `artifacts/stage4_sh1_v4/adapters/atomic_coordinate`; no previous adapter is discarded or reinitialized.
- Baseline: PLAN 99.4667%; SH1 APPLY 45.6667%; SC2 99.6667%; REV/AC1/AX1 100%. SH1 by degree is 93% / 27% / 17% for degrees 2/3/4.
- Fields are 5/7/13/19/23/29 and degrees are 2/3/4. State modes are dense, boundary, sparse, monomial, progression, alternating.
- Calibration and final each contain 20 state-disjoint starts per field x degree cell. Every start is evaluated under all five operations, giving 1,800 tasks per split. Splits are disjoint by `(field,start)` and exclude prior v2-v4 evaluation states.
- Train never uses calibration/final states. Where a finite field-degree cell has fewer available states than its registered presentation budget (notably FIELD=5, degree=2), available train states may repeat with unique task IDs; the manifest records unique-state and repeated-presentation counts. No evaluation state may repeat in train.
- Coordinate refresher uses 24,000 SH1 states (4k/8k/12k by degree), expanded into all output coordinates, one epoch at LR 1e-5.
- Prefix assembly uses every prefix length 2..degree+1 for those states (80,000 targets), one epoch at LR 1e-5.
- Full consolidation uses 64,000 SH1 full vectors (8k/24k/32k), 4,000 APPLY tasks for each control operation, and 10,000 balanced PLAN replay targets, one epoch at LR 1e-5.
- All phases use bf16, effective batch 64, microbatch 8, accumulation 8, warmup 3%, clipping 1.0, seed 0. Each phase saves a separate resumable adapter.
- Calibration PASS requires PLAN >=95%, every APPLY >=90%, spread <=10 percentage points, and each control operation no more than 2 percentage points below its v4 baseline.
- If PLAN and controls pass but SH1 fails, one corrective cycle is allowed: 40,000 SH1 full-vector targets allocated by clipped calibration cell error, 10,000 balanced control APPLY, and 5,000 balanced PLAN, one epoch at LR 5e-6. No other v5 cycle is allowed.
- Final is evaluated exactly once and only after a checkpoint passes calibration. Composition remains locked until final atomic PASS.
