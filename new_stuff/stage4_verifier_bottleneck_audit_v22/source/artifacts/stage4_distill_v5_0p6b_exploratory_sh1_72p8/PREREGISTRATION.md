# Stage 4E — Exploratory composition from frozen SH1 72.778%

## Scientific status

This is a new exploratory follow-up on Qwen3-0.6B. It does not amend the sealed
registered Stage 4 v5 endpoint, which remains `FAILED_CALIBRATION_GATE`.
The source checkpoint failed the registered atomic gate and therefore any later
composition effect is an exploratory signal, never a confirmatory Stage 4 PASS.

No pre-pilot atomic rescue and no mutation of the source checkpoint are
permitted in this study. The frozen source is
`artifacts/stage4_sh1_v5_0p6b/adapters/corrective`, with calibration metrics:
PLAN 95.944%, SH1 APPLY 72.778%, and SC2/REV/AC1/AX1 APPLY 100%. The source
adapter, training receipt, calibration gate, raw calibration generations, and
terminal atomic decision are hash-bound before Discover. During the pilot, the
composition branch receives the predeclared 0.2 atomic replay, including SH1,
while the control branch is packed from fresh atomic targets. The two branches
have the same total loss-bearing token budget, but not the same SH1/replay
share; this is branch training, not another atomic-v5 corrective cycle.
The old atomic-final file was generated during v5 preparation, but its official
evaluator was never opened: there is no final gate, generation, or access
receipt. This follow-up does not read that file and does not use it.

## Exploratory source eligibility

The source may be exported only when all checks hold: model is exactly
Qwen/Qwen3-0.6B; the original decision is `FAILED_CALIBRATION_GATE` with
`composition_unlocked=false` and no final gate; PLAN is at least 95%; SH1 APPLY
is at least 70%; every control APPLY is at least 98%; adapter, calibration data,
raw generations, gate, protocol, and receipt hashes agree. This eligibility is
not an atomic PASS.

## Discover and pilot

Use the frozen Stage 4 data construction: known fields 5/7/13/19/23/29,
transfer fields 11/17, and held motifs AX1->SH1, AC1->REV, SC2->AX1. Generate
4,000 train tasks, 500 dev-A tasks, and 500 descriptive dev-B tasks. Exact
Discover must provide at least 2,000 unique trajectories, 1,000 solvable tasks,
20 full signatures, 25 occurrences of every operation at every position, and
balanced first operations. All new states and task IDs must be disjoint from
the atomic training/calibration material and across composition splits.

Seed-0 compares one frozen `atomic_base` with equal-budget `atomic_control` and
`composition_distill`. Initial settings are LR 1e-4, two epochs, atomic replay
0.2, effective batch 64, microbatch 4, warmup 3%, clipping 1.0, bf16, LoRA r32.
Candidate programs are exhaustively ranked: exactly 25/125/625 programs at
depth 2/3/4, using full-vocabulary action-token log probabilities and the exact
verifier. Random generation and analytic pass@K approximations are forbidden.

The frozen corrective function is evaluated and exported in FP32 because the
registered atomic gate was measured after FP32 promotion. Branch master weights
remain FP32; forward/loss compute uses preregistered BF16 autocast. This exact
representation is frozen by pre-training Amendment 001 after the first import
failed before Discover or branch training due to a BF16 merge discrepancy.

Pre-training Amendment 002 preserves the original first-operation spread gate
and changes only deterministic selection: the requested trajectory count is
allocated into explicit per-operation quotas before candidate choice. It was
recorded after the first prepare stopped on a 448/450/450/451/451 split, before
any branch training or model-based composition metric was computed.

An exploratory pilot signal requires all of: base dev-A depth-3 Hit@32 in
5–50%; distill minus control Hit@32 at least 8 percentage points; atomic
forgetting at most 2 percentage points; correct-program mass increases; zero
leakage; and exact equality of optimizer steps, epochs, effective batch, and
loss-bearing target tokens. If only the effect-size check fails while all other
checks pass, the already specified single sequential LR/epochs/replay dev cycle
is allowed. No other tuning is allowed.

## Future multi-seed study — not executable in this protocol

This protocol and its CLI stop after the seed-0 pilot and cannot create final
data or run confirmation. Only after an exploratory pilot signal may a new,
separately committed preregistration create and hash-freeze composition final
A/B/C/D. That future study may run seeds 0–5 with the same three branches. The
registered numerical criteria remain mean delta at least 5 points, 95% seed CI
above zero, seed t-test p<0.05, positive effect in at least 5/6 seeds, forgetting
at most 2 points, correct-mass growth, 20,000 seed-by-task bootstrap repetitions,
exact sign-flip, and Holm correction. Even if a future study satisfies all of
them, the result is labelled
`EXPLORATORY_MULTI_SEED_SIGNAL`; it cannot be reported as registered Stage 4
success because the source atomic prerequisite failed.

This protocol creates no composition final tests and has no final freeze. The
sealed Stage 4 v5 root, its reports, archives, hashes, and Git commit remain
immutable throughout this follow-up.
