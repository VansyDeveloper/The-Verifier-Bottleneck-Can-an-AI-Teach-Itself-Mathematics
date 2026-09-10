# Preregistration — structure versus diversity, confirmatory series

Frozen 2026-08-01, before any evaluation on `confirmatory_heldout.jsonl`.
This file is not edited after freezing. Corrections go into a new dated file.

## 1. Why this series exists

The July three-seed pilot reported +16.0 pp `delta pass@32` for prefix-balanced
action search over iid action search on held-out PLAN depth-3 tasks. Re-analysis
of that pilot's raw generations (2026-08-01) found two problems:

1. The effect is present **before any RL**. On dev at K=16 the same +12 pp gap
   appears on the pre-GRPO SFT adapter, and the training-branch main effect is
   within one standard error. The frozen set was evaluated only on the diagonal
   (iid adapter with iid search, prefix adapter with prefix search), so the
   published number cannot be attributed to GRPO.
2. iid at T=0.7 produces only 6.2–7.0 distinct programs out of 32 candidates
   while prefix-balanced produces 12.3–13.3. The pre-registered controls that
   separate *structure* from *diversity* — E2 (high temperature) and E3
   (temperature mix) from `docs/05` — were never run in the action space.

This series tests structure against diversity directly.

## 2. Selection stage (already complete, dev only)

`artifacts/runs/20260801T080046Z_qwen3-0.6b_plan-dev_exploration_diversity-screen_exact_seed0`
screened iid at T in {0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0}, the E3 temperature
mix, and prefix-balanced at T=0.7 on `dev_exploration.jsonl` (100 tasks, K=32,
seed 0, pre-GRPO adapter `sft_atomic_r32_pilot_aw4_cont`).

Selection rules, applied to dev only:

- **T\*** := the iid temperature minimising `|distinct_programs_per_task(iid@T) −
  distinct_programs_per_task(prefix_balanced@0.7)|`; ties resolved to the lower
  temperature. Result: prefix-balanced 9.63 distinct/task, iid@2.0 9.88, so
  **T\* = 2.0**.
- **T_best** := the iid temperature with the highest dev `pass@32`.
  Result: **T_best = 3.0** (dev pass@32 0.390).

No confirmatory data was touched during selection.

## 3. Frozen test set

- File: `artifacts/data/pilot/confirmatory_heldout.jsonl`
- SHA256: `4a2e4fd1c7d1e9766ddc23c43014713beb8ddb2f5dc30a518c209137dca96d45`
- 300 tasks, PLAN, depth exactly 3, held-out primes 11 and 17, degree caps 2
  and 3, order-sensitivity required, generator seed 20260801.
- Zero exact-key collisions with any existing split, including the July
  `final_like_heldout.jsonl`.
- Never opened before this preregistration was frozen.

## 4. Search arms

All arms use K = 32 candidates per task, programs of exactly `max_steps`
operations, and the exact verifier applied only to complete programs.

| arm | method | role |
|---|---|---|
| A1 | `iid_action@0.7` | published control (E1) |
| A2 | `iid_action@2.0` | diversity-matched iid (T\*) |
| A3 | `iid_action@3.0` | best iid on dev (T_best) |
| A4 | `prefix_balanced_action@0.7` | structured search (E4) |
| A5 | `temp_mix_action@0.2,0.6,1.0,1.4` | temperature mix (E3) |

## 5. Training

GRPO per `docs/03` Phase G and `docs/06` section 6.5: 400 steps, group size 8,
learning rate 1e-5, sampling temperature 0.7, KL beta 0.01, exact binary reward,
checkpoints every 50 steps, seeds 0, 1, 2, both branches starting from the same
`sft_atomic_r32_pilot_aw4_cont` adapter, separate adapters per branch, prompts
from `rl_train.jsonl`. No confirmatory task is seen during training or selection.

## 6. Hypotheses and decision rules

Statistics per `docs/07` section 7.7: paired per-task differences, 10,000-fold
bootstrap, aggregation inside a seed then across seeds, exact McNemar on
discordant tasks. Three evaluation seeds (0, 1, 2).

### Q1 — primary: does structure beat matched diversity?

Contrast **A4 − A2 on the shared pre-GRPO adapter**, so the policy is identical
and only the search strategy differs.

- 95% CI lower bound > 0 → structure contributes beyond candidate diversity.
- CI contains 0 → the advantage is **not distinguishable from a diversity
  effect**; the July conclusion does not survive.
- CI upper bound < 0 → diversity-matched iid is superior.

### Q2 — secondary: does GRPO contribute anything?

For each branch b in {iid, prefix}: `pass@32(arm_b, adapter_b) −
pass@32(arm_b, pre-GRPO adapter)`, where arm_iid = A1 and arm_prefix = A4.

### Q3 — secondary: reproduction of the published design

A4 on the prefix-GRPO adapter minus A1 on the iid-GRPO adapter — the July
diagonal, on data that has never been opened.

### Q4 — secondary: is any iid arm better than structure?

A3 − A4 on the shared pre-GRPO adapter.

Holm correction is applied across the secondary family {Q2-iid, Q2-prefix, Q3,
Q4}. Q1 is not corrected.

## 7. Cost reporting

Candidate budget K is equal across arms by construction. Model scoring passes
are not, and are reported per arm. A separate equal-model-pass table compares
each iid arm at the K that matches A4's measured pass count.

## 8. What would make this series negative

A negative result is a result. If Q1's CI contains zero, the report states that
prefix-balanced action search shows no advantage over temperature-matched iid
sampling at equal candidate budget, and that the July +16 pp is explained by
candidate diversity plus an inference-time search effect rather than by
structured exploration or by self-improvement. No arm, temperature, adapter or K
may be re-selected after seeing confirmatory results.

## 9. Design also evaluated, explicitly not confirmatory

The same arm grid is additionally run on the July `final_like_heldout.jsonl` for
continuity with the published numbers. That set was already opened, so those
results are reported as re-analysis only.
