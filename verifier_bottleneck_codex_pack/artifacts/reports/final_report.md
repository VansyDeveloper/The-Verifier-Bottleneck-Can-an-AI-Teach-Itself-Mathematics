# Verifier Bottleneck: three-seed confirmatory pilot

## Summary

A three-seed, 150-step GRPO pilot tested whether prefix-balanced action-space
exploration improves discovery and retention of unseen compositions relative to
iid action sampling. The model was Qwen3-0.6B-Base with independent LoRA r=32
adapters. Correctness was determined only by the exact program executor.

On the frozen held-out PLAN depth-3 set over primes 11 and 17:

- mean iid pass@32: **12.67%**;
- mean prefix-balanced pass@32: **28.67%**;
- mean paired effect: **+16.00 percentage points**;
- hierarchical paired bootstrap 95% CI: **[+12.17, +19.83] points**;
- positive direction: **3/3 seeds**;
- pooled exact McNemar p-value: **2.10e-15**.

This is a statistically significant positive confirmatory pilot. It is not the
400-step full protocol.

## Protocol

- Base model: `Qwen/Qwen3-0.6B-Base`.
- Atomic checkpoint: `sft_atomic_r32_pilot_aw4_cont`.
- GRPO: 150 steps, group size 8, exact binary reward, KL beta 0.01.
- Seeds: 0, 1, 2.
- Evaluation: 200 identical frozen tasks per seed, PLAN depth=3, held-out primes
  11/17, K=32.
- Branches were independent and always started from the same atomic checkpoint.
- Held-out data was opened only after training and was never used for tuning.

Action-space exploration scored the five allowed next operations with the model.
IID sampled every action from those conditional probabilities. Prefix-balanced
allocated first actions across the five operations and used the same conditional
sampler for subsequent actions. Both methods had identical candidate K and
program length. Prefix-balanced used more model passes; this cost is reported.

## Held-out results

| Seed | IID pass@32 | Prefix pass@32 | Delta | Bootstrap 95% CI | McNemar p |
|---:|---:|---:|---:|---:|---:|
| 0 | 13.0% | 26.5% | +13.5 pp | [+7.5, +19.5] | 4.19e-5 |
| 1 | 10.5% | 30.0% | +19.5 pp | [+13.0, +26.5] | 4.00e-8 |
| 2 | 14.5% | 29.5% | +15.0 pp | [+8.0, +22.0] | 1.00e-4 |
| Mean | 12.67% | 28.67% | +16.0 pp | [+12.17, +19.83] | 2.10e-15 pooled |

Prefix-only task successes numbered 125 versus 29 iid-only successes across 600
paired seed-task observations.

Mean exact-program coverage increased from approximately **10.44%** to
**22.06%**. Thus the result is not only a pass@k improvement: structured search
covered substantially more valid programs.

## Training behavior

| Seed | IID mean reward | Prefix mean reward | IID positive groups | Prefix positive groups |
|---:|---:|---:|---:|---:|
| 0 | 5.33% | 5.08% | 19/150 | 40/150 |
| 1 | 5.92% | 5.17% | 24/150 | 41/150 |
| 2 | 8.17% | 5.58% | 24/150 | 47/150 |

Prefix-balanced did not consistently increase candidate-level reward during
training. It did increase the number of groups containing at least one correct
program in every seed. This is consistent with a coverage advantage rather than
simple probability sharpening.

## Atomic forgetting

Before GRPO, the shared checkpoint had non-SH1 APPLY pass@1=1.00 and PLAN
pass@1=0.98. After GRPO:

- non-SH1 APPLY remained 1.00 for all six adapters;
- PLAN was 0.97/0.97 for seed 0, 0.97/0.98 for seed 1, and 0.97/0.97 for seed 2
  (iid/prefix respectively).

Maximum PLAN forgetting was one percentage point. There was no material atomic
collapse.

SH1 remained inside PLAN, action search, and exact verification. Under the
user-authorized D-013 protocol, standalone SH1 APPLY semantics were checked by
the exact executor; the earlier full generative APPLY result of 0.83 remains
reported and was not relabelled as a statistical success.

## Cost

The six GRPO runs took 5,208.98 seconds total (about 86.8 minutes of measured run
time). Peak allocated CUDA memory was about 7.81 GB per GRPO run.

For frozen held-out evaluation, average model passes per seed were approximately
1,229 for iid and 2,895 for prefix-balanced. Prefix-balanced therefore used about
2.36 times as many scoring passes despite identical candidate K. The positive
result is valid under candidate-budget normalization, but not under equal
model-pass cost. This is an explicit limitation.

## Interpretation

All preregistered pilot success conditions were met:

1. mean delta exceeded +5 points;
2. the paired 95% CI lower bound was above zero;
3. the effect direction was positive in 3/3 seeds;
4. exact-program coverage increased;
5. atomic forgetting stayed within one point.

The supported conclusion is:

> At fixed candidate budget K=32, prefix-balanced action-space exploration
> substantially and reproducibly increased effective reachability of unseen
> depth-3 programs on held-out primes relative to iid sampling after the same
> 150-step GRPO pilot.

The experiment does not establish zero baseline support, and it does not show
equal efficiency per model pass. A full confirmatory result still requires the
predeclared 400-step protocol.

## Reproduction

```powershell
python scripts/train_action_grpo.py --method iid_action --adapter artifacts/adapters/sft_atomic_r32_pilot_aw4_cont --input artifacts/data/pilot/rl_train.jsonl --steps 150 --group-size 8 --seed 0
python scripts/train_action_grpo.py --method prefix_balanced_action --adapter artifacts/adapters/sft_atomic_r32_pilot_aw4_cont --input artifacts/data/pilot/rl_train.jsonl --steps 150 --group-size 8 --seed 0

# Repeat both commands for seeds 1 and 2.

python scripts/run_action_search_screen.py --adapter artifacts/adapters/grpo_iid_action_pilot_seed0 --input artifacts/data/pilot/final_like_heldout.jsonl --k 32 --seed 0 --methods iid_action
python scripts/run_action_search_screen.py --adapter artifacts/adapters/grpo_prefix_balanced_action_pilot_seed0 --input artifacts/data/pilot/final_like_heldout.jsonl --k 32 --seed 0 --methods prefix_balanced_action

# Repeat held-out evaluation for seeds 1 and 2, then:
python scripts/three_seed_statistics.py --pair 0 <iid-run-0> <prefix-run-0> --pair 1 <iid-run-1> <prefix-run-1> --pair 2 <iid-run-2> <prefix-run-2> --output artifacts/reports/three_seed_heldout_statistics.json --bootstrap 10000
python scripts/analyze_results.py --runs artifacts/runs --output artifacts/reports
python scripts/plot_confirmatory_results.py
```

The exact run paths and every failed attempt are listed in
`artifacts/reports/run_index.csv`.
