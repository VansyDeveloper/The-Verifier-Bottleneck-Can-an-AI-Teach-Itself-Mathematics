# Preregistration — can self-training help once exploration is adequate?

Frozen 2026-08-01, before any evaluation on `confirmatory_heldout_v2.jsonl`.
Not edited after freezing. Second and final confirmatory set of this project.

## 1. What the D-018 series left open

D-018 measured the GRPO contribution at **−0.11 pp** (iid branch) and **−1.89 pp**
(prefix branch), neither CI excluding zero. That null has two possible causes and
D-018 cannot tell them apart:

- **(a) self-training genuinely cannot expand reachability here**, or
- **(b) the training sampler was too cold to produce a learning signal.**

Reading (b) is concrete and quantified. GRPO sampled candidates at T=0.7, where
the action policy emits only ~3.1 distinct programs per 32 candidates and only
25–27 of 400 groups contained any correct program at all. A policy-gradient
update from a group with no positive reward carries no information about which
program to prefer.

The extended dev sweep (run `20260801T...diversity-screen-ext`, dev only) shows how
much headroom the sampler had:

| iid temperature | dev pass@32 | distinct/task |
|---|---:|---:|
| 0.7 (used for GRPO training) | 0.060 | 3.11 |
| 3.0 | 0.390 | 14.57 |
| **8.0** | **0.500** | 21.92 |
| 100.0 (effectively uniform) | 0.420 | 23.74 |

The interior optimum at T≈8 also establishes that the model's action scores carry
real information: uniform sampling has *more* diversity yet scores lower.

## 2. Selection, already complete, dev only

**T_train := 8.0**, the dev-optimal iid temperature by `pass@32`. Selected on
`dev_exploration.jsonl` (100 tasks, seed 0, pre-GRPO adapter) before this file was
frozen. No confirmatory data of any version informed it.

## 3. Frozen test set

- File: `artifacts/data/pilot/confirmatory_heldout_v2.jsonl`
- SHA256: `e75320e0cb6761c79ace95faa68fe70998d213481ca27f986be5351074422892`
- 300 PLAN tasks, depth exactly 3, held-out primes 11 and 17, degree caps 2 and 3,
  order-sensitivity required, generator seed 20260802.
- Zero exact-key collisions with `confirmatory_heldout.jsonl` or any other split.
- Never opened before this preregistration was frozen.

## 4. Training

One new branch, otherwise identical to D-018: GRPO, 400 steps, group size 8,
lr 1e-5, KL beta 0.01, exact binary reward, seeds 0/1/2, from the same
`sft_atomic_r32_pilot_aw4_cont` adapter — with **sampling temperature 8.0 instead
of 0.7**. Adapters `grpo_iid_action_hot_400_seed{0,1,2}`.

Nothing else changes: same prompts, same reward, same steps, same group size,
same LoRA. The single manipulated variable is the exploration temperature of the
training sampler.

## 5. Evaluation

Arms at K=32 on the v2 set: `iid@0.7`, `iid@3.0`, `iid@8.0`, `iid@100.0`,
`prefix_balanced@0.7`. Adapters: pre-GRPO, `grpo_iid_action_400_seed{S}` (the cold
D-018 branch), `grpo_iid_action_hot_400_seed{S}`. Seeds 0/1/2.

## 6. Hypotheses and decision rules

Statistics as in D-018: paired per-task differences, 10,000-fold hierarchical
bootstrap, exact McNemar, three seeds.

### Q5 — primary: does adequately-explored self-training help?

`pass@32(iid@8.0 arm, hot-GRPO adapter) − pass@32(iid@8.0 arm, pre-GRPO adapter)`.

- CI lower bound > 0 → self-training expands effective reachability once the
  training sampler explores; the D-018 null was cause (b), an artefact of an
  under-tempered sampler.
- CI contains 0 → the null is cause (a): self-training does not expand
  reachability in this sandbox even with adequate exploration.
- CI upper bound < 0 → self-training actively harms.

### Q6 — secondary: was the cold sampler the problem?

`pass@32(iid@8.0, hot-GRPO) − pass@32(iid@8.0, cold-GRPO)`.

### Q7 — secondary: does structure lose to random search?

`pass@32(iid@100.0, pre-GRPO) − pass@32(prefix@0.7, pre-GRPO)`. The dev sweep says
uniform random program sampling beats prefix-balanced; this tests it on frozen
data.

### Q8 — secondary: replication of Q4 at the true dev optimum

`pass@32(iid@8.0, pre-GRPO) − pass@32(prefix@0.7, pre-GRPO)`. D-018's Q4 used the
preregistered T_best = 3.0, which the extended sweep later showed was not the
optimum, so +9.78 pp understates the gap.

Holm correction across {Q6, Q7, Q8}. Q5 is not corrected.

## 7. What would make this negative

If Q5's CI contains zero, the report states that in this sandbox, with an exact
verifier and an adequately exploring sampler, 400 steps of GRPO still produce no
measurable expansion of effective reachability — and that the project's central
question is answered in the negative for this task family. That answer is
retained in full. No arm, temperature, adapter or K may be re-selected afterwards.

`confirmatory_heldout_v2` is the last frozen set of this project. Any further
claim requires a newly generated one and a new preregistration.
