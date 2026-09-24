# Frozen design for trajectory selection, 22 September 2026

This is a new exploratory dataset-composition intervention. It does not alter
the prior stage-4 confirmation, and its new holdout is not used for model or
hyperparameter selection. The six training seeds are 83000 through 83005.

## Inputs and selection

Use the frozen Qwen3-0.6B atomic export and the 2,250 correct trajectories
recorded by the stage-4 confirmation protocol. Choose 750 trajectories per
arm and seed, with exactly 250 each at depths 2, 3, and 4. Within each depth,
allocate quotas proportionally by `(p, target_tokens)` using the largest
remainder rule with lexicographic tie-breaking. The target token length is
computed with the frozen tokenizer and exactly the composition answer used by
the original trainer, including the trace and EOS token. Both arms receive
the same quota in every stratum. The random arm samples uniformly without
replacement within strata using the replicate seed.

The diverse arm uses deterministic greedy maximization of an equally weighted
objective. The pair component is the sum over observed ordered adjacent
operation pairs of `log(1 + count)`, divided by the fixed concavity bound
`P*log(1+T/P)`, where `P` is the number of pair types in the full pool and
`T=1500` adjacent-pair occurrences in a 750-row balanced selection. The
state component is the number of covered classes divided by the number of
classes in the full pool. A class is `(p, floor(4*x_i/p) for each coordinate)`
for an intermediate state. Vectors have three, four, or five coordinates.
Greedy ties are broken by SHA-256 of `seed:trajectory_id`. The selection code
and manifest are frozen before training. A manipulation check requires the
diverse arm's objective to exceed the paired random arm's by at least 0.03
absolute units in every seed. If this fails, do not train and report that the
intended intervention was too weak.

## Training and held-out evaluation

Both arms use the same frozen atomic checkpoint, original stage-4 training
configuration (learning rate 1e-4, two epochs, 20% atomic replay, effective
batch 64, microbatch 4, bf16, LoRA rank 32 / alpha 64 / dropout 0.05), the
same seed-specific atomic replay pool, and the same training code. Verify
before and after training that arms have equal record counts, per-epoch
loss-bearing target tokens, effective batch, epochs, optimizer steps and
actual target tokens seen. An incomplete or failed adapter is preserved and
never silently restarted.

Before training, freeze the holdout generation seed 83100 and source/code
hashes. Only after all twelve training runs pass validation, generate one
1,000-task family-A depth-3 holdout, disjoint in task and raw state
fingerprints from the atomic reference, pilot train/dev data, all old final
splits, and the 2,250-row trajectory pool. All twelve adapters evaluate the
same new tasks. Exactly rank all 125 programs per task and retain score,
correctness, per-shard SHA-256 receipts, and evaluation bindings. Perform the
same frozen atomic plan/apply evaluation for each adapter, broken down by
operation. Do not tune models against new holdout outcomes.

## Statistics and interpretation

Primary endpoint: diverse minus random Hit@32, paired by training seed over
the common 1,000 tasks. Report six per-seed differences, mean, sample SD,
95% t confidence interval, and exact two-sided sign-flip p-value. Six
trainings, not 6,000 tasks, define inferential sample size. Full Hit@K,
correct mass, Shannon entropy conditional on 125 admissible programs,
conditional entropy among correct programs, and per-operation forgetting are
secondary exploratory descriptions. The intervention also changes the task
set because each source trajectory belongs to a distinct task; it cannot
identify an effect of replacing one trajectory while holding tasks fixed.

Failed runs, partial artifacts, integrity problems and all amendments remain
visible. No outcome from the new holdout may cause a configuration change.
