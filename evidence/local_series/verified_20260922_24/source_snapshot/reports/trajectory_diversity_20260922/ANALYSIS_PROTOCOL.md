# Analysis protocol, 22 September 2026

This work is separate from the frozen stage-4 confirmation. No model,
checkpoint, task split, or original receipt is modified. All curves below are
post-hoc descriptive analyses of saved final-A exact rankings.

## Exact ranking analysis

For each of the six paired replicates, use the same 1,000 final-A depth-3
tasks in `atomic_control` and `composition_distill`. Read all 40 receipt-bound
shards per arm. Check SHA-256, task identity, 125 unique depth-3 programs per
task, ranks 1 through 125, finite scores, correctness flags, and agreement
between the saved metric and the recomputed first correct rank. The primary
quality curve is H(K), the fraction of tasks with first correct rank at most K,
for every integer K from 1 through 125. Report six replicate curves and the
mean across replicates. H(32) must match the frozen primary table.

Scores are sums of log probabilities of the selected operation tokens. A
softmax across the 125 complete-program scores defines a distribution
conditional on these 125 admissible programs. Report its Shannon entropy in
nats, effective number of programs exp(H), total mass on correct programs,
and conditional entropy among correct programs. The conditional entropy is
undefined only if no correct program exists, which must fail validation here.
These quantities do not measure entropy over the model's full vocabulary.

The unit for paired summaries is the training replicate, n=6. The 1,000 tasks
per replicate are not treated as 6,000 independent trainings. Other K and
entropy comparisons are exploratory. No selection or hyperparameter change
will be made based on final-A results.

## Trajectory selection experiment

Use the frozen 2,250-row training trajectory pool. Each row belongs to one
distinct task, so changing the selected trajectories necessarily changes the
training-task set. This is a dataset-composition intervention, not an
individual-task-matched trajectory intervention. Predefine the selection
size, candidate features, random seeds, tie-breaking, training budget and
held-out endpoint before running or inspecting new held-out outcomes. Both
arms must use the same trajectory count, atomic replay fraction, optimizer
steps, effective batch, epochs and number of loss-bearing target tokens.
Preserve all failed attempts and receipts. New evaluation tasks must be
separate from the already examined final-A set.

## Withheld-pair extension

Treat k=1,2,3,5 as a separate preregistered follow-up. Sample 4-5 pair sets
per k under explicit feasibility constraints, use several independent model
seeds for each set, and compare with a random-removal control of equal size
and training budget. Lock pairs and final tasks before training. Analyze
variation across both set and seed. Do not claim that the existing three-pair
result represents every possible withheld set.
