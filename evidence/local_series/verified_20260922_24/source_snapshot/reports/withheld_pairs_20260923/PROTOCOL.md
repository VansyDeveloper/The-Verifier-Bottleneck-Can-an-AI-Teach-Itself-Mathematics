# Frozen design: withholding operation pairs

This stage is a new training-data intervention. It does not alter or reinterpret the completed trajectory-diversity comparison. No outcome-dependent substitution of operation-pair sets, seeds, tasks, or hyperparameters is allowed.

## Inputs and conditions

The source is the frozen pool of 2,250 correct stage-4 trajectories. The 20 pair sets for `k=1,2,3,5`, five sets per k, are fixed in `../trajectory_diversity_20260922/WITHHELD_PAIR_SETS.json`. All sets are audited in this order. An infeasible set is recorded and not replaced.

For each set and training seed 85000, 85001, and 85002, the withheld arm removes every trajectory containing any of the ordered adjacent operation pairs. The random arm removes the same number of source trajectories in each `(depth,p)` cell by seeded uniform sampling. Each arm then selects exactly 250 retained trajectories at each depth 2, 3, and 4. Selection quotas match the two arms in every `(depth,p,target-answer-token-length)` stratum. This ensures equal record counts, depth and modulus composition, and loss-bearing composition tokens. If these quotas cannot be filled, the set is marked infeasible before training. Both arms use the same seed-specific atomic replay, frozen atomic checkpoint, number of epochs, optimizer steps, effective batch, and training target-token budget. Only source-data availability differs.

The new V100 experiment uses FP16 autocast with FP32 master weights and a dynamic gradient scaler initialized at 128. It retains the original learning rate, LoRA settings, replay fraction, batches, and epochs. A non-finite loss or gradient stops the run. This is a separately frozen precision protocol. No BF16 training from the prior comparison enters this study. All arms and sets in this protocol must use the same V100 GPU type and FP16 implementation.

## Held-out tasks and endpoints

Before any new training, generate 1,000 family-A, depth-3 tasks in each of two strata for every feasible set. For the withheld stratum, at least one exact shortest correct program among the 125 admissible depth-3 programs must contain an excluded pair. For the control stratum, none may contain an excluded pair. A withheld-stratum task may also have an alternative correct program without an excluded pair. The fraction of such tasks is recorded before training and disclosed when interpreting task-level Hit@32. The two strata are disjoint from training data, previous evaluations, the atomic reference, and each other in both task and raw-state fingerprints. Generator seeds are `86000 + 100*k + 2*subset + offset`, with offset 0 for withheld and 1 for control. Training outcomes cannot affect task generation.

Exact ranking of all 125 programs on common tasks yields Hit@32. The primary comparison is withheld minus random on the withheld stratum, paired by training seed. Hit@K for K=1..125, performance on control tasks, exact correct-program mass, and per-operation atomic forgetting are secondary. A complete set has three paired seed differences. Individual tasks are not independent training replications. The first set alone does not support a conclusion about k or about all 20 sets.

## Pre-training amendment

The initial generator required every shortest correct program to contain an excluded pair. Before any model training or evaluation, this was found to be infeasible for the preselected pair `SH1, AC1`: no task was accepted in 10,000 attempts with an empty registry. The condition was therefore changed to at least one correct program containing the pair. Six task files produced before this amendment are preserved in `artifacts/withheld_pairs_20260923/recovery/pre_amendment_all_correct_condition` and are excluded from the frozen study. No pair set, model result, training seed, or hyperparameter was changed in response to outcomes.

## Integrity and stop rules

Before training, freeze the selected IDs, removed IDs, task files, source and code SHA-256, seeds, and configuration. A completed model is accepted only if its receipt has finite loss, expected steps and tokens, intact checkpoint and binding hashes. A partial or failed model is preserved with logs. Never start a duplicate process. If V100 GPU 0 is not free, preflight fails, or the measured runtime cannot fit a paired block, do not silently change GPU or precision. The first executable block is k=1, subset=1. Subsequent sets follow the fixed file order, not observed effects.
