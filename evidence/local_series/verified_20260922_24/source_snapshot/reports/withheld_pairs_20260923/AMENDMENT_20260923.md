# Prospective hardware amendment, 23 September 2026

This amendment is recorded before any new k1/s2 or later model evaluation. It was requested to use the three available university NetBird servers efficiently. It is not a response to the observed k1/s1 effect. The original frozen design, 20 preselected pair sets, training seeds, task files, hyperparameters, matching quotas, 1,000-task holdouts, exact 125-program ranking and primary Hit@32 endpoint remain unchanged.

The original `FREEZE.json` has SHA-256 `99DF70CE2030C18F3E86368E1C4A86EA3704FA729C6C355EE8A90C90D1333DD2`. The source pool has SHA-256 `3C2BD26D16AA7BAA0EB2D6CC277C4FFD97C189B3C3405D6CD9FFEAE325C2550F`. The completed k1/s1 block was trained and evaluated on ccmplanner V100 GPU 0 under the original protocol and remains pre-amendment evidence. Its receipts, checkpoints and raw rankings are not changed.

## Execution rule for new sets

Feasible sets are dispatched in the frozen list order, beginning with k1/s2. Assignment is based only on two fresh GPU/process snapshots at least 60 seconds apart and disk availability, never on any model result. A complete set is bound to one host and physical GPU UUID before its first training. The same GPU runs both arms of all three seed pairs and their evaluations. GPU priority is free V100 on `ccmplanner`, then free V100 on `cds2`, then RTX 3060 GPU 0 on `cdsserver`. If a V100 is occupied by another user's task, it is not available and the next free allowed GPU may take the next set. `cdsserver` RTX 2080 Ti GPUs 1 and 2 are reserved for the separate entropy experiment.

On V100, use the original frozen FP16 training runner. On RTX 3060, use a separately hashed amended runner with the same FP16 autocast, FP32 master weights, dynamic loss scaling, optimizer, schedule, token budget and training data. Its only permitted computation-path change is the GPU type guard. The amended runner must record actual GPU name and UUID. The original frozen module and its hash must remain intact. No BF16 results enter this series.

The coordinator writes a per-set allocation receipt with host, GPU UUID, device index, precision, code hash, input hash, assignment time and the two availability snapshots before training. A started set cannot migrate to a different GPU. A failed or partial set keeps its evidence. Only jobs independently accepted as complete may be skipped on resume.

## Analysis boundary

Report k1/s1 separately as pre-amendment evidence. For new sets, report paired seed-level effects per set and identify GPU type. Do not report one pooled inferential estimate across V100 and RTX devices without a prospectively specified device-stratified analysis. With three seed pairs per set the minimum attainable two-sided exact sign-flip p-value is 0.25. The infeasible k5/s5 set remains excluded with its original reason and is not replaced.
