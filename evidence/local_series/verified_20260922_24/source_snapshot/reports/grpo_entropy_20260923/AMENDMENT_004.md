# Activation-checkpointing amendment before any accepted pair

The first two FP32 30-step diagnostics failed with CUDA out-of-memory errors
during the backward pass. Seed 0 failed at its first group. Seed 1 completed
five groups. Their `FAILED.json`, generated answers, metrics, Docker states,
and logs are retained. They are not scientific results.

The trainable FP32 model now uses non-reentrant activation checkpointing and
requires gradients at the inputs of the LoRA-bearing model. Its cache is
disabled during training. The frozen reference model is unchanged. Checkpoint
recomputation preserves the PyTorch random-number state. This is a memory
management change, not a change to the objective, task order, seed, reward,
group size, temperature, learning rate, or entropy coefficient. Both arms
use the same revised code and begin from the same frozen atomic adapter.

New diagnostics use distinct names ending in `_fp32_gc_30steps`, preserving
both earlier FP16 and FP32 failure evidence. If the new diagnostics succeed,
their measured training and evaluation rates determine the common 400- or
150-step duration before any full paired training or held-out evaluation.
No held-out outcomes were inspected before this amendment.

Both revised 30-step diagnostics subsequently exited successfully with
finite recorded metrics. The two maximum late-step times were 18.54 and
31.79 seconds. Exact ranking of ten depth-three training tasks took
36.58 seconds in total. The prospective calculation in
`DURATION_FREEZE_FP32.json` fixed 150 steps for both arms. Its conservative
makespan with a 25 percent margin is 44,600 seconds against 52,244 seconds
available at the decision time. The corresponding 400-step estimate is
84,600 seconds with margin and is not feasible by the deadline.
