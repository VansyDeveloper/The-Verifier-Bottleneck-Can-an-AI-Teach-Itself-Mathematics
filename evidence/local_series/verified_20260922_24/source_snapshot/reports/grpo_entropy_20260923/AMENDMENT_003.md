# FP32 numerical-stability amendment before any accepted pair

The initial FP16 400-step controls failed at optimizer steps 4 and 17.
Their recorded losses were finite, but the gradient norms were not. Both
containers exited with code 1. The host queues stopped without starting a
treatment arm or an evaluation. Their `FAILED.json`, generated answers,
metrics, Docker state, and logs are retained. Neither control is accepted
or included in a scientific comparison.

The trainable LoRA parameters were FP32, but the frozen base-model layers
computed in FP16. A read-only FP32 load of both the trainable and reference
models on the assigned RTX 2080 Ti used 4.667 GiB of allocated GPU memory.
The repaired comparison loads both models explicitly in FP32 and uses the
same FP32 code for control and entropy arms and exact ranking. The base
model, atomic adapter, task files, reward, random seeds, group size,
temperature, KL coefficient, learning rate, and entropy coefficient remain
unchanged. New run names end in `_fp32`, so failed evidence cannot be
overwritten. The original FP16 timing freeze is superseded. A new speed
measurement will determine 400 versus 150 steps before new full runs.

No held-out outcome was computed or inspected before this amendment.
