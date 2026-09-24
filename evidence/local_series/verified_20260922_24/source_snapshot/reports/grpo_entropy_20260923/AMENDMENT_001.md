# Technical correction before any entropy-ablation training

The first three-step benchmark exited before task loading and before an
optimizer step. It used RTX 2080 Ti GPU 1. PyTorch 2.7.1 reported
`torch.cuda.is_bf16_supported()` as true because its default permits BF16
emulation. The original shared model loader would therefore select BF16,
contrary to the FP16 protocol. The benchmark container, traceback, and
`FAILED.json` are retained.

The new experiment-only runner now loads both the trainable and reference
models with explicit `torch.float16`. Its GPU check uses
`is_bf16_supported(including_emulation=False)`. The original project loader
and all historical results remain unchanged. The retry uses a new run name.
No reward, held-out score, or pair outcome was inspected before this fix.
