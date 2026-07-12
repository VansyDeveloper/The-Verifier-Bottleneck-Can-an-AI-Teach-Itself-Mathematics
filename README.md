# Verifier bottleneck: A/B/AB sandbox

The question is narrow: can self-training help a model solve a held-out composition, or does it only make answers already in reach easier to sample?

We vary two things. The verifier controls which generated answers return to training. The sampling budget controls how much the model gets to explore. This repository only contains a toy dataset, an exact oracle, and a small noisy-verifier example. The training pipeline is still a plan.

## Task and toy code

For a prime `p`, every map is `g(z) = a*z + b (mod p)` with `a != 0`.

- A: collapse `g3(g2(g1(x)))` into one map `A*x + B`.
- B: recover `x` from `A*x + B = y (mod p)`.
- AB: recover `x` from the output of a chain. Collapse-first and backwards inversion are both valid.

[`examples/dataset_checker.ipynb`](examples/dataset_checker.ipynb) builds one public/private example of each mode. It parses the last `FINAL` answer and checks it with exact modular arithmetic. The notebook also has a deterministic noisy verifier with

`alpha = P(accept | correct)` and `beta = P(accept | wrong)`.

Its random draw is keyed only by `run_seed | step | problem_id | rollout_id`, never by the answer text. The perfect oracle remains separate from the reward used for training.

## Model choice

The proposal suggests small open Qwen2.5-Math and Llama-3.2 models. Our first pilot candidate is `Qwen2.5-Math-1.5B-Instruct`; `Llama-3.2-3B-Instruct` is a possible replication model later. These are working choices, not final decisions.

The pilot has one job: find a regime where the model learns A and B, while AB remains difficult but occasionally reachable with several samples. If the math checkpoint already solves nearly every AB task, we try a weaker general checkpoint or deeper chains. If it cannot learn A and B, we simplify the task. After that calibration we freeze the exact checkpoint, tokenizer and chat template, seeds, response limits, decoding settings, and compute budget.

## Minimal tuning plan

We do not need RFT and GRPO at the same time. The first experiment uses one simple path:

1. Fine-tune A and B examples with LoRA SFT. No AB solution is included. Save the resulting atomic checkpoint as `M0`.
2. Generate one fixed pool of 64 answers for each AB training prompt. `K=8` is the first eight answers from that same pool, not a second sampling run.
3. Starting from the same `M0`, compare no update, exact filtering, and a rate-matched random filter.
4. For each filtered condition, run one rejection-SFT/LoRA update. Match selected token counts and optimizer steps so a condition does not win simply by seeing more training text.
5. Evaluate all checkpoints with the perfect oracle on fresh AB tasks and the original A/B tests.

The pilot will choose the LoRA rank, learning rate, batch size, and number of epochs. We will record those values before the comparison and keep them fixed. Inventing precise numbers before we know the model, hardware, and calibration result would only make the plan look more certain than it is.

## Noise comes after the core run

The notebook already demonstrates the mechanism, but noisy-reward training is not required for the first result. If the exact-filter control works, the next small comparison is:

- clean: `(alpha, beta) = (1.0, 0.0)`;
- false-negative heavy: `(0.3, 0.0)`;
- false-positive heavy: `(1.0, 0.7)`.

These cells test different failure modes; they are not expected results. GRPO is a later extension only if we have enough compute and a reason to compare algorithms.

## What we will measure

Evaluation uses fresh, disjoint prompts and identical decoding before and after tuning. We report `pass@1`, fixed-budget `pass@16`, newly solved tasks minus lost tasks, A/B retention, and AB transfer to deeper chains or a held-out prompt format.

- Higher `pass@1` with flat `pass@16` is evidence consistent with sharpening.
- Higher `pass@16` with positive net held-out gains is operational, fixed-budget expansion. It does not prove that the base model assigned zero probability to the answer.
- Lower `pass@16` is collapse under this evaluation budget.

With three seeds we can report the mean, spread, and whether the direction repeats. We will not call that statistical significance.

The short deck is available as [`presentation/verifier-bottleneck.html`](presentation/verifier-bottleneck.html) and [`presentation/verifier-bottleneck.pdf`](presentation/verifier-bottleneck.pdf). Model training, a model grid, real-math benchmarks, and prolonged RL are outside this mentor demo.
