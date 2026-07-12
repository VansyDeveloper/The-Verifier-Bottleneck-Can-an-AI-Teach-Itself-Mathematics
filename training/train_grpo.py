"""GRPO training on the modular-composition sandbox with a noisy (alpha, beta) checker.

Launch (8 GPUs):
  accelerate launch --num_processes 8 training/train_grpo.py --alpha 1.0 --beta 0.0
"""

import argparse
import os

# No cu12 nvcc on this box -> flashinfer JIT can't build. Use FlashAttention +
# the non-flashinfer sampler so vLLM never tries to compile kernels at runtime.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

import random as _rnd
from modcomp.checker import is_correct, noisy_verdict
from modcomp.gen import make_dataset


def parse_args():
    ap = argparse.ArgumentParser()
    # The main sandbox experiments use the small dense Qwen3-0.6B base model.
    # A separate early H2 pilot used Qwen3-1.7B-Base; pass it via --model when
    # reproducing that series.
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--alpha", type=float, default=1.0, help="P(checker accepts | correct)")
    ap.add_argument("--beta", type=float, default=0.0, help="P(checker accepts | wrong)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--noise-mode",
        choices=["hashed", "fresh"],
        default="hashed",
        help="hashed: quenched noise keyed on (seed,prompt,completion); fresh: new Bernoulli draw per verdict",
    )
    ap.add_argument("--train-size", type=int, default=100_000)
    ap.add_argument("--eval-size", type=int, default=128)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=2)
    ap.add_argument(
        "--prime-max",
        type=int,
        default=97,
        help="difficulty ceiling: only use primes <= this (curriculum knob)",
    )
    ap.add_argument(
        "--style",
        choices=["verbose", "compact"],
        default="verbose",
        help="few-shot solution format; 'compact' is O(k) tokens for deep k",
    )
    ap.add_argument(
        "--eval-split",
        choices=["eval", "train"],
        default="eval",
        help="'eval' = held-out primes (tests cross-prime transfer); "
        "'train' = same primes, disjoint instances (tests composition only)",
    )
    ap.add_argument("--num-generations", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--kl-beta", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--eval-steps", type=int, default=25)
    ap.add_argument("--per-device-batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-completion-length", type=int, default=256)
    ap.add_argument("--run-name", default=None)
    ap.add_argument(
        "--no-save",
        action="store_true",
        help="skip the final weight save (sweep runs only need TB metrics)",
    )
    ap.add_argument(
        "--lora", action="store_true", help="train LoRA adapters instead of full fine-tuning"
    )
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    return ap.parse_args()


def main():
    args = parse_args()
    run_name = args.run_name or (
        f"a{args.alpha}_b{args.beta}_p{args.prime_max}"
        f"_k{args.k_min}-{args.k_max}_{args.style}_t{args.temperature}_s{args.seed}"
    )
    out_dir = os.path.join("runs", run_name)

    train_ds = Dataset.from_list(
        make_dataset(
            args.train_size,
            seed=args.seed,
            split="train",
            k_min=args.k_min,
            k_max=args.k_max,
            prime_max=args.prime_max,
            style=args.style,
        )
    )
    eval_ds = Dataset.from_list(
        make_dataset(
            args.eval_size,
            seed=args.seed + 10_000,
            split=args.eval_split,
            k_min=args.k_min,
            k_max=args.k_max,
            prime_max=args.prime_max,
            style=args.style,
        )
    )

    def reward_noisy_checker(completions, prompts, p, target_A, target_B, **kw):
        rewards = []
        for c, pr, pp, ta, tb in zip(completions, prompts, p, target_A, target_B):
            if args.noise_mode == "fresh":
                accepted = _rnd.random() < (args.alpha if is_correct(c, pp, ta, tb) else args.beta)
            else:
                accepted, _ = noisy_verdict(
                    c, pr, pp, ta, tb, args.alpha, args.beta, seed=args.seed
                )
            rewards.append(float(accepted))
        return rewards

    # Weight 0.0: logged as a metric (true noiseless accuracy), never trained on.
    def true_accuracy(completions, prompts, p, target_A, target_B, **kw):
        return [
            float(is_correct(c, pp, ta, tb))
            for c, pp, ta, tb in zip(completions, p, target_A, target_B)
        ]

    cfg = GRPOConfig(
        output_dir=out_dir,
        run_name=run_name,
        seed=args.seed,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        warmup_steps=5,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.per_device_batch,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.kl_beta,
        reward_weights=[1.0, 0.0],
        # Stop once the model finishes its answer or starts a fresh problem, so
        # it doesn't ramble into invented few-shot items and waste tokens.
        generation_kwargs={"stop": ["\nProblem:", "\nSolved:", "\n\n\n"]},
        use_vllm=True,
        vllm_mode="colocate",
        vllm_max_model_length=1024,
        vllm_gpu_memory_utilization=0.25,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        # Never write mid-run optimizer checkpoints: a single 1.7B checkpoint is
        # ~20GB (weights+optimizer) and the shared fuse mount runs near-full.
        # Curriculum stages still persist weights-only via trainer.save_model(final).
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        report_to=["tensorboard"],
        log_completions=True,
        num_completions_to_print=0,
    )

    peft_config = None
    if args.lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules="all-linear",
            bias="none",
        )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[reward_noisy_checker, true_accuracy],
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
    )
    trainer.train()
    if not args.no_save:
        trainer.save_model(os.path.join(out_dir, "final"))


if __name__ == "__main__":
    main()
