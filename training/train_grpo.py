"""GRPO training on the modular-composition sandbox with a noisy (alpha, beta) checker.

Launch (8 GPUs):
  accelerate launch --num_processes 8 training/train_grpo.py --alpha 1.0 --beta 0.0
"""

import argparse
import importlib.metadata
import json
import os
from collections import defaultdict

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from modcomp.checker import is_correct, noisy_verdict, noisy_verdict_from_event
from modcomp.gen import make_dataset
from modcomp.metrics import base_solved_summary, summarize_checker_counts


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
        choices=["iid", "hashed"],
        default="iid",
        help=(
            "iid: reproducible rollout-event noise (primary experiment); "
            "hashed: legacy answer-keyed quenched-noise ablation"
        ),
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
    ap.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.25)
    ap.add_argument(
        "--generation-backend",
        choices=["auto", "transformers", "vllm"],
        default="auto",
        help="auto uses vLLM on CUDA and Transformers model.generate elsewhere",
    )
    ap.add_argument(
        "--precision",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
        help="auto uses bf16 on CUDA, fp16 on MPS, and fp32 on CPU",
    )
    ap.add_argument(
        "--vllm-enable-sleep-mode",
        action="store_true",
        help="offload colocated vLLM state between phases to reduce peak VRAM use",
    )
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
    if not 0 <= args.alpha <= 1 or not 0 <= args.beta <= 1:
        raise SystemExit("--alpha and --beta must lie in [0, 1]")
    if args.num_generations < 2 or args.num_generations % 2:
        raise SystemExit("--num-generations must be an even integer >= 2")
    use_vllm = args.generation_backend == "vllm" or (
        args.generation_backend == "auto" and torch.cuda.is_available()
    )
    if use_vllm and not torch.cuda.is_available():
        raise SystemExit("vLLM training requires CUDA; use --generation-backend transformers")
    if args.vllm_enable_sleep_mode and not use_vllm:
        raise SystemExit("--vllm-enable-sleep-mode requires the vllm generation backend")
    use_mps = torch.backends.mps.is_available()
    precision = args.precision
    if precision == "auto":
        precision = "bf16" if torch.cuda.is_available() else "fp16" if use_mps else "fp32"
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.eval_size % world_size:
        raise SystemExit(
            f"--eval-size ({args.eval_size}) must be divisible by WORLD_SIZE ({world_size}) "
            "to prevent duplicated distributed evaluation rows"
        )
    run_name = args.run_name or (
        f"a{args.alpha}_b{args.beta}_p{args.prime_max}"
        f"_k{args.k_min}-{args.k_max}_{args.style}_t{args.temperature}_s{args.seed}"
    )
    out_dir = os.path.join("runs", run_name)

    train_rows = make_dataset(
        args.train_size,
        seed=args.seed,
        split="train",
        k_min=args.k_min,
        k_max=args.k_max,
        prime_max=args.prime_max,
        style=args.style,
    )
    eval_rows = make_dataset(
        args.eval_size,
        seed=args.seed + 10_000,
        split=args.eval_split,
        k_min=args.k_min,
        k_max=args.k_max,
        prime_max=args.prime_max,
        style=args.style,
    )
    train_ds = Dataset.from_list(
        [
            {**row, "problem_index": index, "is_eval": False}
            for index, row in enumerate(train_rows)
        ]
    )
    eval_ds = Dataset.from_list(
        [
            {**row, "problem_index": index, "is_eval": True}
            for index, row in enumerate(eval_rows)
        ]
    )

    capture_eval = False
    captured_correctness = defaultdict(list)
    checker_counts = defaultdict(int)
    noise_call_index = 0
    trainer = None

    def reward_noisy_checker(
        completions,
        prompts,
        p,
        target_A,
        target_B,
        problem_index,
        is_eval,
        trainer_state,
        **kw,
    ):
        nonlocal noise_call_index
        global_step = trainer_state.global_step
        rank = trainer.accelerator.process_index if trainer is not None else 0
        call_index = noise_call_index
        noise_call_index += 1
        rewards = []
        for rollout_index, (c, pr, pp, ta, tb, problem_id, eval_flag) in enumerate(
            zip(completions, prompts, p, target_A, target_B, problem_index, is_eval)
        ):
            if args.noise_mode == "iid":
                event_id = (
                    f"{args.seed}|{global_step}|{rank}|{call_index}|"
                    f"{int(problem_id)}|{rollout_index}"
                )
            correct = is_correct(c, pp, ta, tb)
            if args.noise_mode == "iid":
                accepted, _ = noisy_verdict_from_event(
                    correct,
                    event_id,
                    args.alpha,
                    args.beta,
                )
            else:
                accepted, _ = noisy_verdict(
                    c, pr, pp, ta, tb, args.alpha, args.beta, seed=args.seed
                )
            rewards.append(float(accepted))
            if not bool(eval_flag):
                if correct:
                    count_key = "tp" if accepted else "fn"
                else:
                    count_key = "fp" if accepted else "tn"
                checker_counts[count_key] += 1
        return rewards

    # Weight 0.0: logged as a metric (true noiseless accuracy), never trained on.
    def true_accuracy(completions, prompts, p, target_A, target_B, problem_index, **kw):
        scores = [
            float(is_correct(c, pp, ta, tb))
            for c, pp, ta, tb in zip(completions, p, target_A, target_B)
        ]
        if capture_eval:
            for index, score in zip(problem_index, scores):
                if int(index) >= 0:
                    captured_correctness[int(index)].append(score)
        return scores

    cfg_kwargs = dict(
        output_dir=out_dir,
        run_name=run_name,
        seed=args.seed,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        warmup_steps=5,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.num_generations,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.kl_beta,
        loss_type="grpo",
        vllm_importance_sampling_correction=False,
        reward_weights=[1.0, 0.0],
        use_vllm=use_vllm,
        bf16=precision == "bf16",
        fp16=precision == "fp16",
        dataloader_pin_memory=not use_mps,
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
    if use_vllm:
        cfg_kwargs.update(
            # Transformers.generate has no vLLM-style string stop parameter.
            generation_kwargs={"stop": ["\nProblem:", "\nSolved:", "\n\n\n"]},
            vllm_mode="colocate",
            vllm_max_model_length=1024,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_enable_sleep_mode=args.vllm_enable_sleep_mode,
        )
    cfg = GRPOConfig(**cfg_kwargs)

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
    os.makedirs(out_dir, exist_ok=True)
    if trainer.accelerator.is_main_process:
        with open(os.path.join(out_dir, "run_config.json"), "w") as f:
            json.dump(
                {
                    **vars(args),
                    "checker_signal": args.alpha - args.beta,
                    "noise_definition": (
                        "one answer-independent deterministic draw per rollout event"
                        if args.noise_mode == "iid"
                        else "legacy answer-keyed quenched noise"
                    ),
                    "resolved_generation_backend": "vllm" if use_vllm else "transformers",
                    "resolved_precision": precision,
                    "accelerator_device": (
                        "cuda" if torch.cuda.is_available() else "mps" if use_mps else "cpu"
                    ),
                    "package_versions": {
                        package: (
                            importlib.metadata.version(package)
                            if package != "vllm" or use_vllm
                            else None
                        )
                        for package in ("torch", "transformers", "trl", "vllm", "peft")
                    },
                },
                f,
                indent=2,
            )
    captured_correctness.clear()
    capture_eval = True
    trainer.evaluate()  # step-0 baseline; required for gain and time-to-target curves
    capture_eval = False
    baseline_capture = {key: list(values) for key, values in captured_correctness.items()}

    trainer.train()

    captured_correctness.clear()
    capture_eval = True
    trainer.evaluate()  # explicit endpoint, including when max_steps is not a multiple of eval_steps
    capture_eval = False
    final_capture = {key: list(values) for key, values in captured_correctness.items()}

    rank = trainer.accelerator.process_index
    rank_path = os.path.join(out_dir, f"forgetting_rank{rank}.json")
    with open(rank_path, "w") as f:
        json.dump(
            {"baseline": baseline_capture, "final": final_capture},
            f,
            indent=2,
        )
    checker_rank_path = os.path.join(out_dir, f"checker_counts_rank{rank}.json")
    with open(checker_rank_path, "w") as f:
        json.dump(dict(checker_counts), f, indent=2)
    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        merged = {"baseline": defaultdict(list), "final": defaultdict(list)}
        for process_index in range(trainer.accelerator.num_processes):
            with open(os.path.join(out_dir, f"forgetting_rank{process_index}.json")) as f:
                part = json.load(f)
            for stage in ("baseline", "final"):
                for problem_index, counts in part[stage].items():
                    merged[stage][problem_index].extend(counts)

        selection, baseline, final = {}, {}, {}
        selection_k = args.num_generations // 2
        measurement_k = args.num_generations - selection_k
        for problem_index, values in merged["baseline"].items():
            final_values = merged["final"][problem_index]
            expected = args.num_generations
            if len(values) != expected or len(final_values) != expected:
                raise RuntimeError(
                    f"problem {problem_index}: expected {expected} eval generations, "
                    f"got baseline={len(values)}, final={len(final_values)}"
                )
            selection[problem_index] = {
                "correct": int(sum(values[:selection_k])),
                "n": selection_k,
            }
            baseline[problem_index] = {
                "correct": int(sum(values[selection_k:])),
                "n": measurement_k,
            }
            final[problem_index] = {
                "correct": int(sum(final_values[:measurement_k])),
                "n": measurement_k,
            }

        forgetting = base_solved_summary(selection, baseline, final)
        forgetting["definition"] = (
            "base-solved problems are selected with one baseline batch; pass@1 change is "
            "estimated with an independent baseline half and a final batch. Nonrediscovery "
            "is a finite-K diagnostic, not by itself a causal forgetting estimate."
        )
        forgetting["selection_k"] = selection_k
        forgetting["measurement_k"] = measurement_k
        with open(os.path.join(out_dir, "forgetting.json"), "w") as f:
            json.dump(forgetting, f, indent=2)

        merged_checker_counts = defaultdict(int)
        for process_index in range(trainer.accelerator.num_processes):
            with open(os.path.join(out_dir, f"checker_counts_rank{process_index}.json")) as f:
                part = json.load(f)
            for key, value in part.items():
                merged_checker_counts[key] += int(value)
        checker_diagnostics = summarize_checker_counts(merged_checker_counts)
        zero_std_history = [
            float(entry["train/frac_reward_zero_std"])
            for entry in trainer.state.log_history
            if "train/frac_reward_zero_std" in entry
        ]
        checker_diagnostics.update(
            {
                "configured_alpha": args.alpha,
                "configured_beta": args.beta,
                "configured_signed_alignment": args.alpha - args.beta,
                "noise_mode": args.noise_mode,
                "zero_variance_group_rate": (
                    sum(zero_std_history) / len(zero_std_history)
                    if zero_std_history
                    else None
                ),
                "zero_variance_group_rate_source": (
                    "mean of TRL global train/frac_reward_zero_std over logged steps"
                    if zero_std_history
                    else "TRL metric unavailable"
                ),
            }
        )
        with open(os.path.join(out_dir, "checker_diagnostics.json"), "w") as f:
            json.dump(checker_diagnostics, f, indent=2)

    if not args.no_save:
        trainer.save_model(os.path.join(out_dir, "final"))


if __name__ == "__main__":
    main()
