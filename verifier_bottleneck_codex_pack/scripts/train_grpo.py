from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import torch

from vbexp.experiment import RunDirectory, load_yaml, seed_everything
from vbexp.io import read_tasks
from vbexp.modeling import completion_log_probability, load_model, load_tokenizer
from vbexp.prompts import build_prompt
from vbexp.verifier import verify

OPERATIONS = ("SH1", "SC2", "REV", "AC1", "AX1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tier", choices=["smoke", "pilot", "full"], default="smoke")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--input", type=Path, default=Path("artifacts/data/smoke/dev_exploration.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_yaml(args.config)
    if args.tier == "smoke" and args.steps > 5:
        raise ValueError("smoke GRPO is limited to 5 steps")
    method = "prefix_balanced" if raw["exploration"]["type"] == "forced_first_operation" else "iid_t07"
    group_size = 4 if args.tier == "smoke" else int(raw["num_generations"])
    adapter = Path("artifacts/adapters/sft_atomic_r16_smoke")
    resolved = {
        **raw,
        "tier": args.tier,
        "seed": args.seed,
        "steps": args.steps,
        "num_generations": group_size,
        "model_name_or_path": "Qwen/Qwen3-0.6B-Base",
        "base_checkpoint": str(adapter),
        "input": str(args.input),
    }
    run = RunDirectory.create(
        kind="grpo",
        model_tag="qwen3-0.6b",
        task="plan",
        method=method,
        verifier="exact",
        seed=args.seed,
        config=resolved,
    )
    started = time.perf_counter()
    try:
        seed_everything(args.seed)
        tokenizer = load_tokenizer(resolved["model_name_or_path"])
        model = load_model(resolved["model_name_or_path"], adapter=adapter, trainable=True)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=float(raw["learning_rate"]))
        tasks = read_tasks(args.input)
        all_rewards = []
        positive_groups = 0
        for step in range(args.steps):
            task = tasks[step % len(tasks)]
            candidates = []
            rewards = []
            model.eval()
            for candidate_id in range(group_size):
                prefix = None
                if method == "prefix_balanced":
                    prefix = OPERATIONS[(step * group_size + candidate_id) % len(OPERATIONS)]
                prompt = build_prompt(task) + "\nPROGRAM:" + (f" {prefix}" if prefix else "")
                encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    sequence = model.generate(
                        **encoded,
                        do_sample=True,
                        temperature=0.7,
                        top_p=1.0,
                        max_new_tokens=int(raw["max_completion_length"]),
                        pad_token_id=tokenizer.pad_token_id,
                    )[0]
                completion = tokenizer.decode(sequence[encoded.input_ids.shape[1] :], skip_special_tokens=True)
                text = "PROGRAM: " + ((prefix + " ") if prefix else "") + completion.strip()
                text = text.splitlines()[0].strip()
                result = verify(task, text)
                reward = float(result.is_correct)
                rewards.append(reward)
                candidates.append((prompt, text, result, sequence.detach().cpu(), int(encoded.input_ids.shape[1])))
                run.generations(
                    [
                        {
                            "run_id": run.run_id,
                            "task_id": task.task_id,
                            "candidate_id": step * group_size + candidate_id,
                            "text": text,
                            "parsed_program": list(result.parsed_program) if result.parsed_program is not None else None,
                            "parsed_result": None,
                            "parse_ok": result.parse_ok,
                            "is_correct": result.is_correct,
                            "accepted": result.is_correct,
                            "alpha": 1.0,
                            "beta": 0.0,
                            "prompt_tokens": int(encoded.input_ids.numel()),
                            "completion_tokens": int(sequence.numel() - encoded.input_ids.shape[1]),
                            "elapsed_seconds": None,
                            "sampling": {"temperature": 0.7, "forced_prefix": prefix},
                        }
                    ]
                )
            mean_reward = sum(rewards) / len(rewards)
            std = max((sum((value - mean_reward) ** 2 for value in rewards) / len(rewards)) ** 0.5, 1e-6)
            advantages = [(value - mean_reward) / std for value in rewards]
            positive_groups += int(any(rewards))
            all_rewards.extend(rewards)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            model.train()
            for advantage, (_, _, _, sequence_cpu, prompt_length) in zip(advantages, candidates):
                sequence = sequence_cpu.unsqueeze(0).to(model.device)
                attention = torch.ones_like(sequence)
                logp = completion_log_probability(model, sequence, attention, prompt_length)
                with torch.no_grad(), model.disable_adapter():
                    reference_logp = completion_log_probability(model, sequence, attention, prompt_length)
                kl_penalty = (logp - reference_logp).square()
                losses.append(-float(advantage) * logp + float(raw["kl_beta"]) * kl_penalty)
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
            step_metrics = {
                "loss": float(loss.detach().cpu()),
                "mean_reward": mean_reward,
                "positive_fraction": sum(rewards) / len(rewards),
                "reward_std": std if std > 1e-6 else 0.0,
            }
            run.metric("grpo_train", step + 1, step_metrics)
            run.log(f"step={step + 1} " + json.dumps(step_metrics))
            checkpoint = run.path / "checkpoints" / f"step-{step + 1}"
            model.save_pretrained(checkpoint)
        final_adapter = run.path / "checkpoints" / "final_adapter"
        model.save_pretrained(final_adapter)
        canonical = Path(f"artifacts/adapters/grpo_{method}_smoke_seed{args.seed}")
        if canonical.exists():
            shutil.rmtree(canonical)
        shutil.copytree(final_adapter, canonical)
        metrics = {
            "steps": args.steps,
            "mean_reward": sum(all_rewards) / len(all_rewards),
            "positive_groups": positive_groups,
            "groups": args.steps,
            "adapter": str(canonical),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        }
        run.finish(metrics)
        print(json.dumps({"run_id": run.run_id, **metrics}, ensure_ascii=False, indent=2))
    except BaseException as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
