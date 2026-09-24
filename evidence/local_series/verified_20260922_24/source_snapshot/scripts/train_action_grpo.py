from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch

from vbexp.action_protocol import forced_training_first
from vbexp.experiment import RunDirectory, seed_everything
from vbexp.io import read_tasks
from vbexp.modeling import load_model, load_tokenizer
from vbexp.prompts import build_prompt
from vbexp.verifier import verify

OPERATIONS = ("SH1", "SC2", "REV", "AC1", "AX1")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Group-normalized policy-gradient training (not clipped GRPO)."
    )
    parser.add_argument("--method", choices=["iid_action", "prefix_balanced_action"], required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--kl-beta", type=float, default=0.01)
    parser.add_argument("--output-adapter", type=Path)
    return parser.parse_args()


def operation_scores(model, tokenizer, task, prefix):
    base = build_prompt(task) + "\nPROGRAM:" + ((" " + " ".join(prefix)) if prefix else "")
    base_ids = tokenizer(base, add_special_tokens=False).input_ids
    rows = []
    lengths = []
    for operation in OPERATIONS:
        added = tokenizer(" " + operation, add_special_tokens=False).input_ids
        rows.append(base_ids + added)
        lengths.append(len(added))
    max_len = max(map(len, rows))
    input_rows = []
    mask_rows = []
    for row in rows:
        pad = max_len - len(row)
        input_rows.append([tokenizer.pad_token_id] * pad + row)
        mask_rows.append([0] * pad + [1] * len(row))
    input_ids = torch.tensor(input_rows, device=model.device)
    attention = torch.tensor(mask_rows, device=model.device)
    logits = model(input_ids=input_ids, attention_mask=attention).logits.float()
    scores = []
    for row_index, added_length in enumerate(lengths):
        row_length = len(rows[row_index])
        pad = max_len - row_length
        first = pad + len(base_ids)
        score = torch.zeros((), device=model.device)
        for position in range(first, pad + row_length):
            target = input_ids[row_index, position]
            score = score + torch.log_softmax(logits[row_index, position - 1], dim=-1)[target]
        scores.append(score)
    return torch.stack(scores)


def sample_group(model, tokenizer, task, method, group_size, temperature, rng):
    programs = []
    model.eval()
    with torch.inference_mode():
        for candidate_id in range(group_size):
            forced = forced_training_first(method, candidate_id)
            prefix = [forced] if forced else []
            while len(prefix) < task.max_steps:
                scores = operation_scores(model, tokenizer, task, prefix)
                probabilities = torch.softmax(scores / temperature, dim=0).cpu().tolist()
                prefix.append(rng.choices(OPERATIONS, weights=probabilities, k=1)[0])
            programs.append((tuple(prefix), forced))
    return programs


def main():
    args = parse_args()
    resolved = {
        "model_name_or_path": "Qwen/Qwen3-0.6B-Base",
        "adapter": str(args.adapter),
        "input": str(args.input),
        "method": args.method,
        "steps": args.steps,
        "group_size": args.group_size,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "temperature": args.temperature,
        "kl_beta": args.kl_beta,
        "reward": "binary_exact",
        "algorithm": "group_normalized_policy_gradient",
        "prefix_group_allocation": "5 forced first actions + 3 iid candidates when G=8",
    }
    run = RunDirectory.create(
        kind="grpo",
        model_tag="qwen3-0.6b",
        task="plan",
        method=args.method,
        verifier="exact",
        seed=args.seed,
        config=resolved,
    )
    started = time.perf_counter()
    try:
        seed_everything(args.seed)
        rng = random.Random(args.seed)
        tokenizer = load_tokenizer(resolved["model_name_or_path"])
        model = load_model(resolved["model_name_or_path"], adapter=args.adapter, trainable=True)
        reference = load_model(resolved["model_name_or_path"], adapter=args.adapter, trainable=False)
        reference.eval()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.learning_rate,
        )
        tasks = read_tasks(args.input)
        rewards_all = []
        positive_groups = 0
        for step in range(args.steps):
            task = tasks[step % len(tasks)]
            programs = sample_group(
                model, tokenizer, task, args.method, args.group_size, args.temperature, rng
            )
            rewards = []
            records = []
            for candidate_id, (program, forced) in enumerate(programs):
                result = verify(task, "PROGRAM: " + " ".join(program))
                reward = float(result.is_correct)
                rewards.append(reward)
                records.append(
                    {
                        "run_id": run.run_id,
                        "task_id": task.task_id,
                        "candidate_id": step * args.group_size + candidate_id,
                        "text": "PROGRAM: " + " ".join(program),
                        "parsed_program": list(program),
                        "parsed_result": None,
                        "parse_ok": True,
                        "is_correct": result.is_correct,
                        "accepted": result.is_correct,
                        "alpha": 1.0,
                        "beta": 0.0,
                        "prompt_tokens": None,
                        "completion_tokens": len(program),
                        "elapsed_seconds": None,
                        "sampling": {"method": args.method, "forced_first": forced},
                    }
                )
            run.generations(records)
            mean_reward = sum(rewards) / len(rewards)
            variance = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
            scale = max(math.sqrt(variance), 1e-6)
            advantages = [(reward - mean_reward) / scale for reward in rewards]
            positive_groups += int(any(rewards))
            rewards_all.extend(rewards)
            optimizer.zero_grad(set_to_none=True)
            loss_values = []
            kl_values = []
            model.train()
            for (program, forced), advantage in zip(programs, advantages):
                log_probability = torch.zeros((), device=model.device)
                kl = torch.zeros((), device=model.device)
                prefix = []
                for index, action in enumerate(program):
                    if index == 0 and forced is not None:
                        prefix.append(action)
                        continue
                    current_scores = operation_scores(model, tokenizer, task, prefix) / args.temperature
                    with torch.no_grad():
                        reference_scores = operation_scores(reference, tokenizer, task, prefix) / args.temperature
                    current_log = torch.log_softmax(current_scores, dim=0)
                    reference_log = torch.log_softmax(reference_scores, dim=0)
                    current_prob = current_log.exp()
                    action_index = OPERATIONS.index(action)
                    log_probability = log_probability + current_log[action_index]
                    kl = kl + (current_prob * (current_log - reference_log)).sum()
                    prefix.append(action)
                candidate_loss = -float(advantage) * log_probability + args.kl_beta * kl
                (candidate_loss / len(programs)).backward()
                loss_values.append(float(candidate_loss.detach().cpu()))
                kl_values.append(float(kl.detach().cpu()))
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
            )
            optimizer.step()
            metrics = {
                "loss": sum(loss_values) / len(loss_values),
                "mean_reward": mean_reward,
                "positive_fraction": sum(rewards) / len(rewards),
                "mean_kl": sum(kl_values) / len(kl_values),
            }
            run.metric("grpo_train", step + 1, metrics)
            run.log(f"step={step + 1} " + json.dumps(metrics))
            if (step + 1) % 50 == 0 or step + 1 == args.steps:
                model.save_pretrained(run.path / "checkpoints" / f"step-{step + 1}")
        final_adapter = run.path / "checkpoints" / "final_adapter"
        model.save_pretrained(final_adapter)
        canonical = args.output_adapter or Path(
            f"artifacts/adapters/grpo_{args.method}_pilot_seed{args.seed}"
        )
        if canonical.exists():
            shutil.rmtree(canonical)
        shutil.copytree(final_adapter, canonical)
        final = {
            "steps": args.steps,
            "mean_reward": sum(rewards_all) / len(rewards_all),
            "positive_groups": positive_groups,
            "groups": args.steps,
            "adapter": str(canonical),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        }
        run.finish(final)
        print(json.dumps({"run_id": run.run_id, **final}, indent=2))
    except BaseException as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
