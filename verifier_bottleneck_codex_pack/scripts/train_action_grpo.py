from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch

from vbexp.action_policy import OPERATIONS, sample_action, score_operations
from vbexp.experiment import RunDirectory, seed_everything
from vbexp.io import read_tasks
from vbexp.modeling import load_policy_with_reference, load_tokenizer
from vbexp.verifier import verify


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["iid_action", "prefix_balanced_action"], required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--kl-beta", type=float, default=0.01)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--output-name")
    return parser.parse_args()


def sample_group(model, tokenizer, task, method, group_size, temperature, rng):
    """Sample `group_size` complete programs from the current policy.

    The per-prefix score cache is exact: the policy is frozen and in eval mode
    for the whole call, so two visits to the same prefix must return the same
    scores. It does not change how many times `rng` is drawn from, so seeds keep
    their original meaning.
    """
    programs = []
    cache: dict[tuple[str, ...], list[float]] = {}
    model.eval()
    with torch.inference_mode():
        for candidate_id in range(group_size):
            forced = None
            if method == "prefix_balanced_action" and candidate_id < len(OPERATIONS):
                forced = OPERATIONS[candidate_id]
            prefix = [forced] if forced else []
            while len(prefix) < task.max_steps:
                key = tuple(prefix)
                if key not in cache:
                    cache[key] = score_operations(model, tokenizer, task, prefix).tolist()
                prefix.append(sample_action(cache[key], temperature, rng))
            programs.append((tuple(prefix), forced))
    return programs


def reference_log_probs(model, tokenizer, task, prefixes, temperature):
    """Frozen-anchor action log-probabilities for every visited prefix.

    Computed once per step under the `reference` adapter in eval mode. The
    reference never changes within a step, so caching is exact.
    """
    table: dict[tuple[str, ...], torch.Tensor] = {}
    model.set_adapter("reference")
    model.eval()
    with torch.no_grad():
        for prefix in prefixes:
            if prefix in table:
                continue
            scores = score_operations(model, tokenizer, task, list(prefix)) / temperature
            table[prefix] = torch.log_softmax(scores, dim=0)
    model.set_adapter("policy")
    return table


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
        "save_every": args.save_every,
        "reward": "binary_exact",
        "memory_profile": "D-017 single-base dual-adapter, logits_to_keep scoring",
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
        model = load_policy_with_reference(resolved["model_name_or_path"], args.adapter)
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

            visited = []
            for program, forced in programs:
                for index in range(len(program)):
                    if index == 0 and forced is not None:
                        continue
                    visited.append(tuple(program[:index]))
            anchor = reference_log_probs(model, tokenizer, task, visited, args.temperature)

            optimizer.zero_grad(set_to_none=True)
            loss_values = []
            kl_values = []
            model.train()
            for (program, forced), advantage in zip(programs, advantages):
                # The candidate loss -advantage * sum_t log p_t + beta * sum_t kl_t
                # is a sum of independent per-position terms, so backward runs
                # position by position. The accumulated gradient is identical to
                # one backward over the whole candidate; only one position's
                # autograd graph is alive at a time, which is what keeps GRPO
                # inside 4 GB (D-017).
                candidate_loss = 0.0
                kl_total = 0.0
                prefix: list[str] = []
                for index, action in enumerate(program):
                    if index == 0 and forced is not None:
                        prefix.append(action)
                        continue
                    current_scores = score_operations(model, tokenizer, task, prefix) / args.temperature
                    current_log = torch.log_softmax(current_scores, dim=0)
                    reference_log = anchor[tuple(prefix)]
                    current_prob = current_log.exp()
                    action_index = OPERATIONS.index(action)
                    kl = (current_prob * (current_log - reference_log)).sum()
                    term = -float(advantage) * current_log[action_index] + args.kl_beta * kl
                    (term / len(programs)).backward()
                    candidate_loss += float(term.detach())
                    kl_total += float(kl.detach())
                    prefix.append(action)
                loss_values.append(candidate_loss)
                kl_values.append(kl_total)
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
            if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
                model.save_pretrained(
                    run.path / "checkpoints" / f"step-{step + 1}", selected_adapters=["policy"]
                )
        final_adapter = run.path / "checkpoints" / "final_adapter"
        model.save_pretrained(final_adapter, selected_adapters=["policy"])
        name = args.output_name or f"grpo_{args.method}_pilot_seed{args.seed}"
        canonical = Path("artifacts/adapters") / name
        if canonical.exists():
            shutil.rmtree(canonical)
        shutil.copytree(final_adapter / "policy", canonical)
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
