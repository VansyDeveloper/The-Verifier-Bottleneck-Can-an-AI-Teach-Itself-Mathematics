"""Action-space exploration screen.

Compares program-search strategies over the model's conditional operation
probabilities at a fixed candidate budget K. Methods are given as `name@T`:

    iid_action@0.7              independent sampling at temperature 0.7 (E1 control)
    iid_action@1.6              same sampler, higher temperature (E2 family)
    temp_mix_action@0.2,0.6,1.0,1.4   equal split of K over four temperatures (E3)
    prefix_balanced_action@0.7  deterministic first-action allocation (E4)

Every method sees the same tasks, the same K and the same maximum program
length. The exact verifier runs only on complete programs.

D-017: scoring goes through `vbexp.action_policy.score_operations`.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from vbexp.action_policy import OPERATIONS, sample_action, score_operations
from vbexp.experiment import RunDirectory, seed_everything, sha256_adapter, sha256_file
from vbexp.io import read_tasks
from vbexp.metrics import signature_entropy
from vbexp.modeling import load_model, load_tokenizer
from vbexp.search import all_solutions, shortest_solutions
from vbexp.verifier import verify

DEFAULT_MIX = (0.2, 0.6, 1.0, 1.4)


def parse_method(spec: str):
    """`name@T` or `name@T1,T2,...` -> (spec, name, temperatures)."""
    name, _, tail = spec.partition("@")
    if name not in ("iid_action", "prefix_balanced_action", "temp_mix_action"):
        raise argparse.ArgumentTypeError(f"unknown method: {name}")
    if not tail:
        temperatures = DEFAULT_MIX if name == "temp_mix_action" else (0.7,)
    else:
        temperatures = tuple(float(part) for part in tail.split(","))
    if name != "temp_mix_action" and len(temperatures) != 1:
        raise argparse.ArgumentTypeError(f"{name} takes exactly one temperature")
    if any(value <= 0 for value in temperatures):
        raise argparse.ArgumentTypeError("temperatures must be positive")
    return spec, name, temperatures


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", nargs="+", type=parse_method, required=True)
    parser.add_argument("--task-tag", default="plan-dev")
    parser.add_argument("--run-method-tag", default="action-search-screen")
    return parser.parse_args()


def candidate_plan(name: str, temperatures, candidate_id: int, k: int):
    """Forced first action and temperature for one candidate."""
    if name == "prefix_balanced_action":
        # Kept verbatim from the published implementation: candidates
        # 0..K-2 cycle through the five operations (SH1 therefore receives one
        # extra slot when K-1 is not a multiple of five) and the last candidate
        # is unconstrained. docs/05 section 5.6 describes 30 forced + 2 free for
        # K=32; the one-slot difference is recorded in DECISIONS.md D-018.
        forced = OPERATIONS[candidate_id % len(OPERATIONS)] if candidate_id < k - 1 else None
        return forced, temperatures[0]
    if name == "temp_mix_action":
        per_bucket = k / len(temperatures)
        return None, temperatures[min(int(candidate_id // per_bucket), len(temperatures) - 1)]
    return None, temperatures[0]


def sample_program(model, tokenizer, task, rng, temperature, first, cache):
    prefix = [first] if first else []
    passes = tokens = 0
    while len(prefix) < task.max_steps:
        key = tuple(prefix)
        if key not in cache:
            cache[key] = score_operations(model, tokenizer, task, prefix).tolist()
            passes += 1
            tokens += sum(
                len(tokenizer(" " + operation, add_special_tokens=False).input_ids)
                for operation in OPERATIONS
            )
        prefix.append(sample_action(cache[key], temperature, rng))
    return tuple(prefix), passes, tokens


def main():
    args = parse_args()
    specs = [spec for spec, _, _ in args.methods]
    resolved = {
        "model_name_or_path": "Qwen/Qwen3-0.6B-Base",
        "adapter": str(args.adapter),
        "adapter_sha256": sha256_adapter(args.adapter),
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "k": args.k,
        "seed": args.seed,
        "methods": specs,
        "program_length": "task.max_steps",
        "verifier": "exact_after_generation",
    }
    run = RunDirectory.create(
        kind="exploration",
        model_tag="qwen3-0.6b",
        task=args.task_tag,
        method=args.run_method_tag,
        verifier="exact",
        seed=args.seed,
        config=resolved,
    )
    started = time.perf_counter()
    try:
        seed_everything(args.seed)
        tokenizer = load_tokenizer(resolved["model_name_or_path"])
        model = load_model(resolved["model_name_or_path"], adapter=args.adapter)
        model.eval()
        tasks = read_tasks(args.input)
        summaries = {}
        for spec, name, temperatures in args.methods:
            rng = random.Random(args.seed)
            correct_count = total = model_passes = scored_tokens = 0
            per_task = []
            coverage = []
            shortest_recall = []
            correct_signatures = []
            method_started = time.perf_counter()
            with torch.inference_mode():
                for task_index, task in enumerate(tasks):
                    cache: dict[tuple[str, ...], list[float]] = {}
                    found = set()
                    attempted = set()
                    records = []
                    task_correct = 0
                    for candidate_id in range(args.k):
                        first, temperature = candidate_plan(name, temperatures, candidate_id, args.k)
                        program, passes, tokens = sample_program(
                            model, tokenizer, task, rng, temperature, first, cache
                        )
                        model_passes += passes
                        scored_tokens += tokens
                        attempted.add(program)
                        text = "PROGRAM: " + " ".join(program)
                        result = verify(task, text)
                        total += 1
                        correct_count += int(result.is_correct)
                        task_correct += int(result.is_correct)
                        if result.is_correct:
                            found.add(program)
                            correct_signatures.append(program)
                        records.append(
                            {
                                "run_id": run.run_id,
                                "task_id": task.task_id,
                                "candidate_id": task_index * args.k + candidate_id,
                                "text": text,
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
                                "sampling": {
                                    "method": spec,
                                    "base_method": name,
                                    "temperature": temperature,
                                    "forced_first": first,
                                },
                            }
                        )
                    run.generations(records)
                    valid = set(
                        all_solutions(task.start, task.target, task.p, task.operations, task.max_steps)
                    )
                    shortest = set(
                        shortest_solutions(task.start, task.target, task.p, task.operations, task.max_steps)
                    )
                    coverage.append(len(found & valid) / len(valid))
                    shortest_recall.append(len(found & shortest) / len(shortest))
                    per_task.append(
                        {
                            "task_id": task.task_id,
                            "success": bool(found),
                            "distinct_programs": len(attempted),
                            "correct_candidates": task_correct,
                            "coverage": coverage[-1],
                        }
                    )
            successes = [item["success"] for item in per_task]
            summary = {
                f"pass_at_{args.k}": sum(successes) / len(successes),
                "candidate_correct_rate": correct_count / total,
                "correct_candidates": len(correct_signatures),
                "unique_correct_programs": len(set(correct_signatures)),
                "distinct_programs_per_task": sum(item["distinct_programs"] for item in per_task)
                / len(per_task),
                "all_solution_coverage": sum(coverage) / len(coverage),
                "shortest_solution_recall": sum(shortest_recall) / len(shortest_recall),
                "signature_entropy": signature_entropy(correct_signatures),
                "model_passes": model_passes,
                "scored_operation_tokens": scored_tokens,
                "elapsed_seconds": time.perf_counter() - method_started,
                "temperatures": list(temperatures),
            }
            summaries[spec] = summary
            run.metric(f"exploration/{spec}", 0, summary)
            run.log(f"{spec} " + json.dumps(summary))
            print(f"[{spec}] " + json.dumps({k: v for k, v in summary.items() if k != "temperatures"}))
            (run.path / f"per_task_{spec.replace('@', '_').replace(',', '-')}.json").write_text(
                json.dumps(per_task, indent=1), encoding="utf-8"
            )
        final = {
            "methods": summaries,
            "tasks": len(tasks),
            "candidates_per_task": args.k,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        }
        run.finish(final)
        print(json.dumps({"run_id": run.run_id, **final}, ensure_ascii=False, indent=2))
    except BaseException as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
