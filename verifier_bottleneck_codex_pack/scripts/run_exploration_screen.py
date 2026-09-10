from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import torch

from vbexp.experiment import RunDirectory, load_yaml, seed_everything
from vbexp.io import read_tasks
from vbexp.metrics import signature_entropy
from vbexp.modeling import load_model, load_tokenizer
from vbexp.prompts import build_prompt
from vbexp.search import all_solutions, shortest_solutions
from vbexp.verifier import verify

OPERATIONS = ("SH1", "SC2", "REV", "AC1", "AX1")


def clean(text: str) -> str:
    for line in text.splitlines():
        if line.strip().upper().startswith("PROGRAM:"):
            return line.strip()
    return text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=Path("artifacts/data/smoke/dev_exploration.jsonl"))
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def method_specs(k: int):
    mix = [0.2, 0.6, 1.0, 1.4]
    return [
        ("low_t", [(0.2, None)] * k),
        ("iid_t07", [(0.7, None)] * k),
        ("high_t", [(1.2, None)] * k),
        ("temp_mix", [(mix[index % 4], None) for index in range(k)]),
        ("prefix_balanced", [(0.7, OPERATIONS[index % len(OPERATIONS)]) for index in range(k)]),
    ]


def main() -> None:
    args = parse_args()
    source = load_yaml(args.config)
    resolved = {
        **source,
        "tier": "smoke",
        "model_name_or_path": "Qwen/Qwen3-0.6B-Base",
        "adapter": str(args.adapter),
        "input": str(args.input),
        "candidates_per_task": args.k,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    run = RunDirectory.create(
        kind="exploration",
        model_tag="qwen3-0.6b",
        task="plan-dev",
        method="e0-e4",
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
        for method, specs in method_specs(args.k):
            parsed = correct = candidates = completion_tokens = 0
            successes = []
            unique_correct_total = 0
            coverage_values = []
            shortest_values = []
            signatures = []
            method_started = time.perf_counter()
            for task_index, task in enumerate(tasks):
                task_correct = []
                found_correct = set()
                records = []
                for candidate_id, (temperature, prefix) in enumerate(specs):
                    prompt = build_prompt(task) + "\nPROGRAM:"
                    if prefix:
                        prompt += " " + prefix
                    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
                    before = time.perf_counter()
                    with torch.inference_mode():
                        output = model.generate(
                            **encoded,
                            do_sample=True,
                            temperature=temperature,
                            top_p=0.95 if temperature >= 1.2 else 1.0,
                            max_new_tokens=args.max_new_tokens,
                            pad_token_id=tokenizer.pad_token_id,
                        )[0]
                    elapsed = time.perf_counter() - before
                    completion = tokenizer.decode(output[encoded.input_ids.shape[1] :], skip_special_tokens=True)
                    text = "PROGRAM: " + ((prefix + " ") if prefix else "") + completion.strip()
                    text = clean(text)
                    result = verify(task, text)
                    candidates += 1
                    parsed += int(result.parse_ok)
                    correct += int(result.is_correct)
                    task_correct.append(result.is_correct)
                    tokens = int(output.numel() - encoded.input_ids.shape[1])
                    completion_tokens += tokens
                    if result.parsed_program is not None:
                        signatures.append(result.parsed_program)
                        if result.is_correct:
                            found_correct.add(result.parsed_program)
                    records.append(
                        {
                            "run_id": run.run_id,
                            "task_id": task.task_id,
                            "candidate_id": task_index * args.k + candidate_id,
                            "method": method,
                            "text": text,
                            "parse_ok": result.parse_ok,
                            "is_correct": result.is_correct,
                            "parsed_program": list(result.parsed_program) if result.parsed_program is not None else None,
                            "parsed_result": None,
                            "accepted": result.is_correct,
                            "alpha": 1.0,
                            "beta": 0.0,
                            "prompt_tokens": int(encoded.input_ids.numel()),
                            "completion_tokens": tokens,
                            "elapsed_seconds": elapsed,
                            "sampling": {"temperature": temperature, "forced_prefix": prefix},
                        }
                    )
                run.generations(records)
                valid = set(all_solutions(task.start, task.target, task.p, task.operations, task.max_steps))
                shortest = set(shortest_solutions(task.start, task.target, task.p, task.operations, task.max_steps))
                successes.append(any(task_correct))
                unique_correct_total += len(found_correct)
                coverage_values.append(len(found_correct & valid) / len(valid))
                shortest_values.append(len(found_correct & shortest) / len(shortest))
            summary = {
                f"pass_at_{args.k}": sum(successes) / len(successes),
                "candidate_correct_rate": correct / candidates,
                "parse_rate": parsed / candidates,
                "unique_correct_programs": unique_correct_total,
                "all_solution_coverage": sum(coverage_values) / len(coverage_values),
                "shortest_solution_recall": sum(shortest_values) / len(shortest_values),
                "signature_entropy": signature_entropy(signatures),
                "actual_completion_tokens": completion_tokens,
                "elapsed_seconds": time.perf_counter() - method_started,
            }
            summaries[method] = summary
            run.metric(f"dev_exploration/{method}", 0, summary)
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
