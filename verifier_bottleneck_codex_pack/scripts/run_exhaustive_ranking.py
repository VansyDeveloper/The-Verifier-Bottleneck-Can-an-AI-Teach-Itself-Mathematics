"""Exhaustive program ranking — Hit@K without any sampling.

Every PLAN task of depth d has exactly 5^d candidate programs (25 / 125 / 625 for
d = 2 / 3 / 4). This scores all of them and reports whether a correct program
appears in the top K of the ranking.

Why this metric replaces sampled pass@K as the primary evidence
---------------------------------------------------------------
This project's central finding was that sampled `pass@K` is confounded by
duplicate collapse: at low temperature the policy spends 32 draws on ~3 distinct
programs, so `pass@32` measures sampling diversity as much as it measures the
policy's preferences. Exhaustive ranking removes the confound by construction —
every candidate is evaluated exactly once, there is nothing to deduplicate, and
the result is deterministic given the adapter. No seed, no temperature, no
sampler. Adopted from the independent Stage-4 protocol, which forbids random
generation and analytic pass@K for the same reason.

Scoring
-------
log p(program) = sum_t log_softmax(action_scores at prefix_t)[action_t]

That is a proper distribution over the 5^d programs, so `correct_mass` is
meaningful. Temperature is fixed at 1.0: it is the model's actual distribution,
and any other value would reorder programs, since the per-prefix normaliser
differs along different paths.

Cost is the prefix tree, not the program count: depth 3 needs 1 + 5 + 25 = 31
scoring passes per task to obtain all 125 program probabilities exactly.

Why prefixes are scored one at a time
-------------------------------------
Batching several prefixes into one forward pass was implemented and measured, and
it is *not* numerically neutral: in bf16 the GEMM kernel depends on batch size, and
grouping equal-width prefixes moved individual action log-probabilities by up to
0.2 nats — consistent with logit magnitudes near 30 and bf16's ~0.4% relative
precision. Program scores would then depend on how many prefixes happened to share
a padded width, which can flip ordering near the top-K boundary. For a metric whose
whole purpose is exactness, that trade is wrong, so every prefix is scored in its
own fixed 5-row call. The cost is roughly 3.5 s per depth-3 task.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from itertools import product
from pathlib import Path

import torch

from vbexp.action_policy import OPERATIONS, score_operations
from vbexp.experiment import RunDirectory, seed_everything, sha256_adapter, sha256_file
from vbexp.io import read_tasks
from vbexp.modeling import load_model, load_tokenizer
from vbexp.search import all_solutions
from vbexp.verifier import verify

DEFAULT_K = (1, 4, 8, 16, 32, 64)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K))
    parser.add_argument("--task-tag", default="ranking")
    parser.add_argument("--run-method-tag", default="exhaustive-rank")
    parser.add_argument("--limit", type=int, help="evaluate only the first N tasks")
    return parser.parse_args()


def program_log_probabilities(model, tokenizer, task) -> dict[tuple[str, ...], float]:
    """Exact log-probability of every 5^depth program, via the prefix tree."""
    depth = task.max_steps
    conditional: dict[tuple[str, ...], list[float]] = {}
    for prefix_length in range(depth):
        for prefix in product(OPERATIONS, repeat=prefix_length):
            scores = score_operations(model, tokenizer, task, list(prefix))
            conditional[prefix] = torch.log_softmax(scores, dim=0).tolist()
    return {
        program: sum(conditional[program[:i]][OPERATIONS.index(program[i])] for i in range(depth))
        for program in product(OPERATIONS, repeat=depth)
    }


def rank_programs(log_probabilities: dict[tuple[str, ...], float]):
    """Descending by log-probability, ties broken lexicographically."""
    return sorted(log_probabilities.items(), key=lambda item: (-item[1], item[0]))


def main():
    args = parse_args()
    resolved = {
        "model_name_or_path": "Qwen/Qwen3-0.6B-Base",
        "adapter": str(args.adapter),
        "adapter_sha256": sha256_adapter(args.adapter),
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "k_values": args.k_values,
        "metric": "exhaustive_hit_at_k",
        "temperature": 1.0,
        "sampling": "none_exhaustive_enumeration",
        "tie_break": "lexicographic_operation_signature",
    }
    run = RunDirectory.create(
        kind="ranking", model_tag="qwen3-0.6b", task=args.task_tag,
        method=args.run_method_tag, verifier="exact", seed=0, config=resolved,
    )
    started = time.perf_counter()
    try:
        seed_everything(0)  # determinism of library state only; no sampling occurs
        tokenizer = load_tokenizer(resolved["model_name_or_path"])
        model = load_model(resolved["model_name_or_path"], adapter=args.adapter)
        model.eval()
        tasks = read_tasks(args.input)
        if args.limit:
            tasks = tasks[: args.limit]

        hits = {k: [] for k in args.k_values}
        best_ranks, correct_masses, reciprocal_ranks, solution_counts = [], [], [], []
        model_passes = 0
        with torch.inference_mode():
            for task in tasks:
                log_probabilities = program_log_probabilities(model, tokenizer, task)
                model_passes += sum(len(OPERATIONS) ** i for i in range(task.max_steps))
                correct = {
                    program for program in log_probabilities
                    if verify(task, "PROGRAM: " + " ".join(program)).is_correct
                }
                # Cross-check the verifier against the independent enumerator.
                enumerated = {
                    p for p in all_solutions(task.start, task.target, task.p,
                                             task.operations, task.max_steps)
                    if len(p) == task.max_steps
                }
                if correct != enumerated:
                    raise RuntimeError(
                        f"verifier/enumerator disagree on {task.task_id}: "
                        f"{sorted(correct ^ enumerated)}"
                    )
                ranked = rank_programs(log_probabilities)
                positions = [i for i, (program, _) in enumerate(ranked, start=1) if program in correct]
                best = positions[0] if positions else None
                for k in args.k_values:
                    hits[k].append(bool(best is not None and best <= k))
                best_ranks.append(best if best is not None else len(ranked) + 1)
                reciprocal_ranks.append(1.0 / best if best else 0.0)
                total = sum(math.exp(v) for v in log_probabilities.values())
                correct_masses.append(
                    sum(math.exp(log_probabilities[p]) for p in correct) / total if total else 0.0
                )
                solution_counts.append(len(correct))
                run.generations([{
                    "run_id": run.run_id, "task_id": task.task_id,
                    "candidates_ranked": len(ranked), "correct_programs": len(correct),
                    "best_correct_rank": best, "correct_mass": correct_masses[-1],
                    "top1_program": list(ranked[0][0]),
                    "top1_is_correct": ranked[0][0] in correct,
                }])

        metrics = {
            **{f"hit_at_{k}": sum(v) / len(v) for k, v in hits.items()},
            "mean_best_rank": sum(best_ranks) / len(best_ranks),
            "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
            "mean_correct_mass": sum(correct_masses) / len(correct_masses),
            "mean_solutions_per_task": sum(solution_counts) / len(solution_counts),
            "candidates_per_task": len(OPERATIONS) ** tasks[0].max_steps,
            "tasks": len(tasks),
            "model_passes": model_passes,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        }
        run.metric("exhaustive_ranking", 0, metrics)
        run.finish(metrics)
        print(json.dumps({"run_id": run.run_id, **metrics}, indent=2))
    except BaseException as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
