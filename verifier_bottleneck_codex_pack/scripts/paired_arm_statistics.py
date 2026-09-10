"""Paired task-level comparison of two search arms across seeds.

Generalises `three_seed_statistics.py`, which hard-codes the method labels
`iid_action` and `prefix_balanced_action`. The screen now writes labels such as
`iid_action@1.6`, so arms are named explicitly:

    --arm-a 0 artifacts/runs/<run> iid_action@0.7
    --arm-b 0 artifacts/runs/<run> prefix_balanced_action@0.7

Statistics follow docs/07 section 7.7: paired per-task differences, 10,000-fold
bootstrap, aggregation inside a seed before aggregation across seeds, and an
exact McNemar test on discordant tasks.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def task_successes(run: Path, method: str) -> dict[str, bool]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for line in (run / "generations.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["sampling"]["method"] == method:
            grouped[record["task_id"]].append(bool(record["is_correct"]))
    if not grouped:
        raise ValueError(f"no generations for method {method!r} in {run}")
    return {task_id: any(values) for task_id, values in grouped.items()}


def mcnemar(a_values: np.ndarray, b_values: np.ndarray):
    b_only = int(np.sum((b_values == 1) & (a_values == 0)))
    a_only = int(np.sum((a_values == 1) & (b_values == 0)))
    discordant = b_only + a_only
    p_value = float(binomtest(b_only, discordant, 0.5).pvalue) if discordant else 1.0
    return b_only, a_only, p_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-a", nargs=3, action="append", metavar=("SEED", "RUN", "METHOD"), required=True)
    parser.add_argument("--arm-b", nargs=3, action="append", metavar=("SEED", "RUN", "METHOD"), required=True)
    parser.add_argument("--label-a", default="arm_a")
    parser.add_argument("--label-b", default="arm_b")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--rng-seed", type=int, default=20260801)
    args = parser.parse_args()

    by_seed_a = {int(seed): (Path(run), method) for seed, run, method in args.arm_a}
    by_seed_b = {int(seed): (Path(run), method) for seed, run, method in args.arm_b}
    if set(by_seed_a) != set(by_seed_b):
        raise ValueError(f"seed sets differ: {sorted(by_seed_a)} vs {sorted(by_seed_b)}")

    rng = np.random.default_rng(args.rng_seed)
    per_seed = []
    seed_differences = []
    pooled_a: list[int] = []
    pooled_b: list[int] = []

    for seed in sorted(by_seed_a):
        a_success = task_successes(*by_seed_a[seed])
        b_success = task_successes(*by_seed_b[seed])
        if set(a_success) != set(b_success):
            raise ValueError(f"seed {seed}: task IDs differ between arms")
        task_ids = sorted(a_success)
        a_values = np.array([int(a_success[t]) for t in task_ids])
        b_values = np.array([int(b_success[t]) for t in task_ids])
        differences = b_values - a_values
        bootstrap = rng.choice(differences, size=(args.bootstrap, len(differences)), replace=True).mean(axis=1)
        b_only, a_only, p_value = mcnemar(a_values, b_values)
        per_seed.append(
            {
                "seed": seed,
                "tasks": len(task_ids),
                f"{args.label_a}_pass": float(a_values.mean()),
                f"{args.label_b}_pass": float(b_values.mean()),
                "delta_percentage_points": 100.0 * float(differences.mean()),
                "paired_bootstrap_95_ci_percentage_points": [
                    100.0 * float(np.quantile(bootstrap, 0.025)),
                    100.0 * float(np.quantile(bootstrap, 0.975)),
                ],
                f"{args.label_b}_only": b_only,
                f"{args.label_a}_only": a_only,
                "mcnemar_exact_two_sided_p": p_value,
                "runs": {args.label_a: str(by_seed_a[seed][0]), args.label_b: str(by_seed_b[seed][0])},
                "methods": {args.label_a: by_seed_a[seed][1], args.label_b: by_seed_b[seed][1]},
            }
        )
        seed_differences.append(differences)
        pooled_a.extend(a_values.tolist())
        pooled_b.extend(b_values.tolist())

    combined = np.empty(args.bootstrap)
    for index in range(args.bootstrap):
        combined[index] = np.mean(
            [rng.choice(d, size=len(d), replace=True).mean() for d in seed_differences]
        )
    pooled_a_array = np.array(pooled_a)
    pooled_b_array = np.array(pooled_b)
    b_only, a_only, pooled_p = mcnemar(pooled_a_array, pooled_b_array)

    deltas = [item["delta_percentage_points"] for item in per_seed]
    ci = [100.0 * float(np.quantile(combined, 0.025)), 100.0 * float(np.quantile(combined, 0.975))]
    result = {
        "comparison": f"{args.label_b} minus {args.label_a}",
        "per_seed": per_seed,
        "aggregate": {
            "seed_count": len(per_seed),
            "paired_seed_tasks": len(pooled_a),
            f"mean_{args.label_a}_pass": float(np.mean([item[f"{args.label_a}_pass"] for item in per_seed])),
            f"mean_{args.label_b}_pass": float(np.mean([item[f"{args.label_b}_pass"] for item in per_seed])),
            "mean_delta_percentage_points": float(np.mean(deltas)),
            "hierarchical_paired_bootstrap_95_ci_percentage_points": ci,
            "bootstrap_repetitions": args.bootstrap,
            "positive_direction_seeds": sum(value > 0 for value in deltas),
            f"{args.label_b}_only": b_only,
            f"{args.label_a}_only": a_only,
            "mcnemar_exact_two_sided_p_pooled": pooled_p,
            "ci_excludes_zero": ci[0] > 0 or ci[1] < 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
