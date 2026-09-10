from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def task_successes(run: Path, method: str) -> dict[str, bool]:
    grouped = defaultdict(list)
    for line in (run / "generations.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["sampling"]["method"] == method:
            grouped[record["task_id"]].append(bool(record["is_correct"]))
    return {task_id: any(values) for task_id, values in grouped.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", nargs=3, action="append", metavar=("SEED", "IID_RUN", "PREFIX_RUN"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--rng-seed", type=int, default=20260727)
    args = parser.parse_args()

    rng = np.random.default_rng(args.rng_seed)
    per_seed = []
    all_iid = []
    all_prefix = []
    seed_differences = []
    for seed_text, iid_path, prefix_path in args.pair:
        seed = int(seed_text)
        iid = task_successes(Path(iid_path), "iid_action")
        prefix = task_successes(Path(prefix_path), "prefix_balanced_action")
        if set(iid) != set(prefix):
            raise ValueError(f"seed {seed}: task IDs differ")
        task_ids = sorted(iid)
        iid_values = np.array([int(iid[t]) for t in task_ids])
        prefix_values = np.array([int(prefix[t]) for t in task_ids])
        differences = prefix_values - iid_values
        bootstrap = rng.choice(differences, size=(args.bootstrap, len(differences)), replace=True).mean(axis=1)
        prefix_only = int(np.sum((prefix_values == 1) & (iid_values == 0)))
        iid_only = int(np.sum((iid_values == 1) & (prefix_values == 0)))
        discordant = prefix_only + iid_only
        p_value = float(binomtest(prefix_only, discordant, 0.5).pvalue) if discordant else 1.0
        item = {
            "seed": seed,
            "tasks": len(task_ids),
            "iid_pass_at_32": float(iid_values.mean()),
            "prefix_pass_at_32": float(prefix_values.mean()),
            "delta_percentage_points": 100.0 * float(differences.mean()),
            "paired_bootstrap_95_ci_percentage_points": [
                100.0 * float(np.quantile(bootstrap, 0.025)),
                100.0 * float(np.quantile(bootstrap, 0.975)),
            ],
            "prefix_only": prefix_only,
            "iid_only": iid_only,
            "mcnemar_exact_two_sided_p": p_value,
        }
        per_seed.append(item)
        all_iid.extend(iid_values.tolist())
        all_prefix.extend(prefix_values.tolist())
        seed_differences.append(differences)

    combined_bootstrap = np.empty(args.bootstrap)
    for index in range(args.bootstrap):
        seed_means = []
        for differences in seed_differences:
            sampled = rng.choice(differences, size=len(differences), replace=True)
            seed_means.append(sampled.mean())
        combined_bootstrap[index] = np.mean(seed_means)
    all_iid_array = np.array(all_iid)
    all_prefix_array = np.array(all_prefix)
    prefix_only = int(np.sum((all_prefix_array == 1) & (all_iid_array == 0)))
    iid_only = int(np.sum((all_iid_array == 1) & (all_prefix_array == 0)))
    discordant = prefix_only + iid_only
    result = {
        "per_seed": per_seed,
        "aggregate": {
            "seed_count": len(per_seed),
            "paired_seed_tasks": len(all_iid),
            "mean_iid_pass_at_32": float(np.mean([item["iid_pass_at_32"] for item in per_seed])),
            "mean_prefix_pass_at_32": float(np.mean([item["prefix_pass_at_32"] for item in per_seed])),
            "mean_delta_percentage_points": float(np.mean([item["delta_percentage_points"] for item in per_seed])),
            "hierarchical_paired_bootstrap_95_ci_percentage_points": [
                100.0 * float(np.quantile(combined_bootstrap, 0.025)),
                100.0 * float(np.quantile(combined_bootstrap, 0.975)),
            ],
            "bootstrap_repetitions": args.bootstrap,
            "positive_direction_seeds": sum(item["delta_percentage_points"] > 0 for item in per_seed),
            "prefix_only": prefix_only,
            "iid_only": iid_only,
            "mcnemar_exact_two_sided_p_pooled": (
                float(binomtest(prefix_only, discordant, 0.5).pvalue) if discordant else 1.0
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
