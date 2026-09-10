from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def successes(path: Path, method: str) -> dict[str, bool]:
    grouped = defaultdict(list)
    for line in (path / "generations.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["sampling"]["method"] == method:
            grouped[record["task_id"]].append(bool(record["is_correct"]))
    return {task_id: any(values) for task_id, values in grouped.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iid-run", type=Path, required=True)
    parser.add_argument("--prefix-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()

    iid = successes(args.iid_run, "iid_action")
    prefix = successes(args.prefix_run, "prefix_balanced_action")
    task_ids = sorted(set(iid) & set(prefix))
    if set(iid) != set(prefix):
        raise ValueError("paired runs do not contain identical task IDs")
    differences = np.array([int(prefix[t]) - int(iid[t]) for t in task_ids], dtype=float)
    rng = np.random.default_rng(args.seed)
    samples = rng.choice(differences, size=(args.bootstrap, len(differences)), replace=True).mean(axis=1)
    prefix_only = sum(prefix[t] and not iid[t] for t in task_ids)
    iid_only = sum(iid[t] and not prefix[t] for t in task_ids)
    discordant = prefix_only + iid_only
    mcnemar_p = (
        float(binomtest(prefix_only, discordant, p=0.5, alternative="two-sided").pvalue)
        if discordant
        else 1.0
    )
    result = {
        "tasks": len(task_ids),
        "iid_pass_at_k": sum(iid.values()) / len(task_ids),
        "prefix_pass_at_k": sum(prefix.values()) / len(task_ids),
        "delta_percentage_points": 100.0 * differences.mean(),
        "paired_bootstrap_95_ci_percentage_points": [
            100.0 * float(np.quantile(samples, 0.025)),
            100.0 * float(np.quantile(samples, 0.975)),
        ],
        "bootstrap_repetitions": args.bootstrap,
        "prefix_only_successes": prefix_only,
        "iid_only_successes": iid_only,
        "both_success": sum(prefix[t] and iid[t] for t in task_ids),
        "both_fail": sum(not prefix[t] and not iid[t] for t in task_ids),
        "mcnemar_exact_two_sided_p": mcnemar_p,
        "iid_run": str(args.iid_run),
        "prefix_run": str(args.prefix_run),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
