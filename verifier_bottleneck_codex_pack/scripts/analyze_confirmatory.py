"""D-018 confirmatory analysis.

Resolves the evaluation grid from `artifacts/runs`, then computes exactly the
four preregistered contrasts from
`artifacts/preregistration_confirmatory_2026-08-01.md`:

  Q1 primary   A4 - A2 on the shared pre-GRPO adapter  (structure vs matched diversity)
  Q2 secondary GRPO branch minus pre-GRPO, per branch  (does training contribute)
  Q3 secondary A4 on prefix-GRPO minus A1 on iid-GRPO  (the published diagonal)
  Q4 secondary A3 - A4 on the pre-GRPO adapter         (is plain hot iid better)

Holm correction is applied across the secondary family; Q1 is uncorrected.
Nothing here selects an arm: the arms and contrasts are fixed by the frozen
preregistration.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import binomtest

A1, A2, A3, A4, A5 = (
    "iid_action@0.7",
    "iid_action@2.0",
    "iid_action@3.0",
    "prefix_balanced_action@0.7",
    "temp_mix_action@0.2,0.6,1.0,1.4",
)

ROLE_BY_ADAPTER = {
    "sft_atomic_r32_pilot_aw4_cont": "pre",
    "grpo_iid_action_400_seed0": "iidgrpo",
    "grpo_iid_action_400_seed1": "iidgrpo",
    "grpo_iid_action_400_seed2": "iidgrpo",
    "grpo_prefix_balanced_action_400_seed0": "prefixgrpo",
    "grpo_prefix_balanced_action_400_seed1": "prefixgrpo",
    "grpo_prefix_balanced_action_400_seed2": "prefixgrpo",
}


def discover(runs_root: Path):
    """(dataset, role, seed) -> run directory, keeping the newest DONE run."""
    found: dict[tuple[str, str, int], Path] = {}
    for run in sorted(runs_root.iterdir()):
        config_path = run / "config.resolved.yaml"
        if not (run / "DONE").exists() or not config_path.exists():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or "methods" not in config or "adapter" not in config:
            continue
        adapter = Path(str(config["adapter"]).replace("\\", "/")).name
        role = ROLE_BY_ADAPTER.get(adapter)
        if role is None:
            continue
        dataset = Path(str(config["input"]).replace("\\", "/")).stem
        found[(dataset, role, int(config["seed"]))] = run
    return found


def task_successes(run: Path, method: str) -> dict[str, bool]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for line in (run / "generations.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["sampling"]["method"] == method:
            grouped[record["task_id"]].append(bool(record["is_correct"]))
    if not grouped:
        raise KeyError(f"method {method!r} absent from {run.name}")
    return {task_id: any(values) for task_id, values in grouped.items()}


def contrast(pairs, rng, bootstrap: int):
    """pairs: list of (seed, run_a, method_a, run_b, method_b). Reports b - a."""
    per_seed, differences_by_seed = [], []
    pooled_a: list[int] = []
    pooled_b: list[int] = []
    for seed, run_a, method_a, run_b, method_b in pairs:
        a_success = task_successes(run_a, method_a)
        b_success = task_successes(run_b, method_b)
        task_ids = sorted(set(a_success) & set(b_success))
        if set(a_success) != set(b_success):
            raise ValueError(f"seed {seed}: task sets differ")
        a_values = np.array([int(a_success[t]) for t in task_ids])
        b_values = np.array([int(b_success[t]) for t in task_ids])
        differences = b_values - a_values
        samples = rng.choice(differences, size=(bootstrap, len(differences)), replace=True).mean(axis=1)
        per_seed.append(
            {
                "seed": seed,
                "tasks": len(task_ids),
                "a_pass": float(a_values.mean()),
                "b_pass": float(b_values.mean()),
                "delta_pp": 100.0 * float(differences.mean()),
                "ci_pp": [
                    100.0 * float(np.quantile(samples, 0.025)),
                    100.0 * float(np.quantile(samples, 0.975)),
                ],
                "run_a": run_a.name,
                "run_b": run_b.name,
            }
        )
        differences_by_seed.append(differences)
        pooled_a.extend(a_values.tolist())
        pooled_b.extend(b_values.tolist())

    combined = np.array(
        [
            np.mean([rng.choice(d, size=len(d), replace=True).mean() for d in differences_by_seed])
            for _ in range(bootstrap)
        ]
    )
    pooled_a_array, pooled_b_array = np.array(pooled_a), np.array(pooled_b)
    b_only = int(np.sum((pooled_b_array == 1) & (pooled_a_array == 0)))
    a_only = int(np.sum((pooled_a_array == 1) & (pooled_b_array == 0)))
    discordant = b_only + a_only
    ci = [100.0 * float(np.quantile(combined, 0.025)), 100.0 * float(np.quantile(combined, 0.975))]
    deltas = [item["delta_pp"] for item in per_seed]
    return {
        "per_seed": per_seed,
        "mean_a_pass": float(np.mean([item["a_pass"] for item in per_seed])),
        "mean_b_pass": float(np.mean([item["b_pass"] for item in per_seed])),
        "mean_delta_pp": float(np.mean(deltas)),
        "hierarchical_bootstrap_95_ci_pp": ci,
        "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
        "positive_direction_seeds": int(sum(value > 0 for value in deltas)),
        "seed_count": len(per_seed),
        "b_only": b_only,
        "a_only": a_only,
        "mcnemar_exact_two_sided_p": float(binomtest(b_only, discordant, 0.5).pvalue) if discordant else 1.0,
    }


def holm(entries: dict[str, float]) -> dict[str, float]:
    ordered = sorted(entries.items(), key=lambda item: item[1])
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * p_value))
        adjusted[name] = running
    return adjusted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--dataset", default="confirmatory_heldout")
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports/confirmatory_statistics.json"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--rng-seed", type=int, default=20260801)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--contrasts",
        nargs="+",
        choices=["Q1", "Q2iid", "Q2prefix", "Q3", "Q4"],
        default=["Q1", "Q2iid", "Q2prefix", "Q3", "Q4"],
        help=(
            "Subset to compute. The July continuity grid only carries A1 and A4, "
            "so Q1 and Q4 are unavailable there."
        ),
    )
    args = parser.parse_args()

    found = discover(args.runs)
    dataset = args.dataset

    def run_for(role: str, seed: int) -> Path:
        key = (dataset, role, seed)
        if key not in found:
            raise SystemExit(f"missing evaluation run for {key}; run scripts/run_confirmatory_eval.sh first")
        return found[key]

    rng = np.random.default_rng(args.rng_seed)
    results = {}

    definitions = {
        "Q1": (
            "Q1_primary_structure_vs_matched_diversity",
            lambda s: (s, run_for("pre", s), A2, run_for("pre", s), A4),
        ),
        "Q2iid": (
            "Q2_grpo_effect_iid_branch",
            lambda s: (s, run_for("pre", s), A1, run_for("iidgrpo", s), A1),
        ),
        "Q2prefix": (
            "Q2_grpo_effect_prefix_branch",
            lambda s: (s, run_for("pre", s), A4, run_for("prefixgrpo", s), A4),
        ),
        "Q3": (
            "Q3_published_diagonal",
            lambda s: (s, run_for("iidgrpo", s), A1, run_for("prefixgrpo", s), A4),
        ),
        "Q4": (
            "Q4_hot_iid_vs_structure",
            lambda s: (s, run_for("pre", s), A4, run_for("pre", s), A3),
        ),
    }
    for key in args.contrasts:
        name, pair = definitions[key]
        results[name] = contrast([pair(s) for s in args.seeds], rng, args.bootstrap)

    secondary = {
        name: results[name]["mcnemar_exact_two_sided_p"]
        for name in (
            "Q2_grpo_effect_iid_branch",
            "Q2_grpo_effect_prefix_branch",
            "Q3_published_diagonal",
            "Q4_hot_iid_vs_structure",
        )
        if name in results
    }
    for name, adjusted in holm(secondary).items():
        results[name]["mcnemar_p_holm_adjusted"] = adjusted

    # Descriptive arm table: pass@32, diversity and cost per role and arm.
    arms_table = {}
    for role in ("pre", "iidgrpo", "prefixgrpo"):
        for arm in (A1, A2, A3, A4, A5):
            rows = []
            for seed in args.seeds:
                final = json.loads((run_for(role, seed) / "final_metrics.json").read_text())
                if arm in final["methods"]:
                    rows.append(final["methods"][arm])
            if rows:
                arms_table[f"{role}|{arm}"] = {
                    "pass_at_32": float(np.mean([r["pass_at_32"] for r in rows])),
                    "distinct_programs_per_task": float(np.mean([r["distinct_programs_per_task"] for r in rows])),
                    "unique_correct_programs": float(np.mean([r["unique_correct_programs"] for r in rows])),
                    "all_solution_coverage": float(np.mean([r["all_solution_coverage"] for r in rows])),
                    "model_passes": float(np.mean([r["model_passes"] for r in rows])),
                    "seeds": len(rows),
                }

    payload = {
        "dataset": dataset,
        "preregistration": "artifacts/preregistration_confirmatory_2026-08-01.md",
        "bootstrap_repetitions": args.bootstrap,
        "arms": {"A1": A1, "A2": A2, "A3": A3, "A4": A4, "A5": A5},
        "arm_table": arms_table,
        "contrasts": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"dataset: {dataset}\n")
    print(f"{'role|arm':52s} {'pass@32':>8s} {'distinct':>9s} {'uniqOK':>7s} {'cover':>7s} {'passes':>8s}")
    for key, row in arms_table.items():
        print(
            f"{key:52s} {row['pass_at_32']:8.3f} {row['distinct_programs_per_task']:9.2f} "
            f"{row['unique_correct_programs']:7.1f} {row['all_solution_coverage']:7.3f} {row['model_passes']:8.0f}"
        )
    print()
    for name, item in results.items():
        holm_note = (
            f"  holm_p={item['mcnemar_p_holm_adjusted']:.3g}" if "mcnemar_p_holm_adjusted" in item else ""
        )
        print(
            f"{name}\n  b-a = {item['mean_delta_pp']:+.2f} pp  "
            f"CI95 [{item['hierarchical_bootstrap_95_ci_pp'][0]:+.2f}, "
            f"{item['hierarchical_bootstrap_95_ci_pp'][1]:+.2f}]  "
            f"excludes_zero={item['ci_excludes_zero']}  "
            f"dir {item['positive_direction_seeds']}/{item['seed_count']}  "
            f"mcnemar_p={item['mcnemar_exact_two_sided_p']:.3g}{holm_note}"
        )
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
