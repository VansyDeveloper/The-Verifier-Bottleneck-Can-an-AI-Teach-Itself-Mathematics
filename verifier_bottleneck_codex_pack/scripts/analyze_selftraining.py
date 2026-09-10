"""D-020 analysis: does self-training help once exploration is adequate?

Computes exactly the four contrasts fixed in
`artifacts/preregistration_selftraining_2026-08-01.md`:

  Q5 primary   iid@8.0 on hot-GRPO minus iid@8.0 on pre-GRPO
  Q6 secondary iid@8.0 on hot-GRPO minus iid@8.0 on cold-GRPO
  Q7 secondary iid@100.0 minus prefix@0.7, pre-GRPO adapter
  Q8 secondary iid@8.0   minus prefix@0.7, pre-GRPO adapter

Holm across {Q6, Q7, Q8}; Q5 uncorrected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from analyze_confirmatory import contrast, holm

HOT, COLD, PREFIX = "iid_action@8.0", "iid_action@0.7", "prefix_balanced_action@0.7"
UNIFORM, MID = "iid_action@100.0", "iid_action@3.0"
ARMS = [COLD, MID, HOT, UNIFORM, PREFIX]

ROLE_BY_ADAPTER = {
    "sft_atomic_r32_pilot_aw4_cont": "pre",
    **{f"grpo_iid_action_400_seed{s}": "coldgrpo" for s in (0, 1, 2)},
    **{f"grpo_iid_action_hot_400_seed{s}": "hotgrpo" for s in (0, 1, 2)},
}
ROLE_LABEL = {"pre": "pre-GRPO", "coldgrpo": "GRPO T_train=0.7", "hotgrpo": "GRPO T_train=8.0"}


def discover(runs_root: Path, dataset: str):
    found: dict[tuple[str, int], Path] = {}
    for run in sorted(runs_root.iterdir()):
        config_path = run / "config.resolved.yaml"
        if not (run / "DONE").exists() or not config_path.exists():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or "methods" not in config or "adapter" not in config:
            continue
        if Path(str(config["input"]).replace("\\", "/")).stem != dataset:
            continue
        role = ROLE_BY_ADAPTER.get(Path(str(config["adapter"]).replace("\\", "/")).name)
        if role:
            found[(role, int(config["seed"]))] = run
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--dataset", default="confirmatory_heldout_v2")
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports/selftraining_statistics.json"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--rng-seed", type=int, default=20260802)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    found = discover(args.runs, args.dataset)

    def run_for(role: str, seed: int) -> Path:
        if (role, seed) not in found:
            raise SystemExit(f"missing run for {role} seed {seed}; run scripts/run_selftraining_series.sh")
        return found[(role, seed)]

    rng = np.random.default_rng(args.rng_seed)
    definitions = {
        "Q5_primary_selftraining_with_adequate_exploration": lambda s: (
            s, run_for("pre", s), HOT, run_for("hotgrpo", s), HOT
        ),
        "Q6_hot_versus_cold_training_sampler": lambda s: (
            s, run_for("coldgrpo", s), HOT, run_for("hotgrpo", s), HOT
        ),
        "Q7_uniform_random_versus_structure": lambda s: (
            s, run_for("pre", s), PREFIX, run_for("pre", s), UNIFORM
        ),
        "Q8_best_iid_versus_structure": lambda s: (
            s, run_for("pre", s), PREFIX, run_for("pre", s), HOT
        ),
    }
    results = {name: contrast([pair(s) for s in args.seeds], rng, args.bootstrap)
               for name, pair in definitions.items()}

    secondary = {
        name: results[name]["mcnemar_exact_two_sided_p"]
        for name in ("Q6_hot_versus_cold_training_sampler",
                     "Q7_uniform_random_versus_structure",
                     "Q8_best_iid_versus_structure")
    }
    for name, adjusted in holm(secondary).items():
        results[name]["mcnemar_p_holm_adjusted"] = adjusted

    arm_table = {}
    for role in ("pre", "coldgrpo", "hotgrpo"):
        for arm in ARMS:
            rows = [json.load(open(run_for(role, s) / "final_metrics.json"))["methods"][arm]
                    for s in args.seeds]
            arm_table[f"{role}|{arm}"] = {
                key: float(np.mean([row[key] for row in rows]))
                for key in ("pass_at_32", "distinct_programs_per_task", "unique_correct_programs",
                            "candidate_correct_rate", "all_solution_coverage", "model_passes")
            }

    payload = {
        "dataset": args.dataset,
        "preregistration": "artifacts/preregistration_selftraining_2026-08-01.md",
        "bootstrap_repetitions": args.bootstrap,
        "arm_table": arm_table,
        "contrasts": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"dataset: {args.dataset}  (300 tasks, mean of {len(args.seeds)} seeds)\n")
    print(f"{'adapter':18s} {'arm':22s} {'pass@32':>8s} {'distinct':>9s} {'cand_ok':>8s} {'uniqOK':>7s} {'passes':>8s}")
    for role in ("pre", "coldgrpo", "hotgrpo"):
        for arm in ARMS:
            row = arm_table[f"{role}|{arm}"]
            print(f"{ROLE_LABEL[role]:18s} {arm:22s} {row['pass_at_32']:8.3f} "
                  f"{row['distinct_programs_per_task']:9.2f} {row['candidate_correct_rate']:8.4f} "
                  f"{row['unique_correct_programs']:7.1f} {row['model_passes']:8.0f}")
        print()
    for name, item in results.items():
        low, high = item["hierarchical_bootstrap_95_ci_pp"]
        note = f"  holm_p={item['mcnemar_p_holm_adjusted']:.3g}" if "mcnemar_p_holm_adjusted" in item else ""
        print(f"{name}\n  delta = {item['mean_delta_pp']:+.2f} pp  CI95 [{low:+.2f}, {high:+.2f}]  "
              f"excludes_zero={item['ci_excludes_zero']}  dir {item['positive_direction_seeds']}/"
              f"{item['seed_count']}  mcnemar_p={item['mcnemar_exact_two_sided_p']:.3g}{note}")
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
