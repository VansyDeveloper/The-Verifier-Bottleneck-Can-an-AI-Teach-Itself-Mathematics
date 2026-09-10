"""Analysis for the post-hoc control series (night run).

Writes to its own files. `artifacts/reports/ranking_statistics.json` and
`RANKING_FINAL_REPORT.md` are the frozen record of the registered 32-cell grid and
are deliberately NOT regenerated here: `analyze_ranking.py` sums its compute
totals and run index over every ranking run it discovers, so re-running it after
new cells exist would silently change a file that is currently byte-reproducible.

Statistics are imported from `analyze_ranking` rather than reimplemented, so the
crossed training-seed-by-task bootstrap, the centred bootstrap p-value and the CI
convention are identical to the registered analysis.

Every contrast is skipped with a recorded reason when its cells are absent, so
this script is safe to run against a partially completed series.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_ranking import _bootstrap_p, _ci_pp, _crossed_draws  # noqa: E402

# adapter directory name -> role label
ROLE_PATTERNS = [
    ("sft_atomic_r32_pilot_aw4_cont", "atomic"),
    ("sft_composition_oracle_seed", "oracle"),
    ("sft_composition_nomotif_seed", "nomotif"),
    ("sft_atomic_control_eqbudget_seed", "atomctl"),
    ("sft_composition_applykeep_seed", "applykeep"),
]
SEEDS = (0, 1, 2)
ORACLE = [f"oracle{s}" for s in SEEDS]
NOMOTIF = [f"nomotif{s}" for s in SEEDS]
ATOMCTL = [f"atomctl{s}" for s in SEEDS]
APPLYKEEP = [f"applykeep{s}" for s in SEEDS]


def role_of(adapter: str) -> str | None:
    name = Path(adapter.replace("\\", "/")).name
    for pattern, label in ROLE_PATTERNS:
        if name == pattern:
            return label
        if name.startswith(pattern):
            return f"{label}{name[len(pattern):]}"
    if name.startswith("sft_composition_noisy_b"):
        return "noisy" + name[len("sft_composition_noisy_b") :].split("_")[0]
    return None


def discover(runs_root: Path) -> dict[tuple[str, str], Path]:
    """Complete full-size exhaustive-ranking runs, keyed by (split, role)."""
    found: dict[tuple[str, str], Path] = {}
    for run in sorted(runs_root.iterdir()):
        config_path = run / "config.resolved.yaml"
        if not (run / "DONE").exists() or not config_path.exists():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("metric") != "exhaustive_hit_at_k":
            continue
        role = role_of(str(config["adapter"]))
        if role is None:
            continue
        input_path = Path(str(config["input"]).replace("\\", "/"))
        evaluated = json.loads((run / "final_metrics.json").read_text(encoding="utf-8"))["tasks"]
        available = sum(1 for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip())
        if evaluated != available:
            continue
        key = (input_path.stem, role)
        if key in found:
            raise RuntimeError(f"two complete ranking runs for {key}")
        found[key] = run
    return found


def values(run: Path, metric: str, k: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in (run / "generations.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        task_id = record["task_id"]
        if task_id in out:
            raise RuntimeError(f"duplicate task {task_id} in {run.name}")
        rank = record["best_correct_rank"]
        if metric == "hit":
            out[task_id] = float(rank is not None and rank <= k)
        elif metric == "mass":
            out[task_id] = float(record["correct_mass"])
        elif metric == "reciprocal_rank":
            out[task_id] = 0.0 if rank is None else 1.0 / float(rank)
        else:
            raise ValueError(f"unknown metric {metric}")
    return out


def matrix(found, split, roles, metric, k):
    rows, ids = [], None
    for role in roles:
        row = values(found[(split, role)], metric, k)
        if ids is None:
            ids = sorted(row)
        elif set(row) != set(ids):
            raise RuntimeError(f"task IDs differ across roles on {split}")
        rows.append(np.array([row[t] for t in ids], dtype=float))
    return np.stack(rows), ids


def contrast(found, split, base_roles, branch_roles, metric, k, rng, bootstrap):
    """Branch-minus-baseline on shared tasks, crossed over training seeds.

    A single-role baseline (the deterministic atomic adapter) is broadcast across
    the branch seeds; equal-length role lists are paired seed by seed.
    """
    branch, ids = matrix(found, split, branch_roles, metric, k)
    base, _ = matrix(found, split, base_roles, metric, k)
    if base.shape[0] == 1 and branch.shape[0] > 1:
        base = np.repeat(base, branch.shape[0], axis=0)
    if base.shape != branch.shape:
        raise RuntimeError(f"cannot pair {base_roles} with {branch_roles} on {split}")
    differences = branch - base
    draws = _crossed_draws(differences, rng, bootstrap)
    point = float(differences.mean())
    ci = _ci_pp(draws)
    per_seed = [
        {
            "baseline_role": b,
            "branch_role": r,
            "baseline": float(bv.mean()),
            "branch": float(rv.mean()),
            "delta_pp": 100.0 * float((rv - bv).mean()),
        }
        for b, r, bv, rv in zip(
            base_roles if len(base_roles) == len(branch_roles) else base_roles * len(branch_roles),
            branch_roles,
            base,
            branch,
        )
    ]
    deltas = np.array([item["delta_pp"] for item in per_seed])
    return {
        "split": split,
        "metric": metric,
        "k": k if metric == "hit" else None,
        "tasks": len(ids),
        "baseline_roles": list(base_roles),
        "branch_roles": list(branch_roles),
        "per_seed": per_seed,
        "mean_baseline": float(base.mean()),
        "mean_branch": float(branch.mean()),
        "mean_delta_pp": 100.0 * point,
        "crossed_seed_task_bootstrap_95_ci_pp": ci,
        "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
        "positive_direction_seeds": int(np.count_nonzero(deltas > 0)),
        "negative_direction_seeds": int(np.count_nonzero(deltas < 0)),
        "seed_count": len(per_seed),
        "bootstrap_two_sided_p": _bootstrap_p(draws, point),
    }


def motif_main_effect(found, roles, metric, k, rng, bootstrap):
    families = {"A": "motif_a_d3", "B": "motif_b_d3", "C": "motif_c_d3", "D": "motif_d_d3"}
    matrices = {}
    for label, split in families.items():
        branch, _ = matrix(found, split, roles, metric, k)
        base, _ = matrix(found, split, ["atomic"], metric, k)
        matrices[label] = branch - np.repeat(base, branch.shape[0], axis=0)
    motif_draws = np.empty(bootstrap)
    field_draws = np.empty(bootstrap)
    for i in range(bootstrap):
        seeds = rng.integers(0, len(roles), len(roles))
        means = {}
        for label, v in matrices.items():
            tasks = rng.integers(0, v.shape[1], v.shape[1])
            means[label] = v[np.ix_(seeds, tasks)].mean()
        motif_draws[i] = (means["A"] + means["C"] - means["B"] - means["D"]) / 2
        field_draws[i] = (means["A"] + means["B"] - means["C"] - means["D"]) / 2
    cell = {label: float(v.mean()) for label, v in matrices.items()}
    motif = (cell["A"] + cell["C"] - cell["B"] - cell["D"]) / 2
    field = (cell["A"] + cell["B"] - cell["C"] - cell["D"]) / 2
    motif_ci, field_ci = _ci_pp(motif_draws), _ci_pp(field_draws)
    return {
        "metric": metric,
        "k": k if metric == "hit" else None,
        "branch_roles": list(roles),
        "cell_delta_pp": {label: 100.0 * value for label, value in cell.items()},
        "motif_effect_pp": 100.0 * motif,
        "motif_95_ci_pp": motif_ci,
        "motif_ci_excludes_zero": bool(motif_ci[0] > 0 or motif_ci[1] < 0),
        "motif_bootstrap_two_sided_p": _bootstrap_p(motif_draws, motif),
        "field_effect_pp": 100.0 * field,
        "field_95_ci_pp": field_ci,
        "field_ci_excludes_zero": bool(field_ci[0] > 0 or field_ci[1] < 0),
    }


# name -> (question, split, baseline roles, branch roles, metric, k)
PLAN: list[tuple] = [
    # N1 - continuous endpoints on the already-registered grid, no new compute.
    ("N1a_mass_oracle_vs_atomic_d3_heldout",
     "probability mass on correct programs, held-out fields",
     "capacity_d3_heldout", ["atomic"], ORACLE, "mass", 32),
    ("N1b_mass_oracle_vs_atomic_d3_train",
     "probability mass on correct programs, known fields",
     "capacity_d3_train", ["atomic"], ORACLE, "mass", 32),
    ("N1c_mrr_oracle_vs_atomic_d3_heldout",
     "reciprocal rank of the best correct program, held-out fields",
     "capacity_d3_heldout", ["atomic"], ORACLE, "reciprocal_rank", 32),
    ("N1d_mass_nomotif_vs_atomic_motif_a",
     "probability mass, seen motifs / known fields",
     "motif_a_d3", ["atomic"], NOMOTIF, "mass", 32),
    ("N1e_mass_nomotif_vs_atomic_motif_b",
     "probability mass, held motifs / known fields",
     "motif_b_d3", ["atomic"], NOMOTIF, "mass", 32),
    ("N1f_mass_nomotif_vs_atomic_motif_c",
     "probability mass, seen motifs / new fields",
     "motif_c_d3", ["atomic"], NOMOTIF, "mass", 32),
    ("N1g_mass_nomotif_vs_atomic_motif_d",
     "probability mass, held motifs / new fields",
     "motif_d_d3", ["atomic"], NOMOTIF, "mass", 32),
    # N2 - the missing D-022 positive control.
    ("N2a_oracle_vs_atomic_motif_b",
     "POSITIVE CONTROL: are held-motif tasks rankable when the motif IS in training?",
     "motif_b_d3", ["atomic"], ORACLE, "hit", 32),
    ("N2b_oracle_vs_atomic_motif_d",
     "POSITIVE CONTROL: same, new fields",
     "motif_d_d3", ["atomic"], ORACLE, "hit", 32),
    ("N2c_oracle_vs_nomotif_motif_b",
     "effect of withholding the motif, seed-paired, known fields",
     "motif_b_d3", NOMOTIF, ORACLE, "hit", 32),
    ("N2d_oracle_vs_nomotif_motif_d",
     "effect of withholding the motif, seed-paired, new fields",
     "motif_d_d3", NOMOTIF, ORACLE, "hit", 32),
    ("N2e_mass_oracle_vs_nomotif_motif_b",
     "same on probability mass",
     "motif_b_d3", NOMOTIF, ORACLE, "mass", 32),
    # N3 - equal-budget control for C1/C2.
    ("N3a_atomctl_vs_atomic_d3_train",
     "does 1875 more steps of atomic training alone move depth-3 ranking?",
     "capacity_d3_train", ["atomic"], ATOMCTL, "hit", 32),
    ("N3b_atomctl_vs_atomic_d3_heldout",
     "same, held-out fields",
     "capacity_d3_heldout", ["atomic"], ATOMCTL, "hit", 32),
    ("N3c_oracle_vs_atomctl_d3_train",
     "C2 de-confounded: composition data vs equal-budget atomic data",
     "capacity_d3_train", ATOMCTL, ORACLE, "hit", 32),
    ("N3d_oracle_vs_atomctl_d3_heldout",
     "C1 de-confounded: composition data vs equal-budget atomic data",
     "capacity_d3_heldout", ATOMCTL, ORACLE, "hit", 32),
    # N4 - composition supervision with APPLY retained in the mixture.
    ("N4a_applykeep_vs_atomic_d3_train",
     "composition gain when APPLY stays in the mixture, known fields",
     "capacity_d3_train", ["atomic"], APPLYKEEP, "hit", 32),
    ("N4b_applykeep_vs_atomic_d3_heldout",
     "composition gain when APPLY stays in the mixture, held-out fields",
     "capacity_d3_heldout", ["atomic"], APPLYKEEP, "hit", 32),
    ("N4c_applykeep_vs_oracle_d3_heldout",
     "ranking cost of keeping APPLY, seed-paired",
     "capacity_d3_heldout", ORACLE, APPLYKEEP, "hit", 32),
    # N5 - depth 4.
    ("N5a_oracle_vs_atomic_d4_hit32",
     "does the composition gain survive at depth 4 (625 candidates)?",
     "capacity_d4_train", ["atomic"], ORACLE, "hit", 32),
    ("N5b_oracle_vs_atomic_d4_hit8",
     "same at Hit@8",
     "capacity_d4_train", ["atomic"], ORACLE, "hit", 8),
    ("N5c_mass_oracle_vs_atomic_d4",
     "same on probability mass",
     "capacity_d4_train", ["atomic"], ORACLE, "mass", 32),
    # N6 - degraded verifier on the distillation labels (one seed per beta).
    ("N6a_noisy10_vs_atomic_d3_heldout",
     "distillation gain when the verifier false-accepts 10% of trajectories",
     "capacity_d3_heldout", ["atomic"], ["noisy10"], "hit", 32),
    ("N6b_noisy25_vs_atomic_d3_heldout",
     "same at beta = 0.25",
     "capacity_d3_heldout", ["atomic"], ["noisy25"], "hit", 32),
    ("N6c_noisy50_vs_atomic_d3_heldout",
     "same at beta = 0.50",
     "capacity_d3_heldout", ["atomic"], ["noisy50"], "hit", 32),
    # The clean comparison: same seed, same recipe, only the labels differ.
    ("N6d_noisy10_vs_exact_labels",
     "gain lost to a 10% false-accept rate, against the exact-verifier adapter",
     "capacity_d3_heldout", ["oracle0"], ["noisy10"], "hit", 32),
    ("N6e_noisy25_vs_exact_labels",
     "same at beta = 0.25",
     "capacity_d3_heldout", ["oracle0"], ["noisy25"], "hit", 32),
    ("N6f_noisy50_vs_exact_labels",
     "same at beta = 0.50",
     "capacity_d3_heldout", ["oracle0"], ["noisy50"], "hit", 32),
    ("N6g_mass_noisy50_vs_exact_labels",
     "same on probability mass at beta = 0.50",
     "capacity_d3_heldout", ["oracle0"], ["noisy50"], "mass", 32),
]


def report(payload: dict) -> str:
    rows = payload["contrasts"]
    lines = [
        "# Post-hoc control series - results",
        "",
        "These are **disclosed post-hoc controls**, not confirmatory tests. Every",
        "evaluation set they use was already opened by D-018 / D-020 / D-021 / D-022,",
        "with the single exception of `capacity_d4_train`, which is newly generated here.",
        "Their purpose is to rule out alternative explanations of results that are already",
        "reported, so they carry no new preregistered decision rule and no Holm family.",
        "",
        "The registered `ranking_statistics.json` and `RANKING_FINAL_REPORT.md` are",
        "untouched; statistics come from the same crossed seed-by-task bootstrap.",
        "",
        "| contrast | question | delta (pp) | 95% CI | seeds +/- | p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, question, *_ in PLAN:
        item = rows.get(name)
        if item is None or "skipped" in item:
            reason = (item or {}).get("skipped", "not run")
            lines.append(f"| `{name}` | {question} | — | — | — | {reason} |")
            continue
        low, high = item["crossed_seed_task_bootstrap_95_ci_pp"]
        lines.append(
            f"| `{name}` | {question} | {item['mean_delta_pp']:+.2f} | "
            f"[{low:+.2f}, {high:+.2f}] | "
            f"{item['positive_direction_seeds']}/{item['negative_direction_seeds']} | "
            f"{item['bootstrap_two_sided_p']:.4g} |"
        )
    for key, title in (
        ("motif_main_effect_mass", "Motif main effect on probability mass"),
        ("motif_main_effect_hit32_oracle", "Motif main effect for the oracle adapters (Hit@32)"),
    ):
        item = payload.get(key)
        if not item or "skipped" in item:
            continue
        m_low, m_high = item["motif_95_ci_pp"]
        f_low, f_high = item["field_95_ci_pp"]
        lines += [
            "",
            f"## {title}",
            "",
            f"- motif effect **{item['motif_effect_pp']:+.2f} pp**, 95% CI [{m_low:+.2f}, {m_high:+.2f}]",
            f"- field effect **{item['field_effect_pp']:+.2f} pp**, 95% CI [{f_low:+.2f}, {f_high:+.2f}]",
            "- per-family deltas (pp): "
            + ", ".join(f"{k} {v:+.2f}" for k, v in sorted(item["cell_delta_pp"].items())),
        ]
    lines += [
        "",
        "## Cells discovered",
        "",
        "```",
        *[f"{split}|{role}" for split, role in sorted(payload["cells"])],
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("artifacts/runs"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/reports/night_statistics.json")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("artifacts/reports/NIGHT_CONTROLS_REPORT.md")
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--rng-seed", type=int, default=20260805)
    args = parser.parse_args()

    found = discover(args.runs)
    rng = np.random.default_rng(args.rng_seed)
    results: dict[str, dict] = {}

    for name, question, split, base_roles, branch_roles, metric, k in PLAN:
        missing = [
            f"{split}|{role}"
            for role in [*base_roles, *branch_roles]
            if (split, role) not in found
        ]
        if missing:
            results[name] = {"question": question, "skipped": f"missing {', '.join(missing)}"}
            continue
        try:
            item = contrast(found, split, base_roles, branch_roles, metric, k, rng, args.bootstrap)
            results[name] = {"question": question, **item}
        except Exception as exc:  # a bad cell must not lose the rest of the series
            results[name] = {"question": question, "skipped": f"{type(exc).__name__}: {exc}"}

    payload = {
        "status": "post-hoc disclosed controls; not confirmatory",
        "frozen_files_not_regenerated": [
            "artifacts/reports/ranking_statistics.json",
            "artifacts/reports/RANKING_FINAL_REPORT.md",
        ],
        "bootstrap_repetitions": args.bootstrap,
        "bootstrap_unit": "crossed training seed x task",
        "cells": sorted(found),
        "contrasts": results,
    }
    for key, roles, metric in (
        ("motif_main_effect_mass", NOMOTIF, "mass"),
        ("motif_main_effect_hit32_oracle", ORACLE, "hit"),
    ):
        need = [(s, r) for s in ("motif_a_d3", "motif_b_d3", "motif_c_d3", "motif_d_d3")
                for r in ["atomic", *roles]]
        if all(cell in found for cell in need):
            payload[key] = motif_main_effect(found, roles, metric, 32, rng, args.bootstrap)
        else:
            payload[key] = {"skipped": "incomplete 2x2 for these roles"}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(report(payload), encoding="utf-8")

    for name, item in results.items():
        if "skipped" in item:
            print(f"{name}: SKIPPED ({item['skipped']})")
        else:
            low, high = item["crossed_seed_task_bootstrap_95_ci_pp"]
            print(f"{name}: {item['mean_delta_pp']:+.2f} pp [{low:+.2f}, {high:+.2f}]")
    print(f"\nwritten: {args.output}")
    print(f"written: {args.report}")


if __name__ == "__main__":
    main()
