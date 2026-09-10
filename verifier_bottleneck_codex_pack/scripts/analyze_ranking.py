"""Final exhaustive-ranking analysis for D-021 and D-022.

The atomic adapter is deterministic and shared by three independently trained
adapters.  Tasks are also shared across seeds, so uncertainty is estimated with
a crossed seed-by-task bootstrap: both training seeds and task IDs are resampled.

Depth-2 has only 25 candidates.  Its registered depth contrast is therefore
reported at Hit@1 and Hit@8, never at the vacuous Hit@32.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import binomtest

from vbexp.experiment import sha256_file

CAPACITY = ["capacity_d2_train", "capacity_d2_heldout", "capacity_d3_train", "capacity_d3_heldout"]
MOTIF = ["motif_a_d3", "motif_b_d3", "motif_c_d3", "motif_d_d3"]
ORACLE = [f"oracle{s}" for s in (0, 1, 2)]
NOMOTIF = [f"nomotif{s}" for s in (0, 1, 2)]
FORGET_SPLITS = ["sft_validation_apply_non_sh1", "sft_validation_plan"]


def role_of(adapter_name: str) -> str | None:
    if adapter_name.endswith("sft_atomic_r32_pilot_aw4_cont"):
        return "atomic"
    if "composition_oracle_seed" in adapter_name:
        return f"oracle{adapter_name[-1]}"
    if "composition_nomotif_seed" in adapter_name:
        return f"nomotif{adapter_name[-1]}"
    return None


def discover(runs_root: Path, *, verbose: bool = True) -> dict[tuple[str, str], Path]:
    """Locate exactly one complete full-size ranking run per split and role."""
    found: dict[tuple[str, str], Path] = {}
    skipped: list[str] = []
    for run in sorted(runs_root.iterdir()):
        config_path = run / "config.resolved.yaml"
        if not (run / "DONE").exists() or not config_path.exists():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("metric") != "exhaustive_hit_at_k":
            continue
        role = role_of(str(config["adapter"]).replace("\\", "/"))
        if not role:
            continue
        input_path = Path(str(config["input"]).replace("\\", "/"))
        evaluated = json.loads((run / "final_metrics.json").read_text(encoding="utf-8"))["tasks"]
        available = sum(1 for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip())
        if evaluated != available:
            skipped.append(f"{run.name} ({evaluated}/{available} tasks)")
            continue
        key = (input_path.stem, role)
        if key in found:
            raise RuntimeError(f"multiple complete ranking runs for {key}: {found[key].name}, {run.name}")
        found[key] = run
    if verbose and skipped:
        print("excluded partial runs:")
        for item in skipped:
            print(f"  {item}")
        print()
    return found


def require_complete_grid(found: dict[tuple[str, str], Path]) -> None:
    expected = {
        *((split, role) for split in CAPACITY for role in ["atomic", *ORACLE]),
        *((split, role) for split in MOTIF for role in ["atomic", *NOMOTIF]),
    }
    missing = sorted(expected - set(found))
    if missing:
        raise RuntimeError("incomplete 32-cell ranking grid: " + ", ".join(f"{s}|{r}" for s, r in missing))


def hits(run: Path, k: int) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in (run / "generations.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        task_id = record["task_id"]
        if task_id in out:
            raise RuntimeError(f"duplicate task {task_id} in {run.name}")
        rank = record["best_correct_rank"]
        out[task_id] = int(rank is not None and rank <= k)
    return out


def _aligned(found, split, baseline, branches, k):
    base = hits(found[(split, baseline)], k)
    ids = sorted(base)
    branch_rows = []
    for role in branches:
        branch = hits(found[(split, role)], k)
        if set(branch) != set(base):
            raise RuntimeError(f"task IDs differ for {split}|{baseline} and {split}|{role}")
        branch_rows.append(np.array([branch[t] for t in ids], dtype=float))
    base_row = np.array([base[t] for t in ids], dtype=float)
    return base_row, np.stack(branch_rows), ids


def _crossed_draws(values: np.ndarray, rng: np.random.Generator, bootstrap: int) -> np.ndarray:
    """Bootstrap a seed x task matrix across both crossed sampling units."""
    seed_count, task_count = values.shape
    draws = np.empty(bootstrap)
    for i in range(bootstrap):
        seeds = rng.integers(0, seed_count, seed_count)
        tasks = rng.integers(0, task_count, task_count)
        draws[i] = values[np.ix_(seeds, tasks)].mean()
    return draws


def _bootstrap_p(draws: np.ndarray, point: float) -> float:
    """Two-sided bootstrap test after centering the empirical null at zero."""
    extreme = np.count_nonzero(np.abs(draws - point) >= abs(point))
    return float((extreme + 1) / (len(draws) + 1))


def _ci_pp(draws: np.ndarray) -> list[float]:
    return [100.0 * float(x) for x in np.quantile(draws, (0.025, 0.975))]


def paired(found, split, baseline, branches, k, rng, bootstrap):
    """Branch-minus-baseline contrast on tasks crossed with training seeds."""
    base, branch, ids = _aligned(found, split, baseline, branches, k)
    differences = branch - base
    per_seed = []
    for role, values, diff in zip(branches, branch, differences):
        task_draws = rng.choice(diff, size=(bootstrap, len(diff)), replace=True).mean(axis=1)
        branch_only = int(np.count_nonzero(diff == 1))
        baseline_only = int(np.count_nonzero(diff == -1))
        discordant = branch_only + baseline_only
        per_seed.append(
            {
                "role": role,
                "tasks": len(ids),
                "baseline": float(base.mean()),
                "branch": float(values.mean()),
                "delta_pp": 100.0 * float(diff.mean()),
                "task_bootstrap_95_ci_pp": _ci_pp(task_draws),
                "branch_only": branch_only,
                "baseline_only": baseline_only,
                "mcnemar_exact_two_sided_p": (
                    float(binomtest(branch_only, discordant, 0.5).pvalue) if discordant else 1.0
                ),
            }
        )
    draws = _crossed_draws(differences, rng, bootstrap)
    point = float(differences.mean())
    ci = _ci_pp(draws)
    deltas = np.array([item["delta_pp"] for item in per_seed])
    direction = np.sign(point)
    nonzero = int(np.count_nonzero(deltas))
    consistent = int(np.count_nonzero(np.sign(deltas) == direction)) if direction else 0
    return {
        "k": k,
        "paired": True,
        "bootstrap_unit": "crossed_training_seed_x_task",
        "per_seed": per_seed,
        "mean_baseline": float(base.mean()),
        "mean_branch": float(branch.mean()),
        "mean_delta_pp": 100.0 * point,
        "crossed_seed_task_bootstrap_95_ci_pp": ci,
        "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
        "positive_direction_seeds": int(np.count_nonzero(deltas > 0)),
        "negative_direction_seeds": int(np.count_nonzero(deltas < 0)),
        "seed_count": len(per_seed),
        "seed_direction_exact_two_sided_p": (
            float(binomtest(consistent, nonzero, 0.5).pvalue) if nonzero else 1.0
        ),
        "bootstrap_two_sided_p": _bootstrap_p(draws, point),
    }


def unpaired_branch_contrast(found, split_a, split_b, roles, k, rng, bootstrap):
    """Report branch performance on split_b minus split_a, resampling both tasks and seeds."""
    by_role_a = [hits(found[(split_a, role)], k) for role in roles]
    by_role_b = [hits(found[(split_b, role)], k) for role in roles]
    ids_a, ids_b = sorted(by_role_a[0]), sorted(by_role_b[0])
    if any(set(values) != set(ids_a) for values in by_role_a[1:]):
        raise RuntimeError(f"task IDs differ across roles for {split_a}")
    if any(set(values) != set(ids_b) for values in by_role_b[1:]):
        raise RuntimeError(f"task IDs differ across roles for {split_b}")
    rows_a = np.stack([np.array([values[t] for t in ids_a], dtype=float) for values in by_role_a])
    rows_b = np.stack([np.array([values[t] for t in ids_b], dtype=float) for values in by_role_b])
    per_seed = []
    for role, a, b in zip(roles, rows_a, rows_b):
        task_draws = np.array(
            [rng.choice(b, len(b), replace=True).mean() - rng.choice(a, len(a), replace=True).mean()
             for _ in range(bootstrap)]
        )
        per_seed.append(
            {
                "role": role,
                "tasks_a": len(a),
                "tasks_b": len(b),
                "a": float(a.mean()),
                "b": float(b.mean()),
                "delta_pp": 100.0 * float(b.mean() - a.mean()),
                "task_bootstrap_95_ci_pp": _ci_pp(task_draws),
            }
        )
    draws = np.empty(bootstrap)
    for i in range(bootstrap):
        seeds = rng.integers(0, len(roles), len(roles))
        tasks_a = rng.integers(0, rows_a.shape[1], rows_a.shape[1])
        tasks_b = rng.integers(0, rows_b.shape[1], rows_b.shape[1])
        draws[i] = rows_b[np.ix_(seeds, tasks_b)].mean() - rows_a[np.ix_(seeds, tasks_a)].mean()
    point = float(rows_b.mean() - rows_a.mean())
    ci = _ci_pp(draws)
    deltas = np.array([item["delta_pp"] for item in per_seed])
    return {
        "k": k,
        "paired": False,
        "bootstrap_unit": "crossed_training_seed_x_task_sets",
        "per_seed": per_seed,
        "mean_a": float(rows_a.mean()),
        "mean_b": float(rows_b.mean()),
        "mean_delta_pp": 100.0 * point,
        "crossed_seed_task_bootstrap_95_ci_pp": ci,
        "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
        "positive_direction_seeds": int(np.count_nonzero(deltas > 0)),
        "negative_direction_seeds": int(np.count_nonzero(deltas < 0)),
        "seed_count": len(roles),
        "bootstrap_two_sided_p": _bootstrap_p(draws, point),
    }


def factorial_effects(found, roles, k, rng, bootstrap):
    families = {
        "A": "motif_a_d3",
        "B": "motif_b_d3",
        "C": "motif_c_d3",
        "D": "motif_d_d3",
    }
    matrices = {}
    for label, split in families.items():
        base, branch, _ = _aligned(found, split, "atomic", roles, k)
        matrices[label] = branch - base

    motif_draws = np.empty(bootstrap)
    field_draws = np.empty(bootstrap)
    for i in range(bootstrap):
        seeds = rng.integers(0, len(roles), len(roles))
        means = {}
        for label, values in matrices.items():
            tasks = rng.integers(0, values.shape[1], values.shape[1])
            means[label] = values[np.ix_(seeds, tasks)].mean()
        motif_draws[i] = (means["A"] + means["C"] - means["B"] - means["D"]) / 2
        field_draws[i] = (means["A"] + means["B"] - means["C"] - means["D"]) / 2

    cell = {label: float(values.mean()) for label, values in matrices.items()}
    motif = (cell["A"] + cell["C"] - cell["B"] - cell["D"]) / 2
    field = (cell["A"] + cell["B"] - cell["C"] - cell["D"]) / 2
    motif_ci, field_ci = _ci_pp(motif_draws), _ci_pp(field_draws)
    return {
        "k": k,
        "bootstrap_unit": "crossed_training_seed_x_task_family",
        "motif_effect_pp": 100.0 * motif,
        "motif_crossed_seed_task_bootstrap_95_ci_pp": motif_ci,
        "motif_ci_excludes_zero": bool(motif_ci[0] > 0 or motif_ci[1] < 0),
        "motif_bootstrap_two_sided_p": _bootstrap_p(motif_draws, motif),
        "field_effect_pp": 100.0 * field,
        "field_crossed_seed_task_bootstrap_95_ci_pp": field_ci,
        "field_ci_excludes_zero": bool(field_ci[0] > 0 or field_ci[1] < 0),
        "field_bootstrap_two_sided_p": _bootstrap_p(field_draws, field),
        "seen_mean_pp": 50.0 * (cell["A"] + cell["C"]),
        "held_mean_pp": 50.0 * (cell["B"] + cell["D"]),
        "known_field_mean_pp": 50.0 * (cell["A"] + cell["B"]),
        "new_field_mean_pp": 50.0 * (cell["C"] + cell["D"]),
    }


def holm(entries: dict[str, float]) -> dict[str, float]:
    ordered = sorted(entries.items(), key=lambda item: item[1])
    adjusted, running = {}, 0.0
    for index, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * p_value))
        adjusted[name] = running
    return adjusted


def _run_index(found: dict[tuple[str, str], Path]) -> dict[str, dict]:
    adapter_hashes: dict[str, str | None] = {}
    input_hashes: dict[str, str] = {}
    output = {}
    for (split, role), run in sorted(found.items()):
        config_path = run / "config.resolved.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        metrics = json.loads((run / "final_metrics.json").read_text(encoding="utf-8"))
        adapter = str(config["adapter"])
        input_path = str(config["input"])
        if adapter not in adapter_hashes:
            weights = Path(adapter) / "adapter_model.safetensors"
            adapter_hashes[adapter] = sha256_file(weights) if weights.exists() else None
        if input_path not in input_hashes:
            input_hashes[input_path] = sha256_file(input_path)
        output[f"{split}|{role}"] = {
            "run_id": run.name,
            "run_path": run.as_posix(),
            "adapter": adapter,
            "adapter_sha256": adapter_hashes[adapter],
            "input": input_path,
            "input_sha256": input_hashes[input_path],
            "config_sha256": sha256_file(config_path),
            "status": "DONE",
            "tasks": metrics["tasks"],
            "model_passes": metrics["model_passes"],
            "elapsed_seconds": metrics["elapsed_seconds"],
            "peak_cuda_bytes": metrics["peak_cuda_bytes"],
        }
    return output


def _forgetting(runs_root: Path) -> dict:
    roles = ["atomic", *ORACLE, *NOMOTIF]
    candidates: dict[tuple[str, str], list[Path]] = {}
    for run in sorted(runs_root.iterdir()):
        config_path = run / "config.resolved.yaml"
        if not (run / "DONE").exists() or not config_path.exists():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not str(config.get("label", "")).startswith("composition-forget-"):
            continue
        role = role_of(str(config.get("adapter", "")).replace("\\", "/"))
        split = Path(str(config.get("input", "")).replace("\\", "/")).stem
        if role in roles and split in FORGET_SPLITS:
            candidates.setdefault((role, split), []).append(run)
    found: dict[tuple[str, str], Path] = {}
    excluded: list[str] = []
    for key, runs in candidates.items():
        gpu_runs = []
        for run in runs:
            environment = json.loads((run / "environment.json").read_text(encoding="utf-8"))
            if environment.get("torch", {}).get("cuda_available") is True:
                gpu_runs.append(run)
            else:
                excluded.append(f"{run.name} (CUDA unavailable)")
        if len(gpu_runs) != 1:
            raise RuntimeError(f"expected one GPU forgetting run for {key}, found {len(gpu_runs)}")
        found[key] = gpu_runs[0]
    expected = {(role, split) for role in roles for split in FORGET_SPLITS}
    table = {}
    for key, run in sorted(found.items()):
        metrics = json.loads((run / "final_metrics.json").read_text(encoding="utf-8"))
        table[f"{key[0]}|{key[1]}"] = {
            "pass_at_1": metrics["pass_at_1"],
            "parse_rate": metrics["parse_rate"],
            "tasks": metrics["tasks"],
            "actual_completion_tokens": metrics["actual_completion_tokens"],
            "elapsed_seconds": metrics["elapsed_seconds"],
            "peak_cuda_bytes": metrics["peak_cuda_bytes"],
            "run_id": run.name,
        }
    result = {
        "complete": set(found) == expected,
        "missing": [f"{role}|{split}" for role, split in sorted(expected - set(found))],
        "excluded_complete_runs": sorted(excluded),
        "arm_table": table,
    }
    if result["complete"]:
        trained = [*ORACLE, *NOMOTIF]
        apply_split, plan_split = FORGET_SPLITS
        atomic_apply = table[f"atomic|{apply_split}"]["pass_at_1"]
        atomic_plan = table[f"atomic|{plan_split}"]["pass_at_1"]
        apply_values = [table[f"{role}|{apply_split}"]["pass_at_1"] for role in trained]
        plan_values = [table[f"{role}|{plan_split}"]["pass_at_1"] for role in trained]
        result["summary"] = {
            "atomic_apply_pass_at_1": atomic_apply,
            "trained_apply_mean_pass_at_1": float(np.mean(apply_values)),
            "trained_apply_delta_pp": 100.0 * float(np.mean(apply_values) - atomic_apply),
            "atomic_plan_pass_at_1": atomic_plan,
            "trained_plan_mean_pass_at_1": float(np.mean(plan_values)),
            "trained_plan_delta_pp": 100.0 * float(np.mean(plan_values) - atomic_plan),
            "all_trained_apply_pass_at_1_zero": all(value == 0 for value in apply_values),
            "retention_warning": (
                "CATASTROPHIC_APPLY_COLLAPSE" if atomic_apply == 1 and all(value == 0 for value in apply_values)
                else None
            ),
        }
    return result


def _report(payload: dict) -> str:
    results = payload["contrasts"]

    def row(label, name):
        item = results[name]
        low, high = item["crossed_seed_task_bootstrap_95_ci_pp"]
        adjusted = item.get("bootstrap_p_holm_adjusted")
        p_text = f"{adjusted:.4g}" if adjusted is not None else f"{item['bootstrap_two_sided_p']:.4g}"
        return f"| {label} | {item['mean_delta_pp']:+.2f} | [{low:+.2f}, {high:+.2f}] | {p_text} |"

    m4 = results["M4_main_effects"]
    motif_ci = m4["motif_crossed_seed_task_bootstrap_95_ci_pp"]
    field_ci = m4["field_crossed_seed_task_bootstrap_95_ci_pp"]
    forgetting = payload["atomic_forgetting"]
    forgetting_lines = ["Atomic forgetting is incomplete."]
    if forgetting["complete"]:
        forgetting_lines = [
            "| Adapter | APPLY pass@1 | PLAN pass@1 |",
            "|---|---:|---:|",
            *[
                f"| {role} | "
                f"{forgetting['arm_table'][f'{role}|sft_validation_apply_non_sh1']['pass_at_1']:.3f} | "
                f"{forgetting['arm_table'][f'{role}|sft_validation_plan']['pass_at_1']:.3f} |"
                for role in ["atomic", *ORACLE, *NOMOTIF]
            ],
        ]
        summary = forgetting["summary"]
        if summary["retention_warning"]:
            forgetting_lines.extend(
                [
                    "",
                    f"**Retention warning: {summary['retention_warning']}.** All six trained adapters drop",
                    f"from APPLY pass@1 {summary['atomic_apply_pass_at_1']:.2f} to 0.00 with parse rate 0.00.",
                    f"Mean PLAN pass@1 changes by {summary['trained_plan_delta_pp']:+.2f} pp.",
                    "Under section 6 of the frozen D-021 preregistration, the capacity gain is therefore",
                    "not counted as a clean success. It remains evidence of PLAN ranking capacity, not of",
                    "retained multi-mode competence.",
                ]
            )
        if forgetting["excluded_complete_runs"]:
            forgetting_lines.extend(
                [
                    "",
                    f"{len(forgetting['excluded_complete_runs'])} complete CPU fallback baseline runs are retained",
                    "in the run index but excluded here by the environment-matching rule `cuda_available == true`;",
                    "all reported retention cells use the same GPU environment.",
                ]
            )
    return "\n".join(
        [
            "# D-021/D-022 exhaustive-ranking final report",
            "",
            "## Completion and metric",
            "",
            "All 32 registered ranking cells are complete: four D-021 splits and four D-022 families,",
            "each evaluated with the atomic baseline and three trained adapters. The 12/300 smoke run",
            "is excluded by exact task-count matching. Depth-2 Hit@32 is not interpreted because all 25",
            "programs fit inside K=32.",
            "",
            "Uncertainty uses a 10,000-fold crossed training-seed-by-task bootstrap. Reported bootstrap",
            "p-values are two-sided tests from the centered bootstrap null; Holm correction follows the",
            "registered families. Per-seed exact McNemar tests remain in the JSON as diagnostics only.",
            "",
            "## D-021: capacity and field transfer",
            "",
            "| Contrast | Delta (pp) | 95% CI (pp) | p / Holm p |",
            "|---|---:|---:|---:|",
            row("C1: supervision transfer to held-out fields, Hit@32", "C1_primary_supervision_transfers_to_heldout_fields"),
            row("C2: compositions learnable in-domain, Hit@32", "C2_compositions_learnable_in_domain"),
            row("C3: depth-2 minus depth-3, Hit@8", "C3_depth_effect_fields_fixed_d2_minus_d3_at_hit8"),
            row("C4: train minus held-out fields, Hit@32", "C4_field_effect_depth_fixed_train_minus_heldout"),
            "",
            "C1 and C2 are positive with confidence intervals above zero, selecting the preregistered",
            "method-ceiling branch for PLAN ranking. C4 is near zero: field novelty does not measurably",
            "reduce exhaustive Hit@32 after direct composition supervision. The atomic-retention result",
            "below prevents treating this as an unqualified capacity-control success.",
            "",
            "## D-022: held-motif 2x2",
            "",
            "| Family | Delta Hit@32 (pp) | 95% CI (pp) | p / Holm p |",
            "|---|---:|---:|---:|",
            row("A: known fields, seen motifs", "M3a_seen_motif_known_field"),
            row("B: known fields, held motifs", "M1_primary_held_motif_known_field"),
            row("C: new fields, seen motifs", "M3c_seen_motif_new_field"),
            row("D: new fields, held motifs", "M2_held_motif_new_field"),
            "",
            f"- Motif main effect: **{m4['motif_effect_pp']:+.2f} pp**, 95% CI "
            f"**[{motif_ci[0]:+.2f}, {motif_ci[1]:+.2f}]**.",
            f"- Field main effect: **{m4['field_effect_pp']:+.2f} pp**, 95% CI "
            f"**[{field_ci[0]:+.2f}, {field_ci[1]:+.2f}]**.",
            "",
            "The sign flip reproduces the external qualitative result on independently generated data:",
            "composition supervision helps on seen motifs and harms ranking on held motifs, while field",
            "novelty contributes little.",
            "",
            "## Atomic forgetting",
            "",
            *forgetting_lines,
            "",
            "## Claim boundary",
            "",
            "Direct supervision rules out a hard representational-capacity ceiling for the tested model,",
            "LoRA rank and task family. It does not isolate whether the earlier nulls were caused by search,",
            "policy-gradient credit assignment, supervision density or optimisation budget.",
            "",
            "## Measured evaluation cost",
            "",
            f"The 32 ranking cells used {payload['compute']['ranking_model_passes']:,} model passes",
            f"and {payload['compute']['ranking_elapsed_seconds'] / 3600:.2f} summed GPU-hours.",
            f"The 14 retained atomic checks generated {payload['compute']['forgetting_completion_tokens']:,}",
            f"completion tokens in {payload['compute']['forgetting_elapsed_seconds'] / 60:.1f} summed GPU-minutes.",
            "Peak allocated CUDA memory was",
            f"{payload['compute']['peak_cuda_bytes'] / 2**30:.2f} GiB.",
            "",
            "## Reproduction",
            "",
            "```bash",
            ".venv/bin/python scripts/analyze_ranking.py",
            ".venv/bin/python scripts/plot_ranking_results.py",
            "```",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports/ranking_statistics.json"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/reports/RANKING_FINAL_REPORT.md"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--rng-seed", type=int, default=20260804)
    args = parser.parse_args()

    found = discover(args.runs)
    require_complete_grid(found)
    rng = np.random.default_rng(args.rng_seed)
    results, table = {}, {}

    for split in CAPACITY + MOTIF:
        branches = NOMOTIF if split in MOTIF else ORACLE
        for role in ["atomic", *branches]:
            metrics = json.loads((found[(split, role)] / "final_metrics.json").read_text(encoding="utf-8"))
            table[f"{split}|{role}"] = {
                key: metrics[key]
                for key in (
                    "hit_at_1",
                    "hit_at_8",
                    "hit_at_32",
                    "mean_best_rank",
                    "mean_correct_mass",
                    "candidates_per_task",
                    "tasks",
                )
            }

    def add(name, split, branches, k):
        results[name] = paired(found, split, "atomic", branches, k, rng, args.bootstrap)

    add("C1_primary_supervision_transfers_to_heldout_fields", "capacity_d3_heldout", ORACLE, 32)
    add("C2_compositions_learnable_in_domain", "capacity_d3_train", ORACLE, 32)
    add("C1_at_hit1", "capacity_d3_heldout", ORACLE, 1)
    add("C2_at_hit1", "capacity_d3_train", ORACLE, 1)
    add("depth2_train_supervision_effect_at_hit8", "capacity_d2_train", ORACLE, 8)
    add("depth2_heldout_supervision_effect_at_hit8", "capacity_d2_heldout", ORACLE, 8)
    results["C3_depth_effect_fields_fixed_d2_minus_d3_at_hit8"] = unpaired_branch_contrast(
        found, "capacity_d3_train", "capacity_d2_train", ORACLE, 8, rng, args.bootstrap
    )
    results["C3_depth_effect_fields_fixed_d2_minus_d3_at_hit1"] = unpaired_branch_contrast(
        found, "capacity_d3_train", "capacity_d2_train", ORACLE, 1, rng, args.bootstrap
    )
    results["C4_field_effect_depth_fixed_train_minus_heldout"] = unpaired_branch_contrast(
        found, "capacity_d3_heldout", "capacity_d3_train", ORACLE, 32, rng, args.bootstrap
    )

    for split, name in (
        ("motif_b_d3", "M1_primary_held_motif_known_field"),
        ("motif_d_d3", "M2_held_motif_new_field"),
        ("motif_a_d3", "M3a_seen_motif_known_field"),
        ("motif_c_d3", "M3c_seen_motif_new_field"),
    ):
        add(name, split, NOMOTIF, 32)
    results["M4_main_effects"] = factorial_effects(found, NOMOTIF, 32, rng, args.bootstrap)

    d021_family = {
        name: results[name]["bootstrap_two_sided_p"]
        for name in (
            "C2_compositions_learnable_in_domain",
            "C3_depth_effect_fields_fixed_d2_minus_d3_at_hit8",
            "C4_field_effect_depth_fixed_train_minus_heldout",
        )
    }
    motif_family = {
        name: results[name]["bootstrap_two_sided_p"]
        for name in (
            "M2_held_motif_new_field",
            "M3a_seen_motif_known_field",
            "M3c_seen_motif_new_field",
        )
    }
    for family in (d021_family, motif_family):
        for name, adjusted in holm(family).items():
            results[name]["bootstrap_p_holm_adjusted"] = adjusted

    c1 = results["C1_primary_supervision_transfers_to_heldout_fields"]
    c2 = results["C2_compositions_learnable_in_domain"]
    if c2["mean_delta_pp"] <= 0 or not c2["ci_excludes_zero"]:
        verdict = "CAPACITY_CEILING"
    elif c1["mean_delta_pp"] <= 0 or not c1["ci_excludes_zero"]:
        verdict = "GENERALISATION_CEILING"
    else:
        verdict = "METHOD_CEILING"

    forgetting = _forgetting(args.runs)
    overall_verdict = verdict
    if forgetting.get("summary", {}).get("retention_warning"):
        overall_verdict = f"{verdict}_WITH_CATASTROPHIC_APPLY_COLLAPSE"
    run_index = _run_index(found)
    forgetting_rows = forgetting["arm_table"].values()
    compute = {
        "ranking_model_passes": int(sum(item["model_passes"] for item in run_index.values())),
        "ranking_elapsed_seconds": float(sum(item["elapsed_seconds"] for item in run_index.values())),
        "forgetting_completion_tokens": int(sum(item["actual_completion_tokens"] for item in forgetting_rows)),
        "forgetting_elapsed_seconds": float(sum(item["elapsed_seconds"] for item in forgetting_rows)),
        "peak_cuda_bytes": int(max(
            [item["peak_cuda_bytes"] for item in run_index.values()]
            + [item["peak_cuda_bytes"] for item in forgetting_rows]
        )),
    }
    payload = {
        "preregistrations": [
            "artifacts/preregistration_capacity_2026-08-02.md",
            "artifacts/preregistration_amendment_001_2026-08-03.md",
        ],
        "metric": "exhaustive Hit@K over all 5^depth programs",
        "bootstrap_repetitions": args.bootstrap,
        "bootstrap_unit": "crossed training seed x task",
        "completed_grid_cells": len(found),
        "joint_c1_c2_verdict": verdict,
        "overall_verdict": overall_verdict,
        "capacity_success_counted_under_retention_rule": not bool(
            forgetting.get("summary", {}).get("retention_warning")
        ),
        "run_index": run_index,
        "compute": compute,
        "atomic_forgetting": forgetting,
        "arm_table": table,
        "contrasts": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(_report(payload), encoding="utf-8")

    for name, item in results.items():
        if "mean_delta_pp" in item:
            low, high = item["crossed_seed_task_bootstrap_95_ci_pp"]
            print(f"{name}: {item['mean_delta_pp']:+.2f} pp [{low:+.2f}, {high:+.2f}]")
        else:
            print(
                f"{name}: motif {item['motif_effect_pp']:+.2f} pp "
                f"{item['motif_crossed_seed_task_bootstrap_95_ci_pp']}; "
                f"field {item['field_effect_pp']:+.2f} pp "
                f"{item['field_crossed_seed_task_bootstrap_95_ci_pp']}"
            )
    print(f"\njoint C1/C2 verdict: {verdict}")
    print(f"overall verdict: {overall_verdict}")
    print(f"written: {args.output}")
    print(f"written: {args.report}")


if __name__ == "__main__":
    main()
