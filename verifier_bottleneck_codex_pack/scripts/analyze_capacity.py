"""D-021 analysis: capacity ceiling, and depth separated from prime transfer.

Computes the four contrasts fixed in
`artifacts/preregistration_capacity_2026-08-02.md`:

  C1 primary   oracle - atomic on capacity_d3_heldout   (does supervision transfer)
  C2 secondary oracle - atomic on capacity_d3_train     (is it learnable at all)
  C3 secondary d2_train - d3_train, oracle              (depth effect, primes fixed)
  C4 secondary d3_train - d3_heldout, oracle            (prime effect, depth fixed)

C1 and C2 compare two adapters on the same tasks, so they are paired. C3 and C4
compare different task sets, so they use an unpaired bootstrap over tasks within
each seed. Mixing the two would understate the interval on C3/C4.

Holm across {C2, C3, C4}; C1 uncorrected. The preregistered joint reading of
C1 and C2 is printed verbatim at the end.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import binomtest

try:
    from scripts.analyze_ranking import _bootstrap_p, _ci_pp, _crossed_draws, holm
except ModuleNotFoundError:  # direct `python scripts/analyze_capacity.py`
    from analyze_ranking import _bootstrap_p, _ci_pp, _crossed_draws, holm

PRIMARY_ARM = "iid_action@0.7"
ARMS = ["iid_action@0.7", "iid_action@3.0", "iid_action@8.0"]
SETS = ["capacity_d2_train", "capacity_d3_train", "capacity_d2_heldout", "capacity_d3_heldout"]
EXPECTED_TASKS = 300


def role_of(adapter_name: str) -> str | None:
    if adapter_name == "sft_atomic_r32_pilot_aw4_cont":
        return "atomic"
    if adapter_name.startswith("sft_composition_oracle_seed"):
        return "oracle"
    return None


def discover(runs_root: Path):
    """(dataset, role, seed) -> run directory."""
    found: dict[tuple[str, str, int], Path] = {}
    for run in sorted(runs_root.iterdir()):
        config_path = run / "config.resolved.yaml"
        if not (run / "DONE").exists() or not config_path.exists():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or "methods" not in config or "adapter" not in config:
            continue
        dataset = Path(str(config["input"]).replace("\\", "/")).stem
        if dataset not in SETS:
            continue
        role = role_of(Path(str(config["adapter"]).replace("\\", "/")).name)
        if role:
            found[(dataset, role, int(config["seed"]))] = run
    return found


def outcomes(run: Path, method: str):
    """task_id -> (solved_within_32, solved_by_first_candidate)."""
    ordered: dict[str, list[bool]] = defaultdict(list)
    for line in (run / "generations.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["sampling"]["method"] == method:
            ordered[record["task_id"]].append(bool(record["is_correct"]))
    if not ordered:
        raise KeyError(f"method {method!r} absent from {run.name}")
    return {task_id: (any(values), values[0]) for task_id, values in ordered.items()}


def paired_contrast(pairs, rng, bootstrap):
    """pairs: (seed, run_a, run_b, method). Reports b - a on shared tasks."""
    per_seed, diffs = [], []
    for seed, run_a, run_b, method in pairs:
        a, b = outcomes(run_a, method), outcomes(run_b, method)
        if set(a) != set(b) or len(a) != EXPECTED_TASKS:
            raise ValueError(f"paired task mismatch: {run_a.name} vs {run_b.name}")
        ids = sorted(a)
        av = np.array([int(a[t][0]) for t in ids])
        bv = np.array([int(b[t][0]) for t in ids])
        d = bv - av
        samples = rng.choice(d, size=(bootstrap, len(d)), replace=True).mean(axis=1)
        b_only = int(np.count_nonzero(d == 1))
        a_only = int(np.count_nonzero(d == -1))
        discordant = b_only + a_only
        per_seed.append({"seed": seed, "tasks": len(ids), "a_pass": float(av.mean()),
                         "b_pass": float(bv.mean()), "delta_pp": 100.0 * float(d.mean()),
                         "task_bootstrap_95_ci_pp": _ci_pp(samples),
                         "b_only": b_only, "a_only": a_only,
                         "mcnemar_exact_two_sided_p": float(binomtest(b_only, discordant, 0.5).pvalue)
                         if discordant else 1.0})
        diffs.append(d)
    matrix = np.stack(diffs)
    draws = _crossed_draws(matrix, rng, bootstrap)
    return _finish(per_seed, draws, float(matrix.mean()), paired=True)


def unpaired_contrast(pairs, rng, bootstrap):
    """pairs: (seed, run_a, run_b, method) on different task sets. Reports b - a."""
    per_seed, rows_a, rows_b = [], [], []
    ids_a_expected = ids_b_expected = None
    for seed, run_a, run_b, method in pairs:
        a, b = outcomes(run_a, method), outcomes(run_b, method)
        ids_a, ids_b = sorted(a), sorted(b)
        if len(ids_a) != EXPECTED_TASKS or len(ids_b) != EXPECTED_TASKS:
            raise ValueError(f"expected {EXPECTED_TASKS} tasks: {run_a.name}, {run_b.name}")
        if ids_a_expected is None:
            ids_a_expected, ids_b_expected = ids_a, ids_b
        elif ids_a != ids_a_expected or ids_b != ids_b_expected:
            raise ValueError("task IDs differ across seeds in unpaired contrast")
        av = np.array([int(a[t][0]) for t in ids_a])
        bv = np.array([int(b[t][0]) for t in ids_b])
        samples = (rng.choice(bv, size=(bootstrap, len(bv)), replace=True).mean(axis=1)
                   - rng.choice(av, size=(bootstrap, len(av)), replace=True).mean(axis=1))
        per_seed.append({"seed": seed, "tasks_a": len(av), "tasks_b": len(bv),
                         "a_pass": float(av.mean()), "b_pass": float(bv.mean()),
                         "delta_pp": 100.0 * float(bv.mean() - av.mean()),
                         "task_bootstrap_95_ci_pp": _ci_pp(samples)})
        rows_a.append(av)
        rows_b.append(bv)
    matrix_a, matrix_b = np.stack(rows_a), np.stack(rows_b)
    combined = np.empty(bootstrap)
    for i in range(bootstrap):
        seeds = rng.integers(0, len(per_seed), len(per_seed))
        tasks_a = rng.integers(0, matrix_a.shape[1], matrix_a.shape[1])
        tasks_b = rng.integers(0, matrix_b.shape[1], matrix_b.shape[1])
        combined[i] = (matrix_b[np.ix_(seeds, tasks_b)].mean()
                       - matrix_a[np.ix_(seeds, tasks_a)].mean())
    return _finish(per_seed, combined, float(matrix_b.mean() - matrix_a.mean()), paired=False)


def _finish(per_seed, combined, point, paired):
    ci = _ci_pp(combined)
    deltas = [item["delta_pp"] for item in per_seed]
    result = {
        "paired": paired,
        "bootstrap_unit": "crossed_training_seed_x_task" if paired else "crossed_training_seed_x_task_sets",
        "per_seed": per_seed,
        "mean_a_pass": float(np.mean([i["a_pass"] for i in per_seed])),
        "mean_b_pass": float(np.mean([i["b_pass"] for i in per_seed])),
        "mean_delta_pp": 100.0 * point,
        "crossed_seed_task_bootstrap_95_ci_pp": ci,
        "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
        "positive_direction_seeds": int(sum(v > 0 for v in deltas)),
        "negative_direction_seeds": int(sum(v < 0 for v in deltas)),
        "seed_count": len(per_seed),
        "bootstrap_two_sided_p": _bootstrap_p(combined, point),
    }
    return result


def render_report(payload: dict) -> str:
    names = {
        "C1_primary_supervision_transfers_to_heldout_primes": "C1: transfer to held-out fields",
        "C2_compositions_learnable_in_domain": "C2: learnability in-domain",
        "C3_depth_effect_primes_fixed": "C3: depth-2 minus depth-3",
        "C4_prime_effect_depth_fixed": "C4: train minus held-out fields",
        "C1_secondary_at_hot_arm": "C1 secondary at iid@8.0",
    }
    rows = []
    for key, label in names.items():
        item = payload["contrasts"][key]
        low, high = item["crossed_seed_task_bootstrap_95_ci_pp"]
        p = item.get("bootstrap_p_holm_adjusted", item["bootstrap_two_sided_p"])
        rows.append(f"| {label} | {item['mean_delta_pp']:+.2f} | [{low:+.2f}, {high:+.2f}] | {p:.4g} |")
    return "\n".join([
        "# D-021 sampled-grid report",
        "",
        "These preregistered sampled `pass@32` endpoints are retained for continuity. Under",
        "Amendment 001 they are secondary to exhaustive ranking because duplicate samples make",
        "`pass@32` depend on sampler diversity as well as policy preference.",
        "",
        "| Contrast | Delta (pp) | 95% CI (pp) | p / Holm p |",
        "|---|---:|---:|---:|",
        *rows,
        "",
        f"**Registered decision:** {payload['preregistered_verdict']}",
        "",
        "This decision concerns the sampled endpoint only. The retention rule and final claim",
        "boundary are reported in `RANKING_FINAL_REPORT.md`.",
        "",
        "## Measured evaluation cost",
        "",
        f"The 24 cells used {payload['compute']['model_passes']:,} model passes, scored",
        f"{payload['compute']['scored_operation_tokens']:,} operation tokens and took",
        f"{payload['compute']['elapsed_seconds'] / 3600:.2f} summed GPU-hours. Peak allocated CUDA",
        f"memory was {payload['compute']['peak_cuda_bytes'] / 2**30:.2f} GiB.",
        "",
        "## Reproduction",
        "",
        "```bash",
        ".venv/bin/python scripts/analyze_capacity.py",
        "```",
        "",
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports/capacity_statistics.json"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/reports/CAPACITY_SAMPLED_REPORT.md"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--rng-seed", type=int, default=20260803)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    found = discover(args.runs)

    def run_for(dataset, role, seed):
        if (dataset, role, seed) not in found:
            raise SystemExit(f"missing run for {(dataset, role, seed)}; run scripts/run_capacity_series.sh")
        return found[(dataset, role, seed)]

    rng = np.random.default_rng(args.rng_seed)
    results = {}
    results["C1_primary_supervision_transfers_to_heldout_primes"] = paired_contrast(
        [(s, run_for("capacity_d3_heldout", "atomic", s), run_for("capacity_d3_heldout", "oracle", s),
          PRIMARY_ARM) for s in args.seeds], rng, args.bootstrap)
    results["C2_compositions_learnable_in_domain"] = paired_contrast(
        [(s, run_for("capacity_d3_train", "atomic", s), run_for("capacity_d3_train", "oracle", s),
          PRIMARY_ARM) for s in args.seeds], rng, args.bootstrap)
    results["C3_depth_effect_primes_fixed"] = unpaired_contrast(
        [(s, run_for("capacity_d3_train", "oracle", s), run_for("capacity_d2_train", "oracle", s),
          PRIMARY_ARM) for s in args.seeds], rng, args.bootstrap)
    results["C4_prime_effect_depth_fixed"] = unpaired_contrast(
        [(s, run_for("capacity_d3_heldout", "oracle", s), run_for("capacity_d3_train", "oracle", s),
          PRIMARY_ARM) for s in args.seeds], rng, args.bootstrap)
    results["C1_secondary_at_hot_arm"] = paired_contrast(
        [(s, run_for("capacity_d3_heldout", "atomic", s), run_for("capacity_d3_heldout", "oracle", s),
          "iid_action@8.0") for s in args.seeds], rng, args.bootstrap)

    secondary = {name: results[name]["bootstrap_two_sided_p"] for name in
                 ("C2_compositions_learnable_in_domain", "C3_depth_effect_primes_fixed",
                  "C4_prime_effect_depth_fixed")}
    for name, adjusted in holm(secondary).items():
        results[name]["bootstrap_p_holm_adjusted"] = adjusted

    table = {}
    for dataset in SETS:
        for role in ("atomic", "oracle"):
            for arm in ARMS:
                rows, first = [], []
                for seed in args.seeds:
                    run = run_for(dataset, role, seed)
                    rows.append(json.loads((run / "final_metrics.json").read_text())["methods"][arm])
                    values = outcomes(run, arm)
                    first.append(np.mean([int(v[1]) for v in values.values()]))
                table[f"{dataset}|{role}|{arm}"] = {
                    "pass_at_32": float(np.mean([r["pass_at_32"] for r in rows])),
                    "pass_at_1": float(np.mean(first)),
                    "distinct_programs_per_task": float(np.mean([r["distinct_programs_per_task"] for r in rows])),
                    "all_solution_coverage": float(np.mean([r["all_solution_coverage"] for r in rows])),
                }

    c1, c2 = results["C1_primary_supervision_transfers_to_heldout_primes"], results["C2_compositions_learnable_in_domain"]
    if not c2["ci_excludes_zero"] or c2["mean_delta_pp"] <= 0:
        verdict = ("CAPACITY CEILING - the model cannot learn depth-3 composition even from direct "
                   "supervision; every earlier null is a statement about the model, not about "
                   "exploration or self-training.")
    elif not c1["ci_excludes_zero"] or c1["mean_delta_pp"] <= 0:
        verdict = ("GENERALISATION CEILING - compositions are learnable in-domain but do not transfer "
                   "across primes; the earlier nulls are about transfer, and the sandbox needs a "
                   "harder-to-memorise split before any exploration claim is meaningful.")
    else:
        verdict = ("METHOD CEILING - capacity and transfer are both fine, so the earlier nulls belong "
                   "to search and policy-gradient training; first-step credit assignment becomes "
                   "worth testing.")

    selected_runs = {
        run_for(dataset, role, seed)
        for dataset in SETS for role in ("atomic", "oracle") for seed in args.seeds
    }
    run_metrics = [
        json.loads((run / "final_metrics.json").read_text(encoding="utf-8"))
        for run in selected_runs
    ]
    compute = {
        "model_passes": int(sum(sum(method["model_passes"] for method in item["methods"].values())
                                for item in run_metrics)),
        "scored_operation_tokens": int(sum(
            sum(method["scored_operation_tokens"] for method in item["methods"].values())
            for item in run_metrics
        )),
        "elapsed_seconds": float(sum(item["elapsed_seconds"] for item in run_metrics)),
        "peak_cuda_bytes": int(max(item["peak_cuda_bytes"] for item in run_metrics)),
    }
    payload = {"preregistration": "artifacts/preregistration_capacity_2026-08-02.md",
               "primary_arm": PRIMARY_ARM, "bootstrap_repetitions": args.bootstrap,
               "compute": compute, "arm_table": table, "contrasts": results,
               "preregistered_verdict": verdict}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(payload), encoding="utf-8")

    print(f"primary arm: {PRIMARY_ARM}   (mean of {len(args.seeds)} seeds, 300 tasks per set)\n")
    print(f"{'set':22s} {'adapter':8s} {'arm':16s} {'pass@1':>7s} {'pass@32':>8s} {'distinct':>9s} {'cover':>7s}")
    for dataset in SETS:
        for role in ("atomic", "oracle"):
            for arm in ARMS:
                row = table[f"{dataset}|{role}|{arm}"]
                print(f"{dataset:22s} {role:8s} {arm:16s} {row['pass_at_1']:7.3f} {row['pass_at_32']:8.3f} "
                      f"{row['distinct_programs_per_task']:9.2f} {row['all_solution_coverage']:7.3f}")
        print()
    for name, item in results.items():
        low, high = item["crossed_seed_task_bootstrap_95_ci_pp"]
        extra = f"  bootstrap_p={item['bootstrap_two_sided_p']:.3g}"
        if "bootstrap_p_holm_adjusted" in item:
            extra += f"  holm_p={item['bootstrap_p_holm_adjusted']:.3g}"
        print(f"{name}  [{'paired' if item['paired'] else 'unpaired'}]\n"
              f"  delta = {item['mean_delta_pp']:+.2f} pp  CI95 [{low:+.2f}, {high:+.2f}]  "
              f"excludes_zero={item['ci_excludes_zero']}  dir {item['positive_direction_seeds']}/"
              f"{item['seed_count']}{extra}")
    print(f"\nPREREGISTERED VERDICT: {verdict}")
    print(f"written: {args.output}")
    print(f"written: {args.report}")


if __name__ == "__main__":
    main()
