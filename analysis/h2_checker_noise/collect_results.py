"""Collect true_accuracy curves from the sweep's TensorBoard logs into one table
and plot the H2 picture: eval true_accuracy vs step per (alpha, beta), plus final
true_accuracy vs signed checker signal (alpha - beta).

  python analysis/h2_checker_noise/collect_results.py
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from math import isclose, sqrt
from statistics import fmean, stdev

# Metric logged by GRPOTrainer for the weight-0 reward function (true correctness).
EVAL_TAG = "eval/rewards/true_accuracy/mean"
TRAIN_TAG = "train/rewards/true_accuracy/mean"
ZERO_VARIANCE_TAG = "train/frac_reward_zero_std"
H2_ALPHAS = (1.0, 0.8, 0.6, 0.4, 0.2)
H2_BETAS = (0.0, 0.2, 0.4, 0.6, 0.8)


def parse_run_name(name):
    # a{alpha}_b{beta}_t{temp}_s{seed}
    out = {}
    for part in name.split("_"):
        if part and part[0] in "abnst" and part[1:].replace(".", "", 1).isdigit():
            out[part[0]] = float(part[1:])
    return out


def parse_config_id(name):
    parts = name.split("_")
    return parts[1][1:] if len(parts) > 1 and parts[1].startswith("c") else "legacy"


def read_scalar(run_dir, tag):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ev_dirs = glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True)
    steps, vals = [], []
    for ev in sorted(ev_dirs):
        acc = EventAccumulator(os.path.dirname(ev))
        acc.Reload()
        if tag in acc.Tags().get("scalars", []):
            for s in acc.Scalars(tag):
                steps.append(s.step)
                vals.append(s.value)
    order = sorted(range(len(steps)), key=lambda i: steps[i])
    return [steps[i] for i in order], [vals[i] for i in order]


def mean_se(values):
    return fmean(values), (stdev(values) / sqrt(len(values)) if len(values) > 1 else 0.0)


def first_hitting_step(steps, values, target):
    return next((step for step, value in zip(steps, values) if value >= target), None)


def summarize_cell(
    alpha,
    beta,
    cell_runs,
    expected_seeds,
    config_id=None,
    plateau_window=5,
    plateau_tolerance=0.02,
):
    """Aggregate one checker-grid cell, or return None until all seeds exist."""
    seeds = {run["seed"] for run in cell_runs}
    if len(seeds) != expected_seeds or len(cell_runs) != expected_seeds:
        return None
    curves = [dict(zip(run["eval_steps"], run["eval_true_acc"])) for run in cell_runs]
    if any(0 not in curve for curve in curves):
        return None
    common_steps = set.intersection(*(set(curve) for curve in curves))
    by_step = {step: [curve[step] for curve in curves] for step in sorted(common_steps)}
    gains = []
    for curve in curves:
        if curve:
            ordered = [curve[step] for step in sorted(curve)]
            gains.append(ordered[-1] - ordered[0])
    steps = sorted(by_step)
    means, ses = zip(*(mean_se(by_step[step]) for step in steps)) if steps else ((), ())
    gain, gain_se = mean_se(gains) if gains else (None, None)
    base_solved_changes = [
        run["base_solved_pass1_change"]
        for run in cell_runs
        if run.get("base_solved_pass1_change") is not None
    ]
    mean_base_solved_change, se_base_solved_change = (
        mean_se(base_solved_changes)
        if len(base_solved_changes) == len(cell_runs)
        else (None, None)
    )
    hit_90 = {
        run["seed"]: first_hitting_step(run["eval_steps"], run["eval_true_acc"], 0.90)
        for run in cell_runs
    }
    hit_95 = {
        run["seed"]: first_hitting_step(run["eval_steps"], run["eval_true_acc"], 0.95)
        for run in cell_runs
    }

    def hit_summary(hit_by_seed):
        reached = [step for step in hit_by_seed.values() if step is not None]
        mean, se = mean_se(reached) if reached else (None, None)
        return mean, se, len(reached)

    mean_t90, se_t90, reached_90 = hit_summary(hit_90)
    mean_t95, se_t95, reached_95 = hit_summary(hit_95)

    tail_changes = {}
    for run in cell_runs:
        tail = run["eval_true_acc"][-plateau_window:]
        tail_changes[run["seed"]] = (
            tail[-1] - tail[0] if len(tail) == plateau_window else None
        )
    plateau_flags = {
        seed: abs(change) <= plateau_tolerance if change is not None else None
        for seed, change in tail_changes.items()
    }
    plateau_all_seeds = (
        all(plateau_flags.values())
        if all(flag is not None for flag in plateau_flags.values())
        else None
    )

    def optional_metric(key):
        values = [run.get(key) for run in cell_runs]
        return mean_se(values) if all(value is not None for value in values) else (None, None)

    mean_tpr, se_tpr = optional_metric("realized_tpr")
    mean_fpr, se_fpr = optional_metric("realized_fpr")
    mean_alignment, se_alignment = optional_metric("realized_signed_alignment")
    mean_mi, se_mi = optional_metric("mutual_information_bits")
    mean_zero_variance, se_zero_variance = optional_metric("zero_variance_group_rate")
    return {
        "config_id": config_id,
        "alpha": alpha,
        "beta": beta,
        "info": alpha - beta,
        "seeds": sorted(seeds),
        "eval_steps": steps,
        "n_eval_seeds": [len(by_step[step]) for step in steps],
        "mean_eval_true_acc": list(means),
        "se_eval_true_acc": list(ses),
        "mean_gain": gain,
        "se_gain": gain_se,
        "mean_base_solved_pass1_change": mean_base_solved_change,
        "se_base_solved_pass1_change": se_base_solved_change,
        "time_to_90_by_seed": hit_90,
        "mean_time_to_90": mean_t90,
        "se_time_to_90": se_t90,
        "seeds_reaching_90": reached_90,
        "time_to_95_by_seed": hit_95,
        "mean_time_to_95": mean_t95,
        "se_time_to_95": se_t95,
        "seeds_reaching_95": reached_95,
        "plateau_window": plateau_window,
        "plateau_tolerance": plateau_tolerance,
        "tail_change_by_seed": tail_changes,
        "plateau_by_seed": plateau_flags,
        "plateau_all_seeds": plateau_all_seeds,
        "mean_realized_tpr": mean_tpr,
        "se_realized_tpr": se_tpr,
        "mean_realized_fpr": mean_fpr,
        "se_realized_fpr": se_fpr,
        "mean_realized_signed_alignment": mean_alignment,
        "se_realized_signed_alignment": se_alignment,
        "mean_mutual_information_bits": mean_mi,
        "se_mutual_information_bits": se_mi,
        "mean_zero_variance_group_rate": mean_zero_variance,
        "se_zero_variance_group_rate": se_zero_variance,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--run-prefix", default="h2_")
    ap.add_argument("--expected-seeds", type=int, default=3)
    ap.add_argument("--min-steps", type=int, default=600)
    ap.add_argument(
        "--allow-short-horizon",
        action="store_true",
        help="accept runs shorter than --min-steps for pilot analysis",
    )
    ap.add_argument(
        "--allow-partial-grid",
        action="store_true",
        help="analyze completed cells without requiring all 25 H2 cells",
    )
    ap.add_argument(
        "--allow-missing-forgetting",
        action="store_true",
        help="accept legacy runs without forgetting.json",
    )
    ap.add_argument(
        "--allow-missing-checker-diagnostics",
        action="store_true",
        help="accept legacy runs without checker_diagnostics.json",
    )
    ap.add_argument(
        "--allow-non-iid-checker",
        action="store_true",
        help="accept the legacy answer-keyed checker (not valid for the primary H2 grid)",
    )
    ap.add_argument("--plateau-window", type=int, default=5)
    ap.add_argument("--plateau-tolerance", type=float, default=0.02)
    ap.add_argument(
        "--require-plateau",
        action="store_true",
        help="require every seed's last window endpoint change to be within tolerance",
    )
    ap.add_argument("--figures-out", default="results/figures")
    ap.add_argument("--data-out", default="results/data")
    args = ap.parse_args()
    if args.expected_seeds <= 0:
        ap.error("--expected-seeds must be positive")
    if args.min_steps <= 0:
        ap.error("--min-steps must be positive")
    if args.plateau_window < 2 or args.plateau_tolerance < 0:
        ap.error("--plateau-window must be >=2 and --plateau-tolerance non-negative")
    os.makedirs(args.figures_out, exist_ok=True)
    os.makedirs(args.data_out, exist_ok=True)

    runs = {}
    for run_dir in sorted(glob.glob(os.path.join(args.runs_dir, f"{args.run_prefix}*"))):
        if not os.path.isdir(run_dir):
            continue
        name = os.path.basename(run_dir)
        meta = parse_run_name(name)
        if "a" not in meta or "b" not in meta:
            continue  # skip smoke/unlabeled runs
        est, evv = read_scalar(run_dir, EVAL_TAG)
        tst, tvv = read_scalar(run_dir, TRAIN_TAG)
        zst, zvv = read_scalar(run_dir, ZERO_VARIANCE_TAG)
        if not evv and not tvv:
            continue
        target_steps = int(meta.get("n", 0))
        required_steps = target_steps
        if not args.allow_short_horizon:
            required_steps = max(required_steps, args.min_steps)
        if not est or max(est) < required_steps:
            print(
                f"skip short/incomplete run: {name} "
                f"(need step {required_steps}, found {max(est) if est else 'none'})",
                file=sys.stderr,
            )
            continue
        forgetting_path = os.path.join(run_dir, "forgetting.json")
        forgetting = None
        if os.path.exists(forgetting_path):
            with open(forgetting_path) as f:
                forgetting = json.load(f)
        elif not args.allow_missing_forgetting:
            print(f"skip run without forgetting metrics: {name}", file=sys.stderr)
            continue
        base_solved_change = (
            forgetting.get("pass@1_change_on_base_solved")
            if isinstance(forgetting, dict)
            else None
        )
        if base_solved_change is None and not args.allow_missing_forgetting:
            print(f"skip run with undefined base-solved change: {name}", file=sys.stderr)
            continue
        forgetting_data = forgetting if isinstance(forgetting, dict) else {}
        checker_path = os.path.join(run_dir, "checker_diagnostics.json")
        checker = None
        if os.path.exists(checker_path):
            with open(checker_path) as f:
                checker = json.load(f)
        elif not args.allow_missing_checker_diagnostics:
            print(f"skip run without checker diagnostics: {name}", file=sys.stderr)
            continue
        checker_data = checker if isinstance(checker, dict) else {}
        if checker_data:
            configured_alpha = checker_data.get("configured_alpha")
            configured_beta = checker_data.get("configured_beta")
            matches_name = (
                configured_alpha is not None
                and configured_beta is not None
                and isclose(configured_alpha, meta["a"])
                and isclose(configured_beta, meta["b"])
            )
            if not matches_name:
                print(f"skip checker/config mismatch: {name}", file=sys.stderr)
                continue
            if checker_data.get("noise_mode") != "iid" and not args.allow_non_iid_checker:
                print(f"skip non-iid checker run: {name}", file=sys.stderr)
                continue
        runs[name] = {
            "config_id": parse_config_id(name),
            "alpha": meta.get("a"),
            "beta": meta.get("b"),
            "seed": int(meta.get("s", 0)),
            "target_steps": target_steps,
            "info": (meta.get("a", 0) - meta.get("b", 0)),
            "eval_steps": est,
            "eval_true_acc": evv,
            "train_steps": tst,
            "train_true_acc": tvv,
            "final_eval_true_acc": evv[-1] if evv else None,
            "first_eval_true_acc": evv[0] if evv else None,
            "base_solved_pass1_change": base_solved_change,
            "base_solved_nonrediscovery_rate": (
                forgetting_data.get("base_solved_nonrediscovery_rate")
            ),
            "base_solved_count": forgetting_data.get("base_solved_count"),
            "realized_tpr": checker_data.get("realized_tpr"),
            "realized_fpr": checker_data.get("realized_fpr"),
            "realized_signed_alignment": checker_data.get("realized_signed_alignment"),
            "mutual_information_bits": checker_data.get("mutual_information_bits"),
            "zero_variance_steps": zst,
            "zero_variance_group_rate": (
                fmean(zvv) if zvv else checker_data.get("zero_variance_group_rate")
            ),
        }

    config_ids = sorted({run["config_id"] for run in runs.values()})
    if len(config_ids) > 1:
        choices = ", ".join(f"h2_c{config_id}_" for config_id in config_ids)
        raise SystemExit(f"multiple H2 configs found; rerun with --run-prefix set to one of: {choices}")

    with open(os.path.join(args.data_out, "checker_sweep.json"), "w") as f:
        json.dump(runs, f, indent=2)

    grouped = defaultdict(list)
    for run in runs.values():
        grouped[(run["alpha"], run["beta"])].append(run)

    summary = {}
    for (alpha, beta), cell_runs in sorted(grouped.items(), reverse=True):
        seeds = {run["seed"] for run in cell_runs}
        cell = summarize_cell(
            alpha,
            beta,
            cell_runs,
            args.expected_seeds,
            config_ids[0] if config_ids else None,
            args.plateau_window,
            args.plateau_tolerance,
        )
        if cell is None:
            print(
                f"skip incomplete cell a={alpha}, b={beta}: "
                f"expected {args.expected_seeds} seeds, found {sorted(seeds)}",
                file=sys.stderr,
            )
            continue
        if args.require_plateau and cell["plateau_all_seeds"] is not True:
            print(
                f"skip non-plateau cell a={alpha}, b={beta}: "
                f"tail changes {cell['tail_change_by_seed']}",
                file=sys.stderr,
            )
            continue
        key = f"a{alpha}_b{beta}"
        summary[key] = cell

    expected_cells = {
        f"a{alpha}_b{beta}" for alpha in H2_ALPHAS for beta in H2_BETAS
    }
    missing_cells = sorted(expected_cells - summary.keys())
    if missing_cells and not args.allow_partial_grid:
        raise SystemExit(
            "incomplete H2 grid; missing completed three-seed cells: "
            + ", ".join(missing_cells)
            + "; pass --allow-partial-grid only for pilot analysis"
        )

    with open(os.path.join(args.data_out, "checker_grid_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"{'cell':18s} {'seeds':>5s} {'a-b':>5s} "
        f"{'gain':>10s} {'d_base':>8s} {'t@90[n]':>12s} {'t@95[n]':>12s} {'flat':>5s}"
    )
    for name, cell in sorted(summary.items(), key=lambda item: -item[1]["info"]):
        gain = f"{cell['mean_gain']:.3f}" if cell["mean_gain"] is not None else "nan"
        base_solved_change = (
            f"{cell['mean_base_solved_pass1_change']:.3f}"
            if cell["mean_base_solved_pass1_change"] is not None
            else "nan"
        )
        total_seeds = len(cell["seeds"])
        time_90 = (
            f"{cell['mean_time_to_90']:g}[{cell['seeds_reaching_90']}/{total_seeds}]"
            if cell["mean_time_to_90"] is not None
            else f"NA[0/{total_seeds}]"
        )
        time_95 = (
            f"{cell['mean_time_to_95']:g}[{cell['seeds_reaching_95']}/{total_seeds}]"
            if cell["mean_time_to_95"] is not None
            else f"NA[0/{total_seeds}]"
        )
        print(
            f"{name:18s} {len(cell['seeds']):>5d} {cell['info']:>5.2f} "
            f"{gain:>10s} {base_solved_change:>8s} "
            f"{time_90:>12s} {time_95:>12s} "
            f"{str(cell['plateau_all_seeds']):>5s}"
        )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        for cell in sorted(summary.values(), key=lambda value: -value["info"]):
            if cell["mean_eval_true_acc"]:
                ax1.plot(
                    cell["eval_steps"],
                    cell["mean_eval_true_acc"],
                    marker="o",
                    label=f"α={cell['alpha']}, β={cell['beta']}",
                )
        ax1.set_xlabel("training step")
        ax1.set_ylabel("eval true_accuracy")
        ax1.set_title("H2: self-improvement vs checker (α,β)")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        pts = [
            (cell["info"], cell["mean_gain"], cell["se_gain"])
            for cell in summary.values()
            if cell["mean_gain"] is not None
        ]
        pts.sort()
        if pts:
            ax2.errorbar(
                [p[0] for p in pts],
                [p[1] for p in pts],
                yerr=[p[2] for p in pts],
                fmt="s",
            )
            ax2.axhline(0, color="k", lw=0.8)
            ax2.axvline(0, color="r", ls="--", lw=0.8, label="break-even α=β")
            ax2.set_xlabel("signed checker signal (α − β)")
            ax2.set_ylabel("Δ true_accuracy (final − first)")
            ax2.set_title("H2: mean gain ± SE vs signed checker signal")
            ax2.legend(fontsize=8)
            ax2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.figures_out, "checker_phase_overview.png"), dpi=130)
        print("wrote", os.path.join(args.figures_out, "checker_phase_overview.png"))
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
