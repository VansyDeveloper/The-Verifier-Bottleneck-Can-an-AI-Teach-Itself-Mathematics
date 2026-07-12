"""Collect true_accuracy curves from the sweep's TensorBoard logs into one table
and plot the H2 picture: eval true_accuracy vs step per (alpha, beta), plus final
true_accuracy vs checker informativeness (alpha - beta).

  python analysis/h2_checker_noise/collect_results.py
"""

import argparse
import glob
import json
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Metric logged by GRPOTrainer for the weight-0 reward function (true correctness).
EVAL_TAG = "eval/rewards/true_accuracy/mean"
TRAIN_TAG = "train/rewards/true_accuracy/mean"


def parse_run_name(name):
    # a{alpha}_b{beta}_t{temp}_s{seed}
    out = {}
    for part in name.split("_"):
        if part and part[0] in "abts" and part[1:].replace(".", "", 1).isdigit():
            out[part[0]] = float(part[1:])
    return out


def read_scalar(run_dir, tag):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--figures-out", default="results/figures")
    ap.add_argument("--data-out", default="results/data")
    args = ap.parse_args()
    os.makedirs(args.figures_out, exist_ok=True)
    os.makedirs(args.data_out, exist_ok=True)

    runs = {}
    for run_dir in sorted(glob.glob(os.path.join(args.runs_dir, "*"))):
        if not os.path.isdir(run_dir):
            continue
        name = os.path.basename(run_dir)
        meta = parse_run_name(name)
        if "a" not in meta or "b" not in meta:
            continue  # skip smoke/unlabeled runs
        est, evv = read_scalar(run_dir, EVAL_TAG)
        tst, tvv = read_scalar(run_dir, TRAIN_TAG)
        if not evv and not tvv:
            continue
        runs[name] = {
            "alpha": meta.get("a"),
            "beta": meta.get("b"),
            "info": (meta.get("a", 0) - meta.get("b", 0)),
            "eval_steps": est,
            "eval_true_acc": evv,
            "train_steps": tst,
            "train_true_acc": tvv,
            "final_eval_true_acc": evv[-1] if evv else None,
            "first_eval_true_acc": evv[0] if evv else None,
        }

    with open(os.path.join(args.data_out, "checker_sweep.json"), "w") as f:
        json.dump(runs, f, indent=2)

    print(
        f"{'run':28s} {'alpha':>5s} {'beta':>5s} {'a-b':>5s} {'first_eval':>10s} {'final_eval':>10s}"
    )
    for name, r in sorted(runs.items(), key=lambda kv: -(kv[1]["info"])):
        fe = r["first_eval_true_acc"]
        ff = r["final_eval_true_acc"]
        print(
            f"{name:28s} {r['alpha']:>5.2f} {r['beta']:>5.2f} {r['info']:>5.2f} "
            f"{(fe if fe is not None else float('nan')):>10.3f} "
            f"{(ff if ff is not None else float('nan')):>10.3f}"
        )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        for name, r in sorted(runs.items(), key=lambda kv: -(kv[1]["info"])):
            if r["eval_true_acc"]:
                ax1.plot(
                    r["eval_steps"],
                    r["eval_true_acc"],
                    marker="o",
                    label=f"α={r['alpha']}, β={r['beta']}",
                )
        ax1.set_xlabel("training step")
        ax1.set_ylabel("eval true_accuracy")
        ax1.set_title("H2: self-improvement vs checker (α,β)")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        pts = [
            (r["info"], r["final_eval_true_acc"] - r["first_eval_true_acc"])
            for r in runs.values()
            if r["final_eval_true_acc"] is not None and r["first_eval_true_acc"] is not None
        ]
        pts.sort()
        if pts:
            ax2.plot([p[0] for p in pts], [p[1] for p in pts], marker="s")
            ax2.axhline(0, color="k", lw=0.8)
            ax2.axvline(0, color="r", ls="--", lw=0.8, label="break-even α=β")
            ax2.set_xlabel("checker informativeness (α − β)")
            ax2.set_ylabel("Δ true_accuracy (final − first)")
            ax2.set_title("H2: gain vs checker informativeness")
            ax2.legend(fontsize=8)
            ax2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.figures_out, "checker_phase_overview.png"), dpi=130)
        print("wrote", os.path.join(args.figures_out, "checker_phase_overview.png"))
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
