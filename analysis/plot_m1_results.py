"""Build the figures used in the M1 results deck."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = (
    ("a0.2_b0.8", "Misleading: α=0.2, β=0.8", "#ef6c27"),
    ("a0.6_b0.6", "Uninformative: α=β=0.6", "#626a78"),
    ("a1.0_b0.0", "Reliable: α=1, β=0", "#2468f2"),
)


def load_json(path):
    with open(path) as source:
        return json.load(source)


def style_axes(ax):
    ax.set_facecolor("#ffffff")
    ax.tick_params(colors="#626a78", labelsize=11)
    ax.grid(color="#d7dce4", alpha=0.75, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#c9cfd8")
    ax.xaxis.label.set_color("#17191d")
    ax.yaxis.label.set_color("#17191d")


def save(fig, path):
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)


def plot_accuracy(summary, out_dir):
    fig, ax = plt.subplots(figsize=(11.2, 6.2), facecolor="#ffffff")
    style_axes(ax)
    for key, label, color in CONDITIONS:
        cell = summary[key]
        steps = np.asarray(cell["eval_steps"])
        mean = 100 * np.asarray(cell["mean_eval_true_acc"])
        se = 100 * np.asarray(cell["se_eval_true_acc"])
        ax.plot(
            steps,
            mean,
            marker="o",
            linewidth=2.6,
            markersize=7,
            color=color,
            label=label,
        )
        ax.fill_between(steps, mean - se, mean + se, color=color, alpha=0.14)
    ax.set(xlabel="Training step", ylabel="Held-out accuracy, %", ylim=(0, 35))
    ax.set_xticks([0, 15, 30, 45, 60])
    legend = ax.legend(loc="lower right", frameon=True, fontsize=10)
    legend.get_frame().set_facecolor("#ffffff")
    legend.get_frame().set_edgecolor("#c9cfd8")
    for text in legend.get_texts():
        text.set_color("#17191d")
    fig.tight_layout()
    save(fig, out_dir / "m1_accuracy_curves.png")


def plot_alignment(summary, out_dir):
    fig, ax = plt.subplots(figsize=(13.5, 4.4), facecolor="#ffffff")
    style_axes(ax)
    points = []
    for key, label, color in CONDITIONS:
        cell = summary[key]
        x = cell["mean_realized_signed_alignment"]
        y = 100 * cell["mean_gain"]
        yerr = 100 * cell["se_gain"]
        points.append((x, y))
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            color=color,
            markersize=10,
            capsize=5,
            linewidth=2,
        )
        ax.annotate(
            label.split(":", 1)[0],
            (x, y),
            xytext=(7, 9),
            textcoords="offset points",
            color=color,
            fontsize=11,
        )
    points.sort()
    ax.plot([p[0] for p in points], [p[1] for p in points], color="#8a929f", linestyle="--", linewidth=1.2)
    ax.axhline(0, color="#17191d", linewidth=1, alpha=0.7)
    ax.set(
        xlabel="Realized verifier alignment: TPR − FPR",
        ylabel="Accuracy change at step 60, pp",
        xlim=(-0.72, 1.12),
        ylim=(-5, 6),
    )
    fig.tight_layout()
    save(fig, out_dir / "m1_gain_vs_alignment.png")


def plot_seeds(sweep, out_dir):
    fig, ax = plt.subplots(figsize=(10.5, 6.2), facecolor="#ffffff")
    style_axes(ax)
    keys = [key for key, _, _ in CONDITIONS]
    labels = [label.split(":", 1)[0] for _, label, _ in CONDITIONS]
    seeds = sorted({run["seed"] for run in sweep.values()})
    for seed in seeds:
        gains = []
        for key in keys:
            run = next(
                run
                for run in sweep.values()
                if run["seed"] == seed and f"a{run['alpha']}_b{run['beta']}" == key
            )
            gains.append(100 * (run["final_eval_true_acc"] - run["first_eval_true_acc"]))
        ax.plot(range(3), gains, marker="o", linewidth=1.6, markersize=6, alpha=0.8, label=f"seed {seed}")
    means = []
    for key in keys:
        gains = [
            run["final_eval_true_acc"] - run["first_eval_true_acc"]
            for run in sweep.values()
            if f"a{run['alpha']}_b{run['beta']}" == key
        ]
        means.append(100 * np.mean(gains))
    ax.plot(
        range(3),
        means,
        color="#2468f2",
        marker="D",
        linewidth=3,
        markersize=8,
        label="mean",
    )
    ax.axhline(0, color="#17191d", linewidth=1, alpha=0.7)
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("Accuracy change at step 60, pp")
    ax.set_ylim(-6, 6)
    legend = ax.legend(loc="upper left", frameon=True, fontsize=10)
    legend.get_frame().set_facecolor("#ffffff")
    legend.get_frame().set_edgecolor("#c9cfd8")
    for text in legend.get_texts():
        text.set_color("#17191d")
    fig.tight_layout()
    save(fig, out_dir / "m1_gain_by_seed.png")


def plot_exploration(exploration, out_dir):
    conditions = exploration["conditions"]
    resolutions = sorted({condition["resolution_m"] for condition in conditions})
    colors = dict(zip(resolutions, ("#2468f2", "#ef6c27", "#1c9b57")))
    reference_temperature = exploration["config"]["reference_temperature"]
    baseline = next(
        condition["mean"]["pass@1"]
        for condition in conditions
        if condition["kind"] == "temperature"
        and condition["temperature"] == reference_temperature
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), sharey=True, facecolor="#ffffff")
    for ax, kind, title, parameter in (
        (axes[0], "temperature", "Temperature", "temperature"),
        (axes[1], "mixture", "Policy mixture", "requested_weight"),
    ):
        style_axes(ax)
        for resolution in resolutions:
            group = sorted(
                (
                    condition
                    for condition in conditions
                    if condition["kind"] == kind
                    and condition["resolution_m"] == resolution
                ),
                key=lambda condition: condition[parameter],
            )
            x = 100 * np.asarray(
                [condition["mean"]["delta_epsilon_parseable"] for condition in group]
            )
            y = 100 * np.asarray([condition["mean"]["pass@1"] for condition in group])
            xerr = 100 * np.asarray(
                [condition["mean"]["se_delta_epsilon_parseable"] for condition in group]
            )
            yerr = 100 * np.asarray([condition["mean"]["se_pass@1"] for condition in group])
            ax.errorbar(
                x,
                y,
                xerr=xerr,
                yerr=yerr,
                color=colors[resolution],
                marker="o" if kind == "temperature" else "s",
                markersize=6,
                capsize=3,
                linewidth=1.8,
                label=f"m={resolution}",
            )
            if resolution == max(resolutions):
                label_indices = {0, len(group) - 1}
                if kind == "temperature":
                    label_indices.add(
                        next(
                            index
                            for index, condition in enumerate(group)
                            if condition[parameter] == reference_temperature
                        )
                    )
                symbol = "T" if kind == "temperature" else "λ"
                for index in label_indices:
                    ax.annotate(
                        f"{symbol}={group[index][parameter]:g}",
                        (x[index], y[index]),
                        xytext=(5, 12),
                        textcoords="offset points",
                        color="#626a78",
                        fontsize=9,
                    )
        ax.axvline(0, color="#626a78", linestyle="--", linewidth=1.1)
        ax.axhline(100 * baseline, color="#626a78", linestyle=":", linewidth=1.1)
        ax.set_title(title, loc="left", color="#17191d", fontsize=15, fontweight="bold")
        ax.set_xlabel(r"$\Delta\widehat{\varepsilon}_{parse}^{1/m}$, pp")
        ax.set_xlim(-31, 3)
        ax.set_ylim(-1, 27)
    axes[0].set_ylabel("pass@1, %")
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=True)
    legend.get_frame().set_facecolor("#ffffff")
    legend.get_frame().set_edgecolor("#c9cfd8")
    for text in legend.get_texts():
        text.set_color("#17191d")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save(fig, out_dir / "m1_epsilon_frontier.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--exploration")
    parser.add_argument("--out", default="presentation/assets")
    args = parser.parse_args()

    summary = load_json(args.summary)
    sweep = load_json(args.sweep)
    missing = {key for key, _, _ in CONDITIONS} - summary.keys()
    if missing:
        raise SystemExit(f"missing M1 cells: {sorted(missing)}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_accuracy(summary, out_dir)
    plot_alignment(summary, out_dir)
    plot_seeds(sweep, out_dir)
    if args.exploration:
        plot_exploration(load_json(args.exploration), out_dir)


if __name__ == "__main__":
    main()
