"""Build the three figures used in the Russian M1 results deck."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = (
    ("a0.2_b0.8", "Вредный: α=0,2; β=0,8", "#ffb000"),
    ("a0.6_b0.6", "Нейтральный: α=β=0,6", "#aab7c0"),
    ("a1.0_b0.0", "Точный: α=1; β=0", "#53d8ff"),
)


def load_json(path):
    with open(path) as source:
        return json.load(source)


def style_axes(ax):
    ax.set_facecolor("#141a1f")
    ax.tick_params(colors="#cbd4d9", labelsize=11)
    ax.grid(color="#42505c", alpha=0.45, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#42505c")
    ax.xaxis.label.set_color("#cbd4d9")
    ax.yaxis.label.set_color("#cbd4d9")


def save(fig, path):
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="#0d1013")
    plt.close(fig)


def plot_accuracy(summary, out_dir):
    fig, ax = plt.subplots(figsize=(11.2, 6.2), facecolor="#0d1013")
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
    ax.set(xlabel="Шаг обучения", ylabel="Точность на отложенной выборке, %", ylim=(0, 35))
    ax.set_xticks([0, 15, 30, 45, 60])
    legend = ax.legend(loc="lower right", frameon=True, fontsize=10)
    legend.get_frame().set_facecolor("#0d1013")
    legend.get_frame().set_edgecolor("#42505c")
    for text in legend.get_texts():
        text.set_color("#f2efe6")
    fig.tight_layout()
    save(fig, out_dir / "m1_accuracy_curves.png")


def plot_alignment(summary, out_dir):
    fig, ax = plt.subplots(figsize=(13.5, 4.4), facecolor="#0d1013")
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
    ax.plot([p[0] for p in points], [p[1] for p in points], color="#6f7d87", linestyle="--", linewidth=1.2)
    ax.axhline(0, color="#f2efe6", linewidth=1, alpha=0.7)
    ax.set(
        xlabel="Фактическая согласованность чекера: TPR − FPR",
        ylabel="Изменение точности к шагу 60, п.п.",
        xlim=(-0.72, 1.12),
        ylim=(-5, 6),
    )
    fig.tight_layout()
    save(fig, out_dir / "m1_gain_vs_alignment.png")


def plot_seeds(sweep, out_dir):
    fig, ax = plt.subplots(figsize=(10.5, 6.2), facecolor="#0d1013")
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
        color="#d7ff3f",
        marker="D",
        linewidth=3,
        markersize=8,
        label="среднее",
    )
    ax.axhline(0, color="#f2efe6", linewidth=1, alpha=0.7)
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("Изменение точности к шагу 60, п.п.")
    ax.set_ylim(-6, 6)
    legend = ax.legend(loc="upper left", frameon=True, fontsize=10)
    legend.get_frame().set_facecolor("#0d1013")
    legend.get_frame().set_edgecolor("#42505c")
    for text in legend.get_texts():
        text.set_color("#f2efe6")
    fig.tight_layout()
    save(fig, out_dir / "m1_gain_by_seed.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--sweep", required=True)
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


if __name__ == "__main__":
    main()
