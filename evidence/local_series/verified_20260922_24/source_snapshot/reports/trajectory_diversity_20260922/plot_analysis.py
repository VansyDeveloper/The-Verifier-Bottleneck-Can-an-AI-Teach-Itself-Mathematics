"""Publication-ready descriptive figures from audited CSV outputs."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.grid": True,
                         "grid.alpha": 0.22, "savefig.dpi": 190})


def plot_hitk(rows: list[dict], output: Path, *, title: str) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(8.1, 4.8), constrained_layout=True)
    styling = (("atomic_control", "Атомарный контроль", "#334e68", "-"),
               ("composition_distill", "Композиционное обучение", "#a05a2c", "--"),
               ("random", "Случайный отбор", "#334e68", "-"),
               ("diverse", "Структурированный отбор", "#a05a2c", "--"))
    for arm, label, color, line in styling:
        subset = sorted((row for row in rows if row["arm"] == arm), key=lambda row: int(row["k"]))
        if not subset:
            continue
        x = [int(row["k"]) for row in subset]
        y = [float(row["mean_hit_fraction"]) for row in subset]
        ax.plot(x, y, line, color=color, linewidth=2.0, label=label)
        if all("min_seed" in row and "max_seed" in row for row in subset):
            ax.fill_between(x, [float(row["min_seed"]) for row in subset],
                            [float(row["max_seed"]) for row in subset], color=color, alpha=0.12)
    ax.set(title=title, xlabel="K, число первых программ", ylabel="Доля решённых задач")
    ax.set_xlim(1, max(int(row["k"]) for row in rows))
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_grpo_blocks(rows: list[dict], output: Path) -> None:
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), constrained_layout=True)
    settings = (("sample_entropy_nats", "Выборочная энтропия группы, нат", axes[0, 0]),
                ("collision_fraction", "Доля совпадающих пар ответов", axes[0, 1]),
                ("mean_reward", "Средняя награда группы", axes[1, 0]),
                ("cumulative_tasks_reached", "Накопленное число найденных задач", axes[1, 1]))
    for sampler, label, color, marker in (("iid", "IID", "#334e68", "o"),
                                          ("prefix", "Prefix-balanced", "#a05a2c", "s")):
        by_step = defaultdict(list)
        for row in rows:
            if row["sampler"] == sampler:
                by_step[int(row["step_end"])].append(row)
        x = sorted(by_step)
        for field, ylabel, ax in settings:
            mean = [sum(float(row[field]) for row in by_step[step]) / len(by_step[step]) for step in x]
            low = [min(float(row[field]) for row in by_step[step]) for step in x]
            high = [max(float(row[field]) for row in by_step[step]) for step in x]
            ax.plot(x, mean, marker=marker, markersize=3.2, color=color, label=label)
            ax.fill_between(x, low, high, color=color, alpha=0.12)
            ax.set(xlabel="Шаг обучения", ylabel=ylabel)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("GRPO: разнообразие восьми ответов в группе и покрытие")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_exact_entropy(rows: list[dict], output: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1), constrained_layout=True)
    fields = (("entropy_nats", "Энтропия 125 программ, нат"),
              ("correct_mass", "Масса правильных программ"))
    for ax, (field, ylabel) in zip(axes, fields):
        for index, (arm, label, color) in enumerate((("atomic_control", "Атомарный контроль", "#334e68"),
                                                    ("composition_distill", "Композиционное обучение", "#a05a2c"))):
            values = [float(row[field]) for row in rows if row["arm"] == arm]
            ax.scatter([index + (j - 2.5) * 0.04 for j in range(len(values))], values,
                       marker="o" if index == 0 else "s", color=color, alpha=0.8)
            ax.plot(index, sum(values) / len(values), marker="_", color="black", markersize=18, markeredgewidth=2)
        ax.set_xticks([0, 1], ["Контроль", "Композиция"])
        ax.set_ylabel(ylabel)
    fig.suptitle("Точное ранжирование, шесть независимых обучений")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    data = HERE / "data"
    figures = HERE / "figures"
    plot_hitk(_read(data / "hitk_mean.csv"), figures / "existing_hitk_1_125.png",
              title="Hit@K на ранее сохранённом final-A")
    plot_grpo_blocks(_read(data / "grpo_blocks_per_seed.csv"), figures / "grpo_empirical_entropy.png")
    plot_exact_entropy(_read(data / "seed_metrics.csv"), figures / "existing_exact_entropy.png")
    new_data = data / "new_experiment"
    if (new_data / "hitk_mean.csv").is_file():
        plot_hitk(_read(new_data / "hitk_mean.csv"), figures / "selection_hitk_1_125.png",
                  title="Hit@K на новом отложенном наборе")


if __name__ == "__main__":
    main()
