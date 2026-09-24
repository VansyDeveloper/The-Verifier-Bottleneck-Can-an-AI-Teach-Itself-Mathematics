"""Plot exact Hit@K curves for the audited first pair set."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA = HERE / "data/first_block"
SEEDS = (85000, 85001, 85002)


def aggregate_curve(rows: list[dict], arm: str, split: str, *, maximum: int = 125):
    table = {}
    for row in rows:
        if row["arm"] == arm and row["split"] == split:
            key = (int(row["k"]), int(row["seed"]))
            if key in table:
                raise ValueError(f"duplicate Hit@K observation: {key}")
            table[key] = float(row["hit_k"])
    result = []
    for k in range(1, maximum + 1):
        values = [table.get((k, seed)) for seed in SEEDS]
        if any(value is None for value in values):
            raise ValueError(f"incomplete Hit@K curves at K={k}")
        result.append((k, statistics.mean(values), min(values), max(values)))
    return result


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (DATA / "hit_k.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=True, constrained_layout=True)
    styles = {"random": ("#315f89", "-", "Random exclusion"),
              "withheld": ("#a8493b", "--", "Pair exclusion")}
    for axis, split in zip(axes, ("withheld", "control")):
        for arm in ("random", "withheld"):
            points = aggregate_curve(rows, arm, split)
            k, mean, low, high = zip(*points)
            color, line, label = styles[arm]
            axis.fill_between(k, low, high, color=color, alpha=0.11, linewidth=0)
            axis.plot(k, mean, line, color=color, linewidth=2, label=label)
            axis.plot([32], [mean[31]], "o" if arm == "random" else "s", color=color, markersize=5)
        axis.axvline(32, color="#5b6268", linewidth=0.8, linestyle=":")
        axis.set_xlim(1, 125)
        axis.set_ylim(0, 1.02)
        axis.set_xticks((1, 16, 32, 64, 96, 125))
        axis.set_xlabel("K")
        axis.set_title("Tasks with excluded pair" if split == "withheld" else "Control tasks")
        axis.grid(axis="y", color="#d7dce0", linewidth=0.55)
    axes[0].set_ylabel("Fraction solved within top K")
    axes[0].legend(frameon=False, loc="lower right")
    figure.savefig(DATA / "hit_k_first_block.png", dpi=240)
    figure.savefig(DATA / "hit_k_first_block.svg")
    plt.close(figure)


if __name__ == "__main__":
    main()
