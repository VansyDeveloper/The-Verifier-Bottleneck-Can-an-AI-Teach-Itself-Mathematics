"""Publication figure for the final D-021/D-022 exhaustive-ranking analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("artifacts/reports/ranking_statistics.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports/ranking_contrasts"))
    return parser.parse_args()


def values(results: dict, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    points, errors = [], []
    for name in names:
        item = results[name]
        point = item["mean_delta_pp"]
        low, high = item["crossed_seed_task_bootstrap_95_ci_pp"]
        points.append(point)
        errors.append((point - low, high - point))
    return np.array(points), np.array(errors).T


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    results = payload["contrasts"]
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.15), constrained_layout=True)
    d021_names = [
        "C1_primary_supervision_transfers_to_heldout_fields",
        "C2_compositions_learnable_in_domain",
        "C3_depth_effect_fields_fixed_d2_minus_d3_at_hit8",
        "C4_field_effect_depth_fixed_train_minus_heldout",
    ]
    d021_labels = ["C1 transfer", "C2 learnability", "C3 depth", "C4 field"]
    points, errors = values(results, d021_names)
    y = np.arange(len(points))
    axes[0].barh(y, points, color=["#0072B2", "#009E73", "#E69F00", "#8C8C8C"], height=0.58)
    axes[0].errorbar(points, y, xerr=errors, fmt="none", ecolor="#222222", capsize=3, linewidth=1)
    axes[0].set(yticks=y, yticklabels=d021_labels, xlabel="Delta Hit@K (percentage points)")
    axes[0].set_title("D-021: capacity decomposition")
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="#333333", linewidth=0.8)

    motif_names = [
        "M3a_seen_motif_known_field",
        "M1_primary_held_motif_known_field",
        "M3c_seen_motif_new_field",
        "M2_held_motif_new_field",
    ]
    motif_labels = ["A known / seen", "B known / held", "C new / seen", "D new / held"]
    points, errors = values(results, motif_names)
    y = np.arange(len(points))
    bars = axes[1].barh(
        y,
        points,
        color=["#009E73", "#D55E00", "#56B4E9", "#CC79A7"],
        height=0.58,
    )
    for bar, held in zip(bars, (False, True, False, True)):
        if held:
            bar.set_hatch("///")
    axes[1].errorbar(points, y, xerr=errors, fmt="none", ecolor="#222222", capsize=3, linewidth=1)
    axes[1].set(yticks=y, yticklabels=motif_labels, xlabel="Delta Hit@32 (percentage points)")
    axes[1].set_title("D-022: field x motif transfer")
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="#333333", linewidth=0.8)

    for ax in axes:
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"))
    fig.savefig(args.output.with_suffix(".png"), dpi=300)
    print(args.output.with_suffix(".pdf"))
    print(args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
