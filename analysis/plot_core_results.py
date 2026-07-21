#!/usr/bin/env python3
"""Build the three core H2/epsilon figures from collected JSON results.

The figures are intentionally limited to the claims the experiment can support:
the checker phase map, gain versus checker alignment, and the quality/exploration
frontier at a fixed candidate budget K.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
INK = "#263238"
MUTED = "#66757F"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.16,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
    }
)


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def save_figure(fig, out_dir, stem):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(f"wrote {pdf} and {png}")


def seed_note(cells):
    counts = sorted({len(cell.get("seeds", [])) for cell in cells})
    if counts == [1]:
        return "pilot: 1 seed/cell; uncertainty is not estimated"
    if len(counts) == 1 and counts[0] > 1:
        return f"mean over {counts[0]} seeds/cell; error bars are seed SE"
    return "seed counts vary across cells"


def h2_note(cells):
    observed = {(float(cell["alpha"]), float(cell["beta"])) for cell in cells}
    coverage = "" if len(observed) == 25 else f"{len(observed)}/25 cells; "
    return coverage + seed_note(cells)


def plot_h2_heatmap(summary, out_dir):
    cells = [cell for cell in summary.values() if cell.get("mean_gain") is not None]
    if not cells:
        raise ValueError("H2 summary contains no cells with mean_gain")

    observed_alphas = {float(cell["alpha"]) for cell in cells}
    observed_betas = {float(cell["beta"]) for cell in cells}
    canonical_alphas = (0.2, 0.4, 0.6, 0.8, 1.0)
    canonical_betas = (0.0, 0.2, 0.4, 0.6, 0.8)
    uses_canonical_grid = observed_alphas <= set(canonical_alphas) and observed_betas <= set(
        canonical_betas
    )
    alphas = list(canonical_alphas) if uses_canonical_grid else sorted(observed_alphas)
    betas = list(canonical_betas) if uses_canonical_grid else sorted(observed_betas)
    alpha_index = {value: index for index, value in enumerate(alphas)}
    beta_index = {value: index for index, value in enumerate(betas)}
    matrix = np.full((len(alphas), len(betas)), np.nan)
    for cell in cells:
        matrix[alpha_index[float(cell["alpha"])]][beta_index[float(cell["beta"])]] = (
            100 * float(cell["mean_gain"])
        )

    finite = matrix[np.isfinite(matrix)]
    limit = max(1.0, float(np.max(np.abs(finite))))
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#ECEFF1")

    fig, ax = plt.subplots(figsize=(6.75, 3.8))
    image = ax.imshow(
        np.ma.masked_invalid(matrix),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=-limit,
        vmax=limit,
    )
    ax.grid(False)
    ax.set_xticks(range(len(betas)), [f"{value:g}" for value in betas])
    ax.set_yticks(range(len(alphas)), [f"{value:g}" for value in alphas])
    ax.set_xlabel(r"false-positive rate $\beta$")
    ax.set_ylabel(r"true-positive rate $\alpha$")
    observed_count = len({(float(cell["alpha"]), float(cell["beta"])) for cell in cells})
    title = (
        "H2 phase map: held-out accuracy gain"
        if observed_count == 25
        else f"Partial H2 diagnostic: {observed_count}/25 cells"
    )
    ax.set_title(title, loc="left", pad=12)
    ax.text(1, 1.025, h2_note(cells), transform=ax.transAxes, ha="right", color=MUTED, fontsize=8)

    for row in range(len(alphas)):
        for col in range(len(betas)):
            value = matrix[row, col]
            if not np.isfinite(value):
                continue
            rgba = cmap((value + limit) / (2 * limit))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(
                col,
                row,
                f"{value:+.1f}",
                ha="center",
                va="center",
                color="white" if luminance < 0.55 else INK,
                fontsize=8,
                fontweight="bold",
            )

    lower = max(min(alphas), min(betas))
    upper = min(max(alphas), max(betas))
    if lower <= upper:
        boundary = np.linspace(lower, upper, 100)
        xs = np.interp(boundary, betas, np.arange(len(betas)))
        ys = np.interp(boundary, alphas, np.arange(len(alphas)))
        ax.plot(xs, ys, color=INK, linestyle="--", linewidth=1.2, label=r"$\alpha=\beta$")
        ax.legend(loc="lower right")

    colorbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.03)
    colorbar.set_label(r"$\Delta$ true accuracy (percentage points)")
    fig.tight_layout()
    save_figure(fig, out_dir, "h2_gain_heatmap")


def plot_h2_alignment(summary, out_dir):
    cells = [cell for cell in summary.values() if cell.get("mean_gain") is not None]
    if not cells:
        raise ValueError("H2 summary contains no cells with mean_gain")

    use_realized = all(cell.get("mean_realized_signed_alignment") is not None for cell in cells)
    x_key = "mean_realized_signed_alignment" if use_realized else "info"
    x_se_key = "se_realized_signed_alignment" if use_realized else None
    xlabel = (
        "realized checker alignment (TPR − FPR)"
        if use_realized
        else r"configured checker signal $J=\alpha-\beta$"
    )

    xs = np.asarray([float(cell[x_key]) for cell in cells])
    ys = np.asarray([100 * float(cell["mean_gain"]) for cell in cells])
    betas = np.asarray([float(cell["beta"]) for cell in cells])
    seed_counts = [len(cell.get("seeds", [])) for cell in cells]
    show_seed_errors = any(count > 1 for count in seed_counts)
    yerr = None
    if show_seed_errors:
        yerr = np.asarray([100 * float(cell.get("se_gain") or 0) for cell in cells])
    xerr = None
    if show_seed_errors and x_se_key:
        xerr = np.asarray([float(cell.get(x_se_key) or 0) for cell in cells])

    fig, ax = plt.subplots(figsize=(6.75, 3.8))
    if xerr is not None or yerr is not None:
        ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, fmt="none", color="#90A4AE", capsize=2)
    scatter = ax.scatter(
        xs,
        ys,
        c=betas,
        cmap="cividis",
        s=52,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )

    by_signal = defaultdict(list)
    for cell, x, y in zip(cells, xs, ys):
        by_signal[round(float(cell["info"]), 8)].append((float(x), float(y)))
    trend = [
        (fmean(point[0] for point in points), fmean(point[1] for point in points))
        for _, points in sorted(by_signal.items())
    ]
    if len(trend) > 1:
        ax.plot(
            [point[0] for point in trend],
            [point[1] for point in trend],
            color=INK,
            marker="D",
            markersize=4,
            label="mean at equal configured J",
            zorder=2,
        )

    ax.axhline(0, color=INK, linewidth=0.8)
    ax.axvline(0, color=MUTED, linewidth=0.9, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\Delta$ true accuracy (percentage points)")
    ax.set_title("H2: improvement versus checker alignment", loc="left", pad=12)
    ax.text(1, 1.025, h2_note(cells), transform=ax.transAxes, ha="right", color=MUTED, fontsize=8)
    if len(trend) > 1:
        ax.legend(loc="upper left")
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.82, pad=0.03)
    colorbar.set_label(r"configured $\beta$")
    fig.tight_layout()
    save_figure(fig, out_dir, "h2_gain_vs_alignment")


def plot_epsilon_frontier(result, out_dir):
    conditions = [
        condition
        for condition in result.get("conditions", [])
        if condition.get("kind") in {"temperature", "mixture"}
        and condition.get("mean", {}).get("delta_epsilon_parseable") is not None
        and condition.get("mean", {}).get("pass@1") is not None
    ]
    if not conditions:
        raise ValueError("exploration result contains no plottable conditions")

    kinds = [kind for kind in ("temperature", "mixture") if any(c["kind"] == kind for c in conditions)]
    resolutions = sorted({int(condition["resolution_m"]) for condition in conditions})
    colors = {resolution: OKABE_ITO[index % len(OKABE_ITO)] for index, resolution in enumerate(resolutions)}
    fig, axes = plt.subplots(1, len(kinds), figsize=(6.75, 3.8), sharey=True, squeeze=False)
    axes = axes[0]

    reference_temperature = result.get("config", {}).get("reference_temperature")
    baseline = next(
        (
            condition["mean"]["pass@1"]
            for condition in conditions
            if condition["kind"] == "temperature"
            and condition.get("temperature") == reference_temperature
        ),
        None,
    )

    for ax, kind in zip(axes, kinds):
        parameter_key = "temperature" if kind == "temperature" else "requested_weight"
        for resolution in resolutions:
            group = sorted(
                (
                    condition
                    for condition in conditions
                    if condition["kind"] == kind and int(condition["resolution_m"]) == resolution
                ),
                key=lambda condition: float(condition[parameter_key]),
            )
            if not group:
                continue
            xs = [100 * float(condition["mean"]["delta_epsilon_parseable"]) for condition in group]
            ys = [100 * float(condition["mean"]["pass@1"]) for condition in group]
            xerr_values = [
                100 * float(condition["mean"].get("se_delta_epsilon_parseable") or 0)
                for condition in group
            ]
            yerr_values = [
                100 * float(condition["mean"].get("se_pass@1") or 0) for condition in group
            ]
            ax.errorbar(
                xs,
                ys,
                xerr=xerr_values if any(xerr_values) else None,
                yerr=yerr_values if any(yerr_values) else None,
                color=colors[resolution],
                marker="o" if kind == "temperature" else "s",
                capsize=2,
                alpha=0.9,
            )
            if resolution == max(resolutions):
                symbol = "T" if kind == "temperature" else r"$\lambda$"
                label_indices = {0, len(group) - 1}
                if kind == "temperature" and reference_temperature is not None:
                    label_indices.update(
                        index
                        for index, condition in enumerate(group)
                        if condition.get("temperature") == reference_temperature
                    )
                for index, (x, y, condition) in enumerate(zip(xs, ys, group)):
                    if index not in label_indices:
                        continue
                    ax.annotate(
                        f"{symbol}={float(condition[parameter_key]):g}",
                        (x, y),
                        xytext=(4, 5 if index % 2 == 0 else -10),
                        textcoords="offset points",
                        fontsize=6.8,
                        color=MUTED,
                    )

        if baseline is not None:
            ax.axhline(100 * baseline, color="#90A4AE", linewidth=0.9, linestyle=":")
        ax.axvline(0, color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_xlabel(r"$\Delta\widehat{\varepsilon}_{parseable}^{1/m}$ (percentage points)")
        ax.set_title("Temperature" if kind == "temperature" else "Policy mixture", loc="left")
    axes[0].set_ylabel("pass@1 (%)")

    handles = [
        Line2D([], [], color=colors[resolution], marker="o", label=rf"$m={resolution}$")
        for resolution in resolutions
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.01))
    candidate_k = result.get("definition", {}).get("candidate_k", "?")
    fig.suptitle("Quality–exploration frontier at fixed candidate budget", x=0.01, ha="left", y=1.04)
    fig.text(
        0.01,
        -0.01,
        rf"$K={candidate_k}$ fixed; bars are problem-level SE. $\widehat{{\varepsilon}}$ is answer-level novelty, not a strategy metric.",
        color=MUTED,
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    save_figure(fig, out_dir, "epsilon_frontier")


def demo_data():
    summary = {}
    for alpha in (0.2, 0.4, 0.6, 0.8, 1.0):
        for beta in (0.0, 0.2, 0.4, 0.6, 0.8):
            signal = alpha - beta
            summary[f"a{alpha}_b{beta}"] = {
                "alpha": alpha,
                "beta": beta,
                "info": signal,
                "seeds": [0, 1, 2],
                "mean_gain": 0.24 * signal - 0.015 * beta,
                "se_gain": 0.012,
                "mean_realized_signed_alignment": 0.92 * signal,
                "se_realized_signed_alignment": 0.015,
            }

    conditions = []
    for resolution in (4, 8, 16):
        scale = 1 - 1 / (resolution + 2)
        for temperature in (0.5, 0.8, 1.0, 1.3):
            delta = scale * (temperature - 1) * 0.16
            conditions.append(
                {
                    "kind": "temperature",
                    "temperature": temperature,
                    "resolution_m": resolution,
                    "mean": {
                        "delta_epsilon_parseable": delta,
                        "se_delta_epsilon_parseable": 0.008,
                        "pass@1": 0.42 - 0.16 * max(delta, 0) + 0.04 * min(delta, 0),
                        "se_pass@1": 0.015,
                    },
                }
            )
        for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            delta = scale * weight * 0.12
            conditions.append(
                {
                    "kind": "mixture",
                    "requested_weight": weight,
                    "resolution_m": resolution,
                    "mean": {
                        "delta_epsilon_parseable": delta,
                        "se_delta_epsilon_parseable": 0.008,
                        "pass@1": 0.42 + 0.05 * weight - 0.04 * weight**2,
                        "se_pass@1": 0.015,
                    },
                }
            )
    exploration = {
        "definition": {"candidate_k": 8},
        "config": {"reference_temperature": 1.0},
        "conditions": conditions,
    }
    return summary, exploration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h2-summary", help="path to checker_grid_summary.json")
    parser.add_argument("--exploration", help="path to eval_exploration.py JSON output")
    parser.add_argument("--out", default="results/figures/core")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="render deterministic synthetic data to verify plotting without a GPU",
    )
    args = parser.parse_args()

    if args.demo and (args.h2_summary or args.exploration):
        parser.error("--demo cannot be combined with input files")
    if not args.demo and not (args.h2_summary or args.exploration):
        parser.error("provide --h2-summary, --exploration, or --demo")

    if args.demo:
        print("DEMO ONLY: rendering synthetic values; do not interpret them as results")
        summary, exploration = demo_data()
        plot_h2_heatmap(summary, args.out)
        plot_h2_alignment(summary, args.out)
        plot_epsilon_frontier(exploration, args.out)
        return
    if args.h2_summary:
        summary = read_json(args.h2_summary)
        plot_h2_heatmap(summary, args.out)
        plot_h2_alignment(summary, args.out)
    if args.exploration:
        exploration = read_json(args.exploration)
        plot_epsilon_frontier(exploration, args.out)


if __name__ == "__main__":
    main()
