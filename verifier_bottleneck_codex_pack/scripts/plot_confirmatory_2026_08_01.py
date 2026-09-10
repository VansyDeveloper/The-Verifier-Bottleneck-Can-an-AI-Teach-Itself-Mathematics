"""Figures for the D-018 confirmatory series.

Everything is drawn from `artifacts/reports/confirmatory_statistics.json` and the
GRPO run metrics, never from hand-entered numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPORTS = Path("artifacts/reports")
RUNS = Path("artifacts/runs")

ARM_LABEL = {
    "iid_action@0.7": "iid T=0.7",
    "iid_action@2.0": "iid T=2.0",
    "iid_action@3.0": "iid T=3.0",
    "prefix_balanced_action@0.7": "prefix-balanced",
    "temp_mix_action@0.2,0.6,1.0,1.4": "temp mix",
}
ROLE_LABEL = {"pre": "pre-GRPO", "iidgrpo": "GRPO iid branch", "prefixgrpo": "GRPO prefix branch"}
IID_ARMS = ["iid_action@0.7", "temp_mix_action@0.2,0.6,1.0,1.4", "iid_action@2.0", "iid_action@3.0"]


def load():
    return json.loads((REPORTS / "confirmatory_statistics.json").read_text())


def figure_diversity_curve(table):
    """The central figure: pass@32 against realised candidate diversity."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, xkey, xlabel in (
        (axes[0], "distinct_programs_per_task", "distinct programs per task (of 32 candidates)"),
        (axes[1], "model_passes", "model scoring passes (300 tasks)"),
    ):
        for role, colour in (("pre", "#1f77b4"), ("iidgrpo", "#7f7f7f"), ("prefixgrpo", "#bcbd22")):
            xs = [table[f"{role}|{arm}"][xkey] for arm in IID_ARMS if f"{role}|{arm}" in table]
            ys = [table[f"{role}|{arm}"]["pass_at_32"] for arm in IID_ARMS if f"{role}|{arm}" in table]
            ax.plot(xs, ys, "o-", color=colour, label=f"{ROLE_LABEL[role]}: iid sweep", alpha=0.85)
            key = f"{role}|prefix_balanced_action@0.7"
            if key in table:
                ax.plot(
                    table[key][xkey],
                    table[key]["pass_at_32"],
                    "*",
                    markersize=17,
                    color=colour,
                    markeredgecolor="black",
                    markeredgewidth=0.7,
                    label=f"{ROLE_LABEL[role]}: prefix-balanced",
                )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("pass@32")
        ax.grid(alpha=0.3)
    axes[0].set_title("Structure sits on the diversity curve, not above it")
    axes[1].set_title("Same picture at equal model-pass cost")
    axes[1].legend(fontsize=7, loc="upper left")
    fig.suptitle("confirmatory_heldout, 300 PLAN depth-3 tasks, primes 11/17, K=32, mean of 3 seeds")
    fig.tight_layout()
    fig.savefig(REPORTS / "confirmatory_diversity_curve.png", dpi=160)
    plt.close(fig)


def figure_arm_by_role(table):
    arms = list(ARM_LABEL)
    roles = ["pre", "iidgrpo", "prefixgrpo"]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    width = 0.26
    for index, role in enumerate(roles):
        values = [table.get(f"{role}|{arm}", {}).get("pass_at_32", 0.0) for arm in arms]
        ax.bar(
            [position + (index - 1) * width for position in range(len(arms))],
            values,
            width=width,
            label=ROLE_LABEL[role],
        )
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([ARM_LABEL[arm] for arm in arms], fontsize=9)
    ax.set_ylabel("pass@32")
    ax.set_title("Search arm dominates; 400-step GRPO does not improve any arm")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "confirmatory_arm_by_role.png", dpi=160)
    plt.close(fig)


def figure_contrasts(contrasts):
    order = [
        ("Q1 primary\nprefix - iid@2.0", "Q1_primary_structure_vs_matched_diversity"),
        ("Q2 GRPO effect\niid branch", "Q2_grpo_effect_iid_branch"),
        ("Q2 GRPO effect\nprefix branch", "Q2_grpo_effect_prefix_branch"),
        ("Q3 published\ndiagonal", "Q3_published_diagonal"),
        ("Q4\niid@3.0 - prefix", "Q4_hot_iid_vs_structure"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for index, (label, key) in enumerate(order):
        item = contrasts[key]
        low, high = item["hierarchical_bootstrap_95_ci_pp"]
        centre = item["mean_delta_pp"]
        colour = "#d62728" if not item["ci_excludes_zero"] else "#2ca02c"
        ax.errorbar(
            index, centre, yerr=[[centre - low], [high - centre]],
            fmt="o", color=colour, capsize=6, markersize=8,
        )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([label for label, _ in order], fontsize=8)
    ax.set_ylabel("delta pass@32, percentage points")
    ax.set_title("Preregistered contrasts, hierarchical paired bootstrap 95% CI, 3 seeds")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS / "confirmatory_contrasts.png", dpi=160)
    plt.close(fig)


def figure_grpo_curves():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for run in sorted(RUNS.glob("*_plan_grpo_*")):
        metrics_path = run / "metrics.jsonl"
        if not (run / "DONE").exists() or not metrics_path.exists():
            continue
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        if len(rows) < 400:
            continue
        method = "prefix" if "prefix" in run.name else "iid"
        seed = run.name.rsplit("seed", 1)[-1]
        colour = "#1f77b4" if method == "iid" else "#ff7f0e"
        rewards = [row["metrics"]["mean_reward"] for row in rows]
        window = 25
        smoothed = [
            sum(rewards[max(0, i - window):i + 1]) / len(rewards[max(0, i - window):i + 1])
            for i in range(len(rewards))
        ]
        axes[0].plot(smoothed, color=colour, alpha=0.7, label=f"{method} seed{seed}")
        cumulative = []
        total = 0
        for value in rewards:
            total += 1 if value > 0 else 0
            cumulative.append(total)
        axes[1].plot(cumulative, color=colour, alpha=0.7, label=f"{method} seed{seed}")
    axes[0].set_xlabel("GRPO step")
    axes[0].set_ylabel("mean group reward (25-step window)")
    axes[0].set_title("Candidate-level reward does not separate")
    axes[1].set_xlabel("GRPO step")
    axes[1].set_ylabel("cumulative groups with >=1 correct")
    axes[1].set_title("Prefix-balanced finds correct programs in more groups")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(REPORTS / "confirmatory_grpo_training.png", dpi=160)
    plt.close(fig)


def main():
    payload = load()
    figure_diversity_curve(payload["arm_table"])
    figure_arm_by_role(payload["arm_table"])
    figure_contrasts(payload["contrasts"])
    figure_grpo_curves()
    for name in (
        "confirmatory_diversity_curve.png",
        "confirmatory_arm_by_role.png",
        "confirmatory_contrasts.png",
        "confirmatory_grpo_training.png",
    ):
        print(REPORTS / name)


if __name__ == "__main__":
    main()
