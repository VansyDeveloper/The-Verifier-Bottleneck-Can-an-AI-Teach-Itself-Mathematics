"""Figures for the D-020 self-training series.

Reads only `artifacts/reports/selftraining_statistics.json` and the dev sweep runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPORTS = Path("artifacts/reports")
RUNS = Path("artifacts/runs")

ROLE_LABEL = {"pre": "pre-GRPO", "coldgrpo": "GRPO T_train=0.7", "hotgrpo": "GRPO T_train=8.0"}
ROLE_COLOUR = {"pre": "#1f77b4", "coldgrpo": "#7f7f7f", "hotgrpo": "#d62728"}
IID_ARMS = ["iid_action@0.7", "iid_action@3.0", "iid_action@8.0", "iid_action@100.0"]


def figure_mechanism(table):
    """GRPO raised per-candidate accuracy and lost exactly as much coverage."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    roles = ["pre", "coldgrpo", "hotgrpo"]
    arm = "iid_action@0.7"
    for ax, key, title, fmt in (
        (axes[0], "candidate_correct_rate", "per-candidate accuracy\n(the policy did learn)", "{:.4f}"),
        (axes[1], "distinct_programs_per_task", "distinct programs per task\n(coverage collapsed)", "{:.2f}"),
        (axes[2], "pass_at_32", "pass@32\n(net effect: nothing)", "{:.3f}"),
    ):
        values = [table[f"{role}|{arm}"][key] for role in roles]
        bars = ax.bar([ROLE_LABEL[r] for r in roles], values, color=[ROLE_COLOUR[r] for r in roles])
        for rect, value in zip(bars, values):
            ax.text(rect.get_x() + rect.get_width() / 2, value, fmt.format(value),
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=8, rotation=12)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(values) * 1.25)
    fig.suptitle("Self-training redistributes probability mass: measured at the iid T=0.7 arm, "
                 "confirmatory_heldout_v2, 3 seeds", fontsize=11)
    fig.tight_layout()
    fig.savefig(REPORTS / "selftraining_mechanism.png", dpi=160)
    plt.close(fig)


def figure_arms(table):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for role in ("pre", "coldgrpo", "hotgrpo"):
        xs = [table[f"{role}|{arm}"]["distinct_programs_per_task"] for arm in IID_ARMS]
        ys = [table[f"{role}|{arm}"]["pass_at_32"] for arm in IID_ARMS]
        ax.plot(xs, ys, "o-", color=ROLE_COLOUR[role], alpha=0.85, label=f"{ROLE_LABEL[role]}: iid sweep")
        key = f"{role}|prefix_balanced_action@0.7"
        ax.plot(table[key]["distinct_programs_per_task"], table[key]["pass_at_32"], "*",
                markersize=16, color=ROLE_COLOUR[role], markeredgecolor="black", markeredgewidth=0.7,
                label=f"{ROLE_LABEL[role]}: prefix-balanced")
    ax.set_xlabel("distinct programs per task (of 32 candidates)")
    ax.set_ylabel("pass@32")
    ax.set_title("All three adapters share one diversity curve; structure stays below it")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(REPORTS / "selftraining_arms.png", dpi=160)
    plt.close(fig)


def figure_dev_sweep():
    rows = {}
    for run in RUNS.glob("*diversity-screen*"):
        if (run / "DONE").exists():
            rows.update(json.loads((run / "final_metrics.json").read_text())["methods"])
    iid = sorted(((float(m.split("@")[1]), v) for m, v in rows.items() if m.startswith("iid_action")),
                 key=lambda item: item[0])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot([t for t, _ in iid], [v["pass_at_32"] for _, v in iid], "o-", color="#1f77b4",
            label="iid at temperature T")
    prefix = rows.get("prefix_balanced_action@0.7")
    if prefix:
        ax.axhline(prefix["pass_at_32"], color="#d62728", linestyle="--",
                   label=f"prefix-balanced ({prefix['pass_at_32']:.3f})")
    best = max(iid, key=lambda item: item[1]["pass_at_32"])
    ax.annotate(f"optimum T={best[0]:g}\n{best[1]['pass_at_32']:.3f}",
                xy=(best[0], best[1]["pass_at_32"]), xytext=(best[0] * 0.35, best[1]["pass_at_32"] + 0.02),
                arrowprops=dict(arrowstyle="->", lw=1), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("iid sampling temperature (log scale; T=100 is effectively uniform)")
    ax.set_ylabel("pass@32")
    ax.set_title("Dev temperature sweep: an interior optimum, and structure below every hot arm")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "dev_temperature_sweep.png", dpi=160)
    plt.close(fig)


def figure_contrasts(contrasts):
    order = [
        ("Q5 primary\nself-training,\nadequate exploration", "Q5_primary_selftraining_with_adequate_exploration"),
        ("Q6\nhot vs cold\ntraining sampler", "Q6_hot_versus_cold_training_sampler"),
        ("Q7\nuniform random\n- prefix", "Q7_uniform_random_versus_structure"),
        ("Q8\nbest iid\n- prefix", "Q8_best_iid_versus_structure"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for index, (label, key) in enumerate(order):
        item = contrasts[key]
        low, high = item["hierarchical_bootstrap_95_ci_pp"]
        centre = item["mean_delta_pp"]
        colour = "#2ca02c" if item["ci_excludes_zero"] else "#d62728"
        ax.errorbar(index, centre, yerr=[[centre - low], [high - centre]], fmt="o",
                    color=colour, capsize=6, markersize=8)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([label for label, _ in order], fontsize=8)
    ax.set_ylabel("delta pass@32, percentage points")
    ax.set_title("D-020 preregistered contrasts, hierarchical bootstrap 95% CI, 3 seeds")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS / "selftraining_contrasts.png", dpi=160)
    plt.close(fig)


def main():
    payload = json.loads((REPORTS / "selftraining_statistics.json").read_text())
    figure_mechanism(payload["arm_table"])
    figure_arms(payload["arm_table"])
    figure_dev_sweep()
    figure_contrasts(payload["contrasts"])
    for name in ("selftraining_mechanism.png", "selftraining_arms.png",
                 "dev_temperature_sweep.png", "selftraining_contrasts.png"):
        print(REPORTS / name)


if __name__ == "__main__":
    main()
