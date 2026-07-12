"""Charts for the H1 rework of verifier_bottleneck_plan.html (slides 5-10).

Style matches the deck: Arial, ink #101114, blue #2563eb, green #168a4c,
orange #e95f18, muted #60656f, light grid, no top/right spines.
"""

import json
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUT, LINE = "#101114", "#60656f", "#d9dde5"
BLUE, GREEN, ORANGE = "#2563eb", "#168a4c", "#e95f18"
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 13,
        "axes.edgecolor": LINE,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUT,
        "ytick.color": MUT,
        "axes.titlesize": 15,
        "axes.titlecolor": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)
D = "results_remote/ph3n25p29"
OUT = "outputs/verifier_bottleneck_assets"


def per_problem(path):
    j = json.load(open(os.path.join(D, path)))
    return [(r["n_correct"], r["n"]) for r in j["per_problem"]]


def pass_at_k_vals(pp, k):
    """Unbiased estimator, per problem."""
    vals = []
    for c, n in pp:
        if n - c < k:
            vals.append(1.0)
        else:
            # 1 - C(n-c,k)/C(n,k) computed stably in log space
            v = 1.0
            for i in range(k):
                v *= (n - c - i) / (n - i)
            vals.append(1.0 - v)
    return np.array(vals)


def pass_at_k(pp, k):
    return float(np.mean(pass_at_k_vals(pp, k)))


def boot_band(pp, ks, n_boot=3000, seed=0):
    """95% bootstrap CI over problems for each k."""
    rng = np.random.default_rng(seed)
    mat = np.stack([pass_at_k_vals(pp, k) for k in ks])  # (K, P)
    idx = rng.integers(0, mat.shape[1], size=(n_boot, mat.shape[1]))
    means = mat[:, idx].mean(axis=2)  # (K, n_boot)
    return np.percentile(means, 2.5, axis=1), np.percentile(means, 97.5, axis=1)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(LINE)
    ax.grid(axis="y", color=LINE, lw=0.7, alpha=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- chart 1
# pass@k curves: base (to 1024) vs GRPO m=1.0 (to 64), held-out
def chart_boundary():
    base = per_problem("base_0.6B_eval_k1024.json")
    rl = per_problem("ph3n25p29_lo_a1.0_b0.0_eval.json")
    ks_b = [2**i for i in range(11)]
    ks_r = [2**i for i in range(7)]
    fig, ax = plt.subplots(figsize=(7.6, 4.7), dpi=200)
    ax.plot(
        ks_b, [pass_at_k(base, k) for k in ks_b], "-o", color=INK, ms=5, lw=2, label="Base model"
    )
    ax.plot(
        ks_r,
        [pass_at_k(rl, k) for k in ks_r],
        "-o",
        color=BLUE,
        ms=5,
        lw=2,
        label="GRPO, reliable verifier",
    )
    ax.axhline(
        pass_at_k(base, 1024), color=MUT, lw=1.2, ls=":", label="base boundary at k=1024 (0.90)"
    )
    ax.annotate(
        "pass@1 = 0.88",
        xy=(1, pass_at_k(rl, 1)),
        xytext=(1.5, 0.78),
        fontsize=12,
        color=BLUE,
        arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8),
    )
    ax.annotate(
        "solves all 200 held-out problems by k=64",
        xy=(64, 1.005),
        xytext=(6, 1.09),
        fontsize=12,
        color=BLUE,
        va="center",
        arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8),
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks_b)
    ax.set_xticklabels([str(k) for k in ks_b])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Number of attempts, k")
    ax.set_ylabel("Held-out pass@k")
    ax.set_ylim(0, 1.14)
    style(ax)
    ax.legend(frameon=False, loc="lower right", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/passk_boundary.png")
    plt.close(fig)


# ---------------------------------------------------------------- chart 2
# H1 result: base vs offline SFT vs on-policy GRPO, pass@k curves, 2 panels
def chart_offline():
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), dpi=200, sharey=True)
    for ax, split, title in [
        (axes[0], "train", "Training prompts (pool source)"),
        (axes[1], "eval", "Held-out primes"),
    ]:
        base = per_problem(f"base_0.6B_{split}.json")
        sft = per_problem(f"ph4_offsft_{split}.json")
        rl = per_problem(f"ph3n25p29_mid_a1.0_b0.0_{split}.json")
        ks = [2**i for i in range(7)]
        for pp, col, lab in [
            (base, INK, "Base"),
            (sft, ORANGE, "Offline SFT on frozen pool"),
            (rl, BLUE, "On-policy GRPO"),
        ]:
            lo95, hi95 = boot_band(pp, ks)
            ax.fill_between(ks, lo95, hi95, color=col, alpha=0.15, lw=0)
            ax.plot(ks, [pass_at_k(pp, k) for k in ks], "-o", color=col, ms=4.5, lw=2, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("Number of attempts, k")
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
        style(ax)
    axes[0].set_ylabel("pass@k")
    axes[0].legend(frameon=False, loc="center left", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/h1_offline_vs_onpolicy.png")
    plt.close(fig)


# ---------------------------------------------------------------- chart 2 v2
# Same comparison, honest panel titles: left = fresh train-set problems
# (not the pool prompts SFT saw), right = held-out primes.
def chart_offline_v2():
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), dpi=200, sharey=True)
    for ax, split, title in [
        (axes[0], "train", "Train set"),
        (axes[1], "eval", "Held-out primes"),
    ]:
        base = per_problem(f"base_0.6B_{split}.json")
        sft = per_problem(f"ph4_offsft_{split}.json")
        rl = per_problem(f"ph3n25p29_mid_a1.0_b0.0_{split}.json")
        ks = [2**i for i in range(7)]
        for pp, col, lab in [
            (base, INK, "Base"),
            (sft, ORANGE, "Offline SFT on frozen pool"),
            (rl, BLUE, "On-policy GRPO"),
        ]:
            lo95, hi95 = boot_band(pp, ks)
            ax.fill_between(ks, lo95, hi95, color=col, alpha=0.15, lw=0)
            ax.plot(ks, [pass_at_k(pp, k) for k in ks], "-o", color=col, ms=4.5, lw=2, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("Number of attempts, k")
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
        style(ax)
    axes[0].set_ylabel("pass@k")
    axes[0].legend(frameon=False, loc="center left", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/h1_offline_vs_onpolicy_v2.png")
    plt.close(fig)


# ---------------------------------------------------------------- chart 2 v3
# Left = the 678 pool prompts SFT actually trained on (covered), right = held-out.
def chart_offline_v3():
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), dpi=200, sharey=True)
    for ax, files, title in [
        (
            axes[0],
            (
                "base_0.6B_covered.json",
                "ph4_offsft_covered.json",
                "ph3n25p29_mid_a1.0_b0.0_covered.json",
            ),
            "SFT training prompts",
        ),
        (
            axes[1],
            ("base_0.6B_eval.json", "ph4_offsft_eval.json", "ph3n25p29_mid_a1.0_b0.0_eval.json"),
            "Held-out primes",
        ),
    ]:
        ks = [2**i for i in range(7)]
        for f, col, lab in [
            (files[0], INK, "Base"),
            (files[1], ORANGE, "Offline SFT on frozen pool"),
            (files[2], BLUE, "On-policy GRPO"),
        ]:
            pp = per_problem(f)
            lo95, hi95 = boot_band(pp, ks)
            ax.fill_between(ks, lo95, hi95, color=col, alpha=0.15, lw=0)
            ax.plot(ks, [pass_at_k(pp, k) for k in ks], "-o", color=col, ms=4.5, lw=2, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("Number of attempts, k")
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
        style(ax)
    axes[0].set_ylabel("pass@k")
    axes[0].legend(frameon=False, loc="lower right", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/h1_offline_vs_onpolicy_v3.png")
    plt.close(fig)


# ---------------------------------------------------------------- chart 2b
# Offline SFT vs base, both to k=1024: the boundary does not move
def chart_sft_boundary():
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), dpi=200, sharey=True)
    for ax, split, title in [
        (axes[0], "train", "Training prompts (SFT saw these)"),
        (axes[1], "eval", "Held-out primes"),
    ]:
        base = per_problem(f"base_0.6B_{split}_k1024.json")
        sft = per_problem(f"ph4_offsft_{split}_k1024.json")
        ks = [2**i for i in range(11)]
        ax.plot(
            ks, [pass_at_k(base, k) for k in ks], "-o", color=INK, ms=4.5, lw=2, label="Base model"
        )
        ax.plot(
            ks,
            [pass_at_k(sft, k) for k in ks],
            "-o",
            color=ORANGE,
            ms=4.5,
            lw=2,
            label="Offline SFT on frozen pool",
        )
        b, s = pass_at_k(base, 1024), pass_at_k(sft, 1024)
        ax.annotate(
            f"pass@1024:  SFT {s:.2f}  vs  base {b:.2f}", xy=(1.3, 0.97), fontsize=12.5, color=INK
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks], fontsize=10.5)
        ax.set_xlabel("Number of attempts, k")
        ax.set_title(title)
        ax.set_ylim(0, 1.06)
        style(ax)
    axes[0].set_ylabel("pass@k")
    axes[0].legend(frameon=False, loc="lower right", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/h1_sft_boundary.png")
    plt.close(fig)


# ---------------------------------------------------------------- chart 3
# Boundary movement: newly reached problems beyond base k=1024
def chart_newly():
    tr = json.load(open("results_remote/task_transitions_k1024r.json"))
    rows = [
        ("Offline SFT", "ph4_offsft", ORANGE),
        ("GRPO  t=0.7, G=8", "ph3n25p29_lo_a1.0_b0.0", BLUE),
        ("GRPO  t=1.0, G=16", "ph3n25p29_mid_a1.0_b0.0", BLUE),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.6), dpi=200)
    for ax, split, title, tot in [
        (axes[0], "eval", "Held-out: 20 problems base cannot solve at k=1024", 20),
        (axes[1], "train", "Train prompts: 42 such problems", 42),
    ]:
        names = [r[0] for r in rows]
        vals = [tr[f"{key}_{split}"]["newly_reached"] for _, key, _ in rows]
        cols = [r[2] for r in rows]
        y = np.arange(len(rows))[::-1]
        ax.barh(y, vals, height=0.55, color=cols)
        ax.barh(y, [tot] * len(rows), height=0.55, color="none", edgecolor=LINE, lw=1)
        for yi, v in zip(y, vals):
            ax.text(v + tot * 0.015, yi, f"{v} / {tot}", va="center", fontsize=12.5, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=12.5, color=INK)
        ax.set_xlim(0, tot * 1.14)
        ax.set_title(title, fontsize=13.5)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(LINE)
        ax.set_xlabel("newly reached within 64 attempts")
    fig.tight_layout()
    fig.savefig(f"{OUT}/h1_boundary_moves.png")
    plt.close(fig)


# ---------------------------------------------------------------- chart 4
# Reduced-but-nonzero exploration: hard & easy, xlo & lo
def chart_exploration():
    hard = {
        "Base": (0.029, 0.325),
        "X-low exploration\nt=0.3, G=4": (0.375, 0.975),
        "Low exploration\nt=0.7, G=8": (0.880, 1.000),
    }
    easy = {
        "Base": (0.282, 0.870),
        "X-low exploration\nt=0.3, G=4": (0.799, 1.000),
        "Low exploration\nt=0.7, G=8": (0.948, 1.000),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0), dpi=200, sharey=True)
    for ax, data, title in [
        (axes[0], hard, "Hard task (n≤5, p≤29), held-out"),
        (axes[1], easy, "Easy task (n=2, p≤13), train primes"),
    ]:
        x = np.arange(len(data))
        p1 = [v[0] for v in data.values()]
        p64 = [v[1] for v in data.values()]
        ax.bar(x - 0.19, p1, width=0.34, color=BLUE, label="pass@1")
        ax.bar(x + 0.19, p64, width=0.34, color="#a5c3f6", label="pass@64")
        for xi, (a, b) in zip(x, zip(p1, p64)):
            ax.text(xi - 0.19, a + 0.02, f"{a:.2f}", ha="center", fontsize=11, color=INK)
            ax.text(xi + 0.19, b + 0.02, f"{b:.2f}", ha="center", fontsize=11, color=MUT)
        ax.set_xticks(x)
        ax.set_xticklabels(data.keys(), fontsize=12)
        ax.set_title(title, fontsize=13.5)
        ax.set_ylim(0, 1.12)
        style(ax)
    axes[0].set_ylabel("score")
    axes[0].legend(frameon=False, loc="upper left", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/h1_exploration.png")
    plt.close(fig)


# ---------------------------------------------------------------- chart 5
# Asymmetry seeds + forgetting (status slide)
def chart_asym():
    cells = [
        ("α=0.6, β=0", GREEN, [0.501, 0.535, 0.505]),
        ("α=0.8, β=0.2", ORANGE, [0.062, 0.160]),
        ("α=1.0, β=0.4", ORANGE, [0.147, 0.069, 0.114]),
    ]
    forg = [
        ("β = 0 cells", GREEN, [0, 0, 0, 1, 3, 4]),
        ("β > 0 cells", ORANGE, [12, 20, 13, 18, 25, 19, 17]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0), dpi=200)
    ax = axes[0]
    for i, (name, c, vals) in enumerate(cells):
        ax.scatter([i] * len(vals), vals, s=90, color=c, zorder=3)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([c[0] for c in cells], fontsize=12.5)
    ax.set_ylabel("held-out pass@1 (per seed)")
    ax.set_title("Same margin α−β=0.6, different error mix", fontsize=13.5)
    ax.set_ylim(0, 0.6)
    style(ax)
    ax.annotate("margin 1.0 point\nshown for β mix only", xy=(0, 0), fontsize=1, alpha=0)
    ax = axes[1]
    for i, (name, c, vals) in enumerate(forg):
        ax.scatter([i] * len(vals), vals, s=90, color=c, zorder=3)
    ax.set_xticks(range(len(forg)))
    ax.set_xticklabels([f[0] for f in forg], fontsize=12.5)
    ax.set_ylabel("base-solved problems forgotten (of 65)")
    ax.set_title("False positives destroy previously solved tasks", fontsize=13.5)
    ax.set_ylim(-1, 28)
    style(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/asymmetry_forgetting.png")
    plt.close(fig)


chart_boundary()
chart_offline()
chart_offline_v2()
chart_offline_v3()
chart_sft_boundary()
chart_newly()
chart_exploration()
chart_asym()
print("charts written")
