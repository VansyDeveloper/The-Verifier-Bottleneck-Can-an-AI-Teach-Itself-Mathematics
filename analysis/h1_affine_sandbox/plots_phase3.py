"""Phase-diagram figures (G1/G2/H1) from results_remote/ph3n25p29/*.json.
Notation: k = attempts (pass@k), n = composition depth ~ U[2,5], primes p<=29.
Splits: 'eval' = held-out primes {11,17}; 'train' = train primes, fresh instances.

FIG A: pass@k curves (k=1..64), base vs 9 trained cells, per budget (eval split).
FIG B: 3x3 heatmaps of dpass@1 / dpass@64 + regimes, one row of panels per split.
FIG C: dpass@64 vs exploration budget per verifier margin (G1), both splits.
"""

import json
import math
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 140, "font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

BUDGETS = ["lo", "mid", "hi"]
BUDGET_LAB = {"lo": "low (t=0.7, G=8)", "mid": "default (t=1.0, G=16)", "hi": "high (t=1.3, G=32)"}
MARGINS = [("1.0", "0.0"), ("0.8", "0.2"), ("0.6", "0.4")]
SPLITS = ["eval", "train"]
SPLIT_LAB = {"eval": "held-out primes {11,17}", "train": "train primes, new instances"}
KS = [1, 2, 4, 8, 16, 32, 64]


def passk_from_json(path, ks):
    d = json.load(open(path))
    vals = []
    for K in ks:
        acc = []
        for pp in d["per_problem"]:
            n, c = pp["n"], pp["n_correct"]
            acc.append(1.0 - math.comb(n - c, K) / math.comb(n, K) if n - c >= K else 1.0)
        vals.append(float(np.mean(acc)))
    return np.array(vals)


base = {s: passk_from_json(f"results_remote/ph3n25p29/base_0.6B_{s}.json", KS) for s in SPLITS}
cells = {}
for s in SPLITS:
    for b in BUDGETS:
        for a, be in MARGINS:
            p = f"results_remote/ph3n25p29/ph3n25p29_{b}_a{a}_b{be}_{s}.json"
            if os.path.exists(p):
                cells[(s, b, a)] = passk_from_json(p, KS)
            else:
                print("MISSING", p)

# ---------- FIG A: pass@k curves (eval split) ----------
mcols = {"1.0": "#1a9850", "0.8": "#f46d43", "0.6": "#762a83"}
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
for ax, b in zip(axes, BUDGETS):
    ax.plot(KS, base["eval"], color="k", lw=2.6, marker="o", label="base 0.6B")
    for a, be in MARGINS:
        if ("eval", b, a) in cells:
            ax.plot(
                KS,
                cells[("eval", b, a)],
                color=mcols[a],
                lw=2.0,
                marker="s",
                ms=4,
                label=f"RL  α−β={float(a) - float(be):+.1f}",
            )
    ax.set_xscale("log", base=2)
    ax.set_xticks(KS)
    ax.set_xticklabels(KS)
    ax.set_xlabel("k (attempts)")
    ax.set_title(BUDGET_LAB[b])
    ax.legend(fontsize=8, loc="upper left")
axes[0].set_ylabel("pass@k (held-out primes, n~U[2,5])")
fig.suptitle("pass@k: base vs GRPO across exploration budgets (150 steps, eval t=1.0, p∈{11,17})")
fig.tight_layout()
fig.savefig("results/figures/passk_by_checker_and_budget.png")
plt.close(fig)
print("wrote passk_by_checker_and_budget.png")


def regime(dp1, dp64, eps=0.03):
    if np.isnan(dp1):
        return ""
    if dp64 < -eps:
        return "collapse"
    if dp64 > eps:
        return "expand"
    if dp1 > eps:
        return "sharpen"
    return "stall"


# ---------- FIG B: heatmaps + regimes, per split ----------
i1, i64 = KS.index(1), KS.index(64)
D = {}
for s in SPLITS:
    d1 = np.full((3, 3), np.nan)
    d64 = np.full((3, 3), np.nan)
    for r, b in enumerate(BUDGETS):
        for c, (a, be) in enumerate(MARGINS):
            if (s, b, a) in cells:
                d1[r, c] = cells[(s, b, a)][i1] - base[s][i1]
                d64[r, c] = cells[(s, b, a)][i64] - base[s][i64]
    D[s] = (d1, d64)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
for row, s in enumerate(SPLITS):
    d1, d64 = D[s]
    for col, (mat, ttl, vmax, cmap) in enumerate(
        [
            (d1, f"Δ pass@1 — {SPLIT_LAB[s]}", 0.8, "YlGn"),
            (d64, f"Δ pass@64 — {SPLIT_LAB[s]}", 0.5, "RdBu_r"),
        ]
    ):
        ax = axes[row, col]
        im = ax.imshow(mat, cmap=cmap, vmin=(-vmax if cmap == "RdBu_r" else 0), vmax=vmax)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{float(a) - float(b):+.1f}" for a, b in MARGINS])
        ax.set_yticks(range(3))
        ax.set_yticklabels([BUDGET_LAB[b] for b in BUDGETS])
        ax.set_xlabel("verifier margin α−β")
        ax.set_title(ttl, fontsize=10)
        for r in range(3):
            for c in range(3):
                if not np.isnan(mat[r, c]):
                    ax.annotate(f"{mat[r, c]:+.2f}", (c, r), ha="center", va="center", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle(
    "Phase diagram (G2): verifier informativeness × exploration budget, Δ vs base "
    "(n~U[2,5], p≤29; классификация режимов — по CI в таблице §3)"
)
fig.tight_layout()
fig.savefig("results/figures/checker_budget_phase_diagram.png")
plt.close(fig)
print("wrote checker_budget_phase_diagram.png")

# ---------- FIG C: dpass@64 vs budget per margin (G1) ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
for ax, s in zip(axes, SPLITS):
    d1, d64 = D[s]
    for c, (a, be) in enumerate(MARGINS):
        ax.plot(
            [0, 1, 2],
            [d64[r, c] for r in range(3)],
            marker="o",
            lw=2.4,
            color=mcols[a],
            label=f"α−β={float(a) - float(be):+.1f}",
        )
    ax.axhline(0, color="k", lw=1, ls=":")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["lo", "mid", "hi"])
    ax.set_xlabel("exploration budget")
    ax.set_title(SPLIT_LAB[s])
axes[0].set_ylabel("Δ pass@64 vs base (expansion signal)")
axes[1].legend()
fig.suptitle("G1: does exploration budget buy expansion?")
fig.tight_layout()
fig.savefig("results/figures/exploration_vs_budget.png")
plt.close(fig)
print("wrote exploration_vs_budget.png")

# ---------- table ----------
for s in SPLITS:
    d1, d64 = D[s]
    print(f"\n[{SPLIT_LAB[s]}]")
    print("cell            pass@1  pass@64  d1      d64     regime")
    print(f"base            {base[s][i1]:.3f}   {base[s][i64]:.3f}")
    for r, b in enumerate(BUDGETS):
        for c, (a, be) in enumerate(MARGINS):
            if (s, b, a) in cells:
                v = cells[(s, b, a)]
                print(
                    f"{b:4s} m={float(a) - float(be):+.1f}     {v[i1]:.3f}   {v[i64]:.3f}  "
                    f"{d1[r, c]:+.3f}  {d64[r, c]:+.3f}  {regime(d1[r, c], d64[r, c])}"
                )
