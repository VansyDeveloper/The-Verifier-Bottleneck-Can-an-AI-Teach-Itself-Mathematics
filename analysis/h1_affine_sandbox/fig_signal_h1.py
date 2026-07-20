"""Two extra report figures:
fig_signal_density.png — precision-of-accepted vs training step + final accuracy,
    off-diagonal (od1/od2) vs diagonal cells, same G=16/t=1.0 budget.
fig_h1_tests.png — H1 targeted tests: base vs xlo/lo on hard and easy task.
"""

import json
import numpy as np
import matplotlib.pyplot as plt

sig = json.load(open("results_remote/posthoc_signal.json"))

CELLS = [
    ("ph3n25p29_mid_a1.0_b0.0", r"$\alpha$=1.0, $\beta$=0.0 (m=1.0)", "#1a7a1a", "-"),
    ("ph3n25p29_od1_a0.6_b0.0", r"$\alpha$=0.6, $\beta$=0.0 (m=0.6)", "#66bb44", "-"),
    ("ph3n25p29_od2_a1.0_b0.4", r"$\alpha$=1.0, $\beta$=0.4 (m=0.6)", "#dd8833", "--"),
    ("ph3n25p29_mid_a0.8_b0.2", r"$\alpha$=0.8, $\beta$=0.2 (m=0.6)", "#cc4444", "--"),
    ("ph3n25p29_mid_a0.6_b0.4", r"$\alpha$=0.6, $\beta$=0.4 (m=0.2)", "#884444", ":"),
    ("ph3n25p29_ctlh_a0.5_b0.5", r"$\alpha$=$\beta$=0.5 (m=0)", "#888888", ":"),
]


def smooth(x, w=9):
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)
    xf = np.where(mask, x, 0.0)
    k = np.ones(w)
    num = np.convolve(xf, k, "same")
    den = np.convolve(mask.astype(float), k, "same")
    return np.where(den > 0, num / den, np.nan)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for run, label, c, ls in CELLS:
    rows = sig[run]
    steps = [r["step"] for r in rows]
    prec = smooth([r["precision"] if r["precision"] is not None else np.nan for r in rows])
    a = smooth([r["a"] for r in rows])
    axes[0].plot(steps, prec, color=c, ls=ls, label=label)
    axes[1].plot(steps, a, color=c, ls=ls, label=label)
axes[0].set_title("precision принятых: $P(\\mathrm{correct}\\,|\\,\\mathrm{accepted})$")
axes[1].set_title("точность политики на train-батче $a_t$")
for ax in axes:
    ax.set_xlabel("шаг GRPO")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=7.5, loc="center right")
fig.suptitle(
    "Margin недостаточен: при одинаковом m=0.6 судьба обучения разная — критичен состав ошибок ($\\beta$ vs $\\alpha$)"
)
fig.tight_layout()
fig.savefig("results/figures/verifier_signal_density.png", dpi=130)
plt.close(fig)
print("wrote fig_signal_density.png")


# ---- H1 figure ----
def pk(path, key):
    j = json.load(open(path))
    return j[key]


R = "results_remote"
groups = [
    ("hard, base", f"{R}/ph3n25p29/base_0.6B_eval.json", None),
    ("hard, xlo\n(t=0.3, G=4)", f"{R}/ph3n25p29/ph3n25p29_xlo_a1.0_b0.0_eval.json", None),
    (
        "hard, lo\n(t=0.7, G=8)",
        f"{R}/ph3n25p29/ph3n25p29_lo_a1.0_b0.0_eval.json",
        [
            f"{R}/ph3n25p29/ph3n25p29_loS1_a1.0_b0.0_eval.json",
            f"{R}/ph3n25p29/ph3n25p29_loS2_a1.0_b0.0_eval.json",
        ],
    ),
    ("easy, base", f"{R}/ph3easy/base_0.6B_train.json", None),
    (
        "easy, xlo",
        f"{R}/ph3easy/ph3easy_xlo_a1.0_b0.0_train.json",
        [
            f"{R}/ph3easy/ph3easy_xloS1_a1.0_b0.0_train.json",
            f"{R}/ph3easy/ph3easy_xloS2_a1.0_b0.0_train.json",
        ],
    ),
    (
        "easy, lo",
        f"{R}/ph3easy/ph3easy_lo_a1.0_b0.0_train.json",
        [f"{R}/ph3easy/ph3easy_loS1_a1.0_b0.0_train.json"],
    ),
]

fig, ax = plt.subplots(figsize=(9, 4))
xs = np.arange(len(groups))
w = 0.38
for i, (label, path, seeds) in enumerate(groups):
    p1, p64 = pk(path, "pass@1"), pk(path, "pass@64")
    ax.bar(i - w / 2, p1, w, color="#3366cc", label="pass@1" if i == 0 else None)
    ax.bar(i + w / 2, p64, w, color="#99bbee", label="pass@64" if i == 0 else None)
    if seeds:
        for sp in seeds:
            ax.plot([i - w / 2], [pk(sp, "pass@1")], "k_", ms=14)
            ax.plot([i + w / 2], [pk(sp, "pass@64")], "k_", ms=14)
ax.set_xticks(xs, [g[0] for g in groups], fontsize=8.5)
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)
ax.legend()
ax.set_title(
    "H1-тест: сниженный exploration (xlo), надёжный чекер — pass@64 растёт, а не только pass@1\n"
    "(hard: n∈[2,5], p≤29, held-out p; easy: n=2, p≤13, train-p; чёрные штрихи — сиды)",
    fontsize=9,
)
fig.tight_layout()
fig.savefig("results/figures/offline_vs_onpolicy.png", dpi=130)
plt.close(fig)
print("wrote fig_h1_tests.png")
