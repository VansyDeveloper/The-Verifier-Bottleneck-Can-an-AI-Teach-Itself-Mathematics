"""Plot held-out pass@1 / pass@k as a function of composition depth k, from the
eval_passk JSON dumps (which record k_funcs + n_correct per problem).

  python analysis/depth_generalization/eval_depth_plot.py
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def load(path):
    with open(path) as f:
        d = json.load(f)
    p1, pk = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for r in d["per_problem"]:
        k = r["k_funcs"]
        p1[k][0] += r["n_correct"]
        p1[k][1] += r["n"]
        pk[k][0] += int(r["n_correct"] > 0)
        pk[k][1] += 1
    ks = sorted(p1)
    pass1 = [p1[k][0] / max(1, p1[k][1]) for k in ks]
    passk = [pk[k][0] / max(1, pk[k][1]) for k in ks]
    return ks, pass1, passk, d.get("k")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/data/depth")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    data = {}
    for f in glob.glob(os.path.join(args.dir, "*.json")):
        name = os.path.splitext(os.path.basename(f))[0]  # e.g. p13_train
        data[name] = load(f)

    def get(name):
        return data.get(name)

    # ---- FIG A: same-prime (train split) pass@1 vs depth, p13 vs p23 ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, col, lab in [
        ("p13_train", "#1a9850", "p≤13 (reliable arithmetic)"),
        ("p23_train", "#4575b4", "p≤23 (harder arithmetic)"),
    ]:
        d = get(name)
        if not d:
            continue
        ks, p1, _, _ = d
        ax.plot(ks, p1, "o-", color=col, lw=2.4, label=lab)
        for k, y in zip(ks, p1):
            ax.annotate(
                f"{y:.2f}",
                (k, y),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=7,
                color=col,
            )
    ax.set_xlabel("composition depth  k")
    ax.set_ylabel("held-out pass@1")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(range(2, 11))
    ax.set_title("No depth wall: pass@1 flat across k=2..10 (held-out, disjoint instances)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{args.out}/depth_pass1.png")
    plt.close(fig)

    # ---- FIG B: transfer gap — same-prime vs cross-prime (p23) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, col, lab in [
        ("p23_train", "#4575b4", "same primes, new instances"),
        ("p23_eval", "#d73027", "held-out primes (cross-prime transfer)"),
    ]:
        d = get(name)
        if not d:
            continue
        ks, p1, _, _ = d
        ax.plot(ks, p1, "o-", color=col, lw=2.2, label=lab)
    ax.set_xlabel("composition depth  k")
    ax.set_ylabel("pass@1 (p≤23)")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(range(2, 11))
    ax.set_title("The bottleneck is arithmetic transfer, not depth")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{args.out}/depth_transfer.png")
    plt.close(fig)

    # ---- FIG C: pass@k vs depth (same-prime) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, col, lab in [("p13_train", "#1a9850", "p≤13"), ("p23_train", "#4575b4", "p≤23")]:
        d = get(name)
        if not d:
            continue
        ks, p1, pk, kk = d
        ax.plot(ks, pk, "o--", color=col, lw=1.6, alpha=0.7, label=f"{lab} pass@{kk}")
        ax.plot(ks, p1, "o-", color=col, lw=2.2, label=f"{lab} pass@1")
    ax.set_xlabel("composition depth  k")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(range(2, 11))
    ax.set_title("pass@1 vs pass@k by depth (held-out, same primes)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.out}/depth_passk.png")
    plt.close(fig)

    print("by-depth summary:")
    for name in sorted(data):
        ks, p1, pk, kk = data[name]
        print(
            f"  {name}: pass@1 k2={p1[0]:.2f} k10={p1[-1]:.2f} | "
            f"pass@{kk} k2={pk[0]:.2f} k10={pk[-1]:.2f}"
        )
    print("wrote fig_bydepth_{pass1,transfer,passk}.png")


if __name__ == "__main__":
    main()
