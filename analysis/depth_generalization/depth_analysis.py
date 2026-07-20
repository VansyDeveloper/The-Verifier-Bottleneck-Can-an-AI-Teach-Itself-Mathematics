"""Accuracy as a function of composition depth k, from logged rollout parquets.

The task mixes k~U(k_min,k_max) functions per problem; this bins the noiseless
true_accuracy by k so we can see how deep the model can compose before it breaks.

  python analysis/depth_generalization/depth_analysis.py --run runs/<run> [--tag mix]
"""

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

G_LINE = re.compile(r"^g_\d+\(x\)", re.M)


def depth_of(prompt):
    """k = number of g_i definitions in the final (last) problem block."""
    idx = prompt.rfind("Problem:")
    block = prompt[idx:] if idx >= 0 else prompt
    return len(G_LINE.findall(block))


def load(run):
    pqs = sorted(glob.glob(os.path.join(run, "completions", "*.parquet")))
    df = pd.concat([pd.read_parquet(p) for p in pqs], ignore_index=True)
    df["k"] = df.prompt.map(depth_of)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--tag", default="mix")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load(args.run)
    smax = int(df.step.max())
    early = df[df.step <= max(5, int(smax * 0.05))]
    late = df[df.step >= int(smax * 0.85)]

    ks = sorted(k for k in df.k.unique() if k >= 2)
    e_acc = [early[early.k == k].true_accuracy.mean() for k in ks]
    l_acc = [late[late.k == k].true_accuracy.mean() for k in ks]
    n_late = [int((late.k == k).sum()) for k in ks]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(
        ks, e_acc, "o--", color="#999", lw=1.8, label=f"early (step ≤ {max(5, int(smax * 0.05))})"
    )
    ax.plot(ks, l_acc, "o-", color="#1a9850", lw=2.2, label=f"late (step ≥ {int(smax * 0.85)})")
    for k, y, n in zip(ks, l_acc, n_late):
        ax.annotate(
            f"n={n}",
            (k, y),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=7,
            color="#1a9850",
        )
    ax.set_xlabel("composition depth  k  (number of functions g_1..g_k)")
    ax.set_ylabel("true accuracy (rollouts)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(ks)
    ax.set_title(f"Accuracy vs composition depth — {os.path.basename(args.run)}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(args.out, f"depth_accuracy_{args.tag}.png")
    fig.savefig(out)
    plt.close(fig)

    print("depth breakdown (late):")
    for k, y, n in zip(ks, l_acc, n_late):
        print(f"  k={k:2d}  acc={y:.3f}  (n={n})")
    print("wrote", out)


if __name__ == "__main__":
    main()
