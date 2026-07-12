"""Polished figures for the converged variable-depth run (k up to 10).

  python analysis/depth_generalization/depth_plots.py --run runs/<run> --tag conv

Produces:
  fig_conv_learning.png    held-out true_acc + train reward, entropy, length vs step
  fig_conv_depth.png       accuracy vs depth k, base(early) vs converged(late), +/-SE
  fig_conv_depth_time.png  per-depth accuracy over training (the depth penalty vanishing)
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from depth_analysis import depth_of

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def tb(run):
    evs = glob.glob(os.path.join(run, "**", "events.out.tfevents.*"), recursive=True)
    out = {}
    for ev in sorted(evs):
        ea = EventAccumulator(os.path.dirname(ev))
        ea.Reload()
        for t in ea.Tags().get("scalars", []):
            s = ea.Scalars(t)
            out.setdefault(t, ([], []))
            out[t][0].extend(x.step for x in s)
            out[t][1].extend(x.value for x in s)
    for t, (st, vv) in out.items():
        o = np.argsort(st)
        out[t] = (np.array(st)[o], np.array(vv)[o])
    return out


def load_pq(run):
    pqs = sorted(glob.glob(os.path.join(run, "completions", "*.parquet")))
    df = pd.concat([pd.read_parquet(p) for p in pqs], ignore_index=True)
    df["k"] = df.prompt.map(depth_of)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--tag", default="conv")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    T = tb(args.run)
    df = load_pq(args.run)
    smax = int(df.step.max())

    # ---------- FIG A: learning / convergence ----------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    if "eval/rewards/true_accuracy/mean" in T:
        st, vv = T["eval/rewards/true_accuracy/mean"]
        ax1.plot(st, vv, "-o", ms=3, color="#1a9850", lw=2, label="held-out true accuracy")
    if "train/rewards/true_accuracy/mean" in T:
        st, vv = T["train/rewards/true_accuracy/mean"]
        ax1.plot(st, vv, color="#8888ff", lw=1, alpha=0.6, label="train true accuracy (rollout)")
    ax1.set_ylabel("accuracy")
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_title(f"Convergence — {os.path.basename(args.run)} (k∼U(2,10), p≤23, perfect judge)")
    ax1.legend(fontsize=8, loc="lower right")

    if "train/entropy" in T:
        st, vv = T["train/entropy"]
        ax2.plot(st, vv, color="#d73027", lw=1.5, label="policy entropy")
    ax2.set_ylabel("entropy", color="#d73027")
    ax2.tick_params(axis="y", labelcolor="#d73027")
    axr = ax2.twinx()
    if "train/completions/mean_length" in T:
        st, vv = T["train/completions/mean_length"]
        axr.plot(st, vv, color="#4575b4", lw=1.5, label="mean completion length")
    axr.set_ylabel("mean completion length", color="#4575b4")
    axr.tick_params(axis="y", labelcolor="#4575b4")
    axr.grid(False)
    ax2.set_xlabel("GRPO step")
    fig.tight_layout()
    fig.savefig(f"{args.out}/{args.tag}_learning_curve.png")
    plt.close(fig)

    # ---------- FIG B: accuracy vs depth, base vs converged ----------
    early = df[df.step <= max(5, int(smax * 0.03))]
    late = df[df.step >= int(smax * 0.9)]
    ks = sorted(k for k in df.k.unique() if k >= 2)

    def by_k(d):
        m, se, n = [], [], []
        for k in ks:
            v = d[d.k == k].true_accuracy.values
            m.append(v.mean() if len(v) else np.nan)
            se.append(v.std() / max(1, np.sqrt(len(v))) if len(v) else 0)
            n.append(len(v))
        return np.array(m), np.array(se), n

    em, ese, _ = by_k(early)
    lm, lse, ln = by_k(late)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        ks,
        em,
        yerr=ese,
        fmt="o--",
        color="#999",
        lw=1.8,
        capsize=3,
        label=f"base (step ≤ {max(5, int(smax * 0.03))})",
    )
    ax.errorbar(
        ks,
        lm,
        yerr=lse,
        fmt="o-",
        color="#1a9850",
        lw=2.4,
        capsize=3,
        label=f"converged (step ≥ {int(smax * 0.9)})",
    )
    for k, y in zip(ks, lm):
        ax.annotate(
            f"{y:.2f}",
            (k, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="#1a7a3f",
        )
    ax.set_xlabel("composition depth  k  (number of functions g_1..g_k)")
    ax.set_ylabel("true accuracy (rollouts)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(ks)
    ax.set_title("Accuracy vs composition depth: RL removes the depth penalty")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{args.out}/{args.tag}_accuracy_by_depth.png")
    plt.close(fig)

    # ---------- FIG C: per-depth accuracy over training ----------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    g = df.groupby(["step", "k"]).true_accuracy.mean().reset_index()
    cmap = plt.cm.viridis(np.linspace(0, 0.92, len(ks)))
    for k, c in zip(ks, cmap):
        s = g[g.k == k].sort_values("step")
        if len(s) < 3:
            continue
        roll = s.true_accuracy.rolling(window=25, min_periods=5, center=True).mean()
        ax.plot(s.step, roll, color=c, lw=1.9, label=f"k={k}")
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("true accuracy (25-step rolling)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Per-depth accuracy over training: deep chains catch up to shallow")
    ax.legend(fontsize=8, ncol=2, title="depth")
    fig.tight_layout()
    fig.savefig(f"{args.out}/{args.tag}_accuracy_by_depth_over_time.png")
    plt.close(fig)

    print("converged depth breakdown (late):")
    for k, y, n in zip(ks, lm, ln):
        print(f"  k={k:2d}  acc={y:.3f}  (n={n})")
    print("wrote 3 figs with tag", args.tag)


if __name__ == "__main__":
    main()
