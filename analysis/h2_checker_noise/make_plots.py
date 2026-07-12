"""Build the full set of H2/G2 + curriculum figures from the TensorBoard events.

  python analysis/h2_checker_noise/make_plots.py

Every curve is held-out (eval/*) unless noted; train/* is shown only where the
train-vs-eval or reward-vs-truth gap is the point being made.
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


# ---- regime coloring by informativeness alpha-beta -------------------------
def regime(d):
    d = round(d, 4)
    if d >= 0.4:
        return "sharpen", "#1a9850"
    if d >= 0.15:
        return "marginal", "#fdae61"
    if d <= -0.05:
        return "degrade", "#d73027"
    return "collapse", "#4575b4"


def load(run_dir):
    """All scalars for a run -> {tag: (steps, vals)}."""
    evs = glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True)
    out = {}
    for ev in sorted(evs):
        ea = EventAccumulator(os.path.dirname(ev))
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            s = ea.Scalars(tag)
            steps = [x.step for x in s]
            vals = [x.value for x in s]
            if tag in out:
                out[tag] = (out[tag][0] + steps, out[tag][1] + vals)
            else:
                out[tag] = (steps, vals)
    for tag, (st, vv) in out.items():
        order = sorted(range(len(st)), key=lambda i: st[i])
        out[tag] = ([st[i] for i in order], [vv[i] for i in order])
    return out


def parse_ab(name):
    ab = {}
    for part in name.split("_"):
        if part and part[0] in "ab" and part[1:].replace(".", "", 1).isdigit():
            ab[part[0]] = float(part[1:])
    return ab.get("a"), ab.get("b")


EVAL_TRUE = "eval/rewards/true_accuracy/mean"
EVAL_NOISY = "eval/rewards/reward_noisy_checker/mean"
TRAIN_TRUE = "train/rewards/true_accuracy/mean"
TRAIN_NOISY = "train/rewards/reward_noisy_checker/mean"
ENTROPY = "train/entropy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sweep = {}
    for rd in sorted(glob.glob(os.path.join(args.runs_dir, "a*_p29_*"))):
        a, b = parse_ab(os.path.basename(rd))
        if a is None:
            continue
        sweep[(a, b)] = load(rd)
    keys = sorted(sweep, key=lambda k: (k[0] - k[1]), reverse=True)

    # ---------- FIG 1: held-out true_accuracy curves, all configs -----------
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for a, b in keys:
        d = a - b
        _, col = regime(d)
        st, vv = sweep[(a, b)].get(EVAL_TRUE, ([], []))
        ax.plot(st, vv, color=col, lw=2, marker="o", ms=3, label=f"α={a} β={b}  (α−β={d:+.1f})")
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("held-out true accuracy")
    ax.set_title("H2: held-out accuracy vs training step, by checker (α, β)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(f"{args.out}/eval_accuracy_curves.png")
    plt.close(fig)

    # ---------- FIG 2: reward(noisy) vs truth gap, small multiples ----------
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True, sharey=True)
    for ax, (a, b) in zip(axes.flat, keys):
        s = sweep[(a, b)]
        _, col = regime(a - b)
        st, tv = s.get(EVAL_TRUE, ([], []))
        sn, nv = s.get(EVAL_NOISY, ([], []))
        ax.plot(sn, nv, color="#888", lw=1.8, label="checker reward")
        ax.plot(st, tv, color=col, lw=2.2, label="true accuracy")
        ax.set_title(f"α={a} β={b}", fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("step")
    for ax in axes[:, 0]:
        ax.set_ylabel("held-out")
    axes.flat[-1].axis("off")
    fig.suptitle(
        "Checker reward (what the loop optimizes) vs true accuracy — "
        "the gap is the verifier bottleneck",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(f"{args.out}/reward_vs_true_accuracy.png")
    plt.close(fig)

    # ---------- FIG 3: phase — Delta true_acc vs (alpha-beta) ---------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for a, b in keys:
        st, vv = sweep[(a, b)].get(EVAL_TRUE, ([], []))
        if not vv:
            continue
        start = sum(vv[:3]) / len(vv[:3])
        final = sum(vv[-3:]) / len(vv[-3:])
        d = a - b
        name, col = regime(d)
        ax.scatter([d], [final - start], s=140, color=col, zorder=3, edgecolor="k")
        ax.annotate(
            f"({a},{b})", (d, final - start), textcoords="offset points", xytext=(6, 4), fontsize=8
        )
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("checker informativeness  α − β")
    ax.set_ylabel("Δ held-out true accuracy (final − start)")
    ax.set_title("G2 phase diagram: self-improvement vs checker informativeness")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=c, label=n)
        for n, c in [
            ("sharpen", "#1a9850"),
            ("marginal", "#fdae61"),
            ("collapse", "#4575b4"),
            ("degrade", "#d73027"),
        ]
    ]
    ax.legend(handles=handles, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.out}/accuracy_gain_vs_checker_signal.png")
    plt.close(fig)

    # ---------- FIG 4: final held-out accuracy bar -------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    labels, finals, cols = [], [], []
    for a, b in keys:
        st, vv = sweep[(a, b)].get(EVAL_TRUE, ([], []))
        if not vv:
            continue
        labels.append(f"{a}/{b}\nα−β={a - b:+.1f}")
        finals.append(sum(vv[-3:]) / len(vv[-3:]))
        cols.append(regime(a - b)[1])
    ax.bar(range(len(finals)), finals, color=cols)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("final held-out true accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Final held-out accuracy by checker (α, β)")
    fig.tight_layout()
    fig.savefig(f"{args.out}/final_accuracy_by_checker.png")
    plt.close(fig)

    # ---------- FIG 5: policy entropy vs step ------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for a, b in keys:
        st, vv = sweep[(a, b)].get(ENTROPY, ([], []))
        if not vv:
            continue
        ax.plot(st, vv, color=regime(a - b)[1], lw=1.8, label=f"α={a} β={b}")
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("policy entropy (train)")
    ax.set_title(
        "Policy entropy: sharpening (informative checker) vs no collapse of uncertainty (α≈β)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.out}/policy_entropy.png")
    plt.close(fig)

    # ---------- FIG 6: curriculum ladder (stitched) ------------------------
    ladder = [13, 23, 37, 59, 97]
    fig, ax = plt.subplots(figsize=(9, 5))
    offset = 0
    boundaries = []
    for pm in ladder:
        rd = os.path.join(args.runs_dir, f"curric_p{pm}_s0")
        if not os.path.isdir(rd):
            continue
        s = load(rd)
        st, vv = s.get(EVAL_TRUE, ([], []))
        if not vv:
            continue
        xs = [x + offset for x in st]
        ax.plot(xs, vv, lw=2, marker="o", ms=3, label=f"p≤{pm}")
        mid = offset + (st[-1] if st else 0) / 2
        ax.text(mid, 1.03, f"p≤{pm}", ha="center", fontsize=9)
        offset += st[-1] if st else 0
        boundaries.append(offset)
    for bx in boundaries[:-1]:
        ax.axvline(bx, color="k", lw=0.6, ls=":")
    ax.axhline(0.02, color="#d73027", lw=1, ls="--", label="cold RL-from-base ≈ 0 (p≤97)")
    ax.set_xlabel("cumulative GRPO step (stages concatenated)")
    ax.set_ylabel("held-out true accuracy")
    ax.set_title("Curriculum (perfect judge): climbing the modulus ladder 13→97")
    ax.set_ylim(-0.02, 1.1)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{args.out}/curriculum_accuracy.png")
    plt.close(fig)

    # ---------- FIG 7: curriculum per-stage final bar ----------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    xs, ys = [], []
    for pm in ladder:
        rd = os.path.join(args.runs_dir, f"curric_p{pm}_s0")
        if not os.path.isdir(rd):
            continue
        st, vv = load(rd).get(EVAL_TRUE, ([], []))
        if not vv:
            continue
        xs.append(f"p≤{pm}")
        ys.append(sum(vv[-3:]) / len(vv[-3:]))
    ax.bar(range(len(ys)), ys, color="#1a9850")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("final held-out true accuracy")
    ax.set_title("Curriculum: held-out accuracy stays ≥0.98 up the ladder")
    for i, y in enumerate(ys):
        ax.text(i, y + 0.01, f"{y:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{args.out}/curriculum_final_accuracy.png")
    plt.close(fig)

    print("wrote figs 1-7 to", args.out)


if __name__ == "__main__":
    main()
