"""All figures for the paper, in one style.

Kept separate from the report plotting scripts so that
`plot_confirmatory_2026_08_01.py` and `plot_ranking_results.py` keep producing
byte-identical output.

Two groups of figures:

  confirmatory series - the six-run series. Its per-run values, transfer effects
                      and per-run probability-mass gains are read from its own
                      analysis file; only the three rows of absolute atomic
                      accuracy are transcribed, because the archive stores drops
                      rather than absolutes. The t-interval and t-statistic are
                      recomputed here from the per-run values instead of quoted,
                      and asserted against the values that file reports.
  this series       - read from artifacts/reports/*.json, never typed in.

Titles are omitted on purpose: the claim belongs in the caption. The text block
of the journal class is 321pt, so every figure is 4.6in wide or less and no
figure exceeds about half the text height.

Output: artifacts/reports/paper/fig[2-7]*.pdf. Figure 1 is drawn in TikZ
inside main.tex so that it uses the document's own type.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPORTS = Path("artifacts/reports")
OUT = REPORTS / "paper"
RUNS = Path("artifacts/runs")

# One palette for the whole paper.
#
# The three categorical hues are validated slots: run
#   node validate_palette.js "#2a78d6,#eb6834,#4a3aa7" --mode light
# and every check passes, including the all-pairs normal-vision floor. C_BASE is
# deliberately a neutral and not a fourth categorical hue: it always marks the
# reference arm, never an identity that has to be told apart from another. It
# therefore fails the chroma floor by design and passes every separation check.
C_MAIN = "#2a78d6"   # the treatment under discussion
C_ALT = "#eb6834"    # a second treatment, or a negative direction
C_GOOD = "#4a3aa7"   # the control that succeeds
C_BASE = "#898781"   # neutral: the reference arm
C_FAINT = "#c9d7ea"  # a lighter step of the same blue ramp, for a second bar
INK = "#52514e"      # secondary ink, for value labels
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
WIDTH = 4.6

STYLE = {
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "axes.edgecolor": "#c3c2b7",
    "axes.linewidth": 0.7,
    "xtick.color": "#898781",
    "ytick.color": "#898781",
    "axes.labelcolor": "#52514e",
    "text.color": "#0b0b0b",
    "figure.dpi": 200,
}

# ------------------------------------------------- confirmatory series, read from disk
# The six-run series is analysed in its own archive. Its per-run values, its
# secondary transfer effects and its per-run probability-mass gains are read from
# that file, so nothing about it is hand-entered here. The only exception is the
# atomic-accuracy table, which the archive stores as per-operation drops while the
# write-up reports absolute accuracies; those three rows are transcribed and are
# marked as such.
CONFIRMATORY = Path("../new_stuff/stage4_verifier_bottleneck_audit_v22/"
                    "scientific/statistics/FINAL_ANALYSIS.json")

ATOMIC_ABSOLUTE = {           # transcribed from Table 9 of the write-up
    "starting adapter": (99.80, 91.50, 57.50),
    "atomic control": (99.72, 90.28, 51.50),
    "composition branch": (97.78, 85.70, 29.08),
}
TRANSFER_LABEL = {
    "final_b_depth3_hit32": "B, depth 3",
    "final_c_depth3_hit32": "C, depth 3",
    "final_d_depth3_hit32": "D, depth 3",
    "transfer_bcd_depth2_hit32": "B+C+D, depth 2",
    "transfer_bcd_depth4_hit32": "B+C+D, depth 4",
}
TRANSFER_ORDER = ["final_b_depth3_hit32", "final_c_depth3_hit32", "final_d_depth3_hit32",
                  "transfer_bcd_depth2_hit32", "transfer_bcd_depth4_hit32"]


def load_confirmatory() -> dict:
    """Per-run values of the six-run series, straight from its analysis file."""
    d = json.loads(CONFIRMATORY.read_text(encoding="utf-8"))
    per_seed = d["primary"]["per_seed"]
    runs = [(r["control_hits"] / 10.0, r["distill_hits"] / 10.0) for r in per_seed]
    paired = [(r["both_hit"], r["control_only"], r["distill_only"], r["neither"])
              for r in per_seed]
    mass = d["descriptive_non_gating"]["correct_mass_seed_deltas"]
    transfer = []
    for key in TRANSFER_ORDER:
        fam = d["secondary_holm"]["family"][key]
        st = fam["seed_statistics"]
        lo, hi = st["t_ci95"]
        transfer.append((TRANSFER_LABEL[key], st["mean_delta_pp"], (100.0 * lo, 100.0 * hi)))
    return {
        "runs": runs,
        "paired": paired,
        "mass_pp": [100.0 * mass[str(i)] for i in range(len(per_seed))],
        "transfer": transfer,
        "atomic": ATOMIC_ABSOLUTE,
        "reported": d["primary"]["seed_statistics"],
    }


CONF = load_confirmatory()


def student_t_975(df: int) -> float:
    """Two-sided 97.5 percent quantile, tabulated for the small df used here."""
    return {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}[df]


def companion_summary() -> dict:
    """Recompute the headline statistics from the per-run values.

    The recomputed mean, interval and t are asserted against the values the
    archive reports, so a transcription or parsing slip fails the run.
    """
    runs = CONF["runs"]
    control = [a for a, _ in runs]
    branch = [b for _, b in runs]
    deltas = [b - a for a, b in runs]
    n = len(deltas)
    mean = sum(deltas) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in deltas) / (n - 1))
    se = sd / math.sqrt(n)
    half = student_t_975(n - 1) * se
    # Cross-check the per-run values against the paired matrix.
    for (a, b), (both, only_c, only_b, neither) in zip(runs, CONF["paired"]):
        total = both + only_c + only_b + neither
        assert total == 1000, total
        assert abs((both + only_c) / 10.0 - a) < 0.051, (a, (both + only_c) / 10.0)
        assert abs((both + only_b) / 10.0 - b) < 0.051, (b, (both + only_b) / 10.0)
    reported = CONF["reported"]
    assert abs(mean / 100.0 - reported["mean_delta"]) < 5e-5, mean
    assert abs(mean / se - reported["t"]) < 5e-3, mean / se
    for got, want in zip(((mean - half) / 100.0, (mean + half) / 100.0), reported["t_ci95"]):
        assert abs(got - want) < 5e-5, (got, want)
    return {
        "mean_control": sum(control) / n,
        "mean_branch": sum(branch) / n,
        "mean_delta": mean,
        "t_ci": (mean - half, mean + half),
        "t_stat": mean / se,
        "positive_runs": sum(1 for d in deltas if d > 0),
        "n": n,
    }


# ---------------------------------------------------------------- fig 2, per-run result
def fig_per_run() -> None:
    s = companion_summary()
    runs = CONF["runs"]
    idx = list(range(1, len(runs) + 1))
    fig, (ax, axd) = plt.subplots(1, 2, figsize=(WIDTH, 2.5), gridspec_kw={"width_ratios": [2.1, 1]})
    w = 0.38
    ax.bar([i - w / 2 for i in idx], [a for a, _ in runs], w, color=C_BASE, label="atomic control")
    ax.bar([i + w / 2 for i in idx], [b for _, b in runs], w, color=C_MAIN, label="composition branch")
    ax.set_xticks(idx)
    ax.set_xlabel("independent run")
    ax.set_ylabel("Hit@32, percent of tasks")
    ax.set_ylim(0, 88)
    ax.legend(loc="upper center", ncol=2, fontsize=6.6, columnspacing=0.8, handlelength=1.2)

    deltas = [b - a for a, b in runs]
    axd.scatter(deltas, idx, s=14, color=C_MAIN, zorder=3)
    axd.axvline(s["mean_delta"], color=C_ALT, linewidth=1.0, zorder=2,
                label=f"mean {s['mean_delta']:+.2f}")
    lo, hi = s["t_ci"]
    axd.axvspan(lo, hi, color=C_ALT, alpha=0.15, zorder=1, label="95% $t$ interval")
    axd.set_yticks(idx)
    axd.set_yticklabels([])
    axd.set_xlabel("$\\Delta$, pp")
    axd.set_xlim(22, 33)
    axd.legend(loc="lower right", fontsize=6.4, handlelength=1.0)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig3_per_run.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- fig 3, correct mass
def fig_mass() -> None:
    """Per-run gain in the normalised probability mass of the correct set."""
    vals = CONF["mass_pp"]
    idx = list(range(1, len(vals) + 1))
    mean = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
    half = student_t_975(len(vals) - 1) * sd / math.sqrt(len(vals))
    fig, ax = plt.subplots(figsize=(WIDTH, 2.2))
    ax.bar(idx, vals, 0.55, color=C_MAIN)
    ax.axhline(mean, color=C_ALT, linewidth=1.0,
               label=f"mean {mean:+.2f} pp, 95% $t$ interval [{mean-half:.2f}, {mean+half:.2f}]")
    ax.set_xticks(idx)
    ax.set_xlabel("independent run")
    ax.set_ylabel("gain in correct-set mass, pp")
    ax.set_ylim(0, max(vals) * 1.32)
    ax.legend(loc="upper left", fontsize=6.6)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig4_mass.pdf")
    plt.close(fig)
    print(f"  correct-set mass: mean {mean:+.4f} pp, "
          f"t interval [{mean-half:.4f}, {mean+half:.4f}], positive {sum(1 for v in vals if v>0)}/{len(vals)}")


# ---------------------------------------------------------------- fig 4, transfer forest
def fig_transfer() -> None:
    rows = CONF["transfer"]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.2))
    ys = list(range(len(rows)))[::-1]
    for y, (label, d, (lo, hi)) in zip(ys, rows):
        colour = C_MAIN if d > 0 else C_ALT
        ax.plot([lo, hi], [y, y], color=colour, linewidth=1.2, solid_capstyle="butt")
        ax.plot([d], [y], "o", color=colour, markersize=4.5, zorder=3)
        ax.annotate(f"{d:+.2f}", (d, y), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=6.8, color=INK)
    ax.axvline(0.0, color="black", linewidth=0.8, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("$\\Delta$ Hit@32 against the atomic control, pp")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig5_transfer.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- fig 4, atomic retention
def fig_atomic() -> None:
    """Retention in the two series, side by side but on separate axes.

    The two series do not measure APPLY on the same split: the companion series
    averages over all five operations, and this one uses a split from which SH1 is
    excluded. Putting them in one bar group would compare 85.70 against 99.17 and
    flatter the second. They therefore get one panel each, with the split named on
    the axis.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH, 2.4),
                                   gridspec_kw={"width_ratios": [1.5, 1.0]})
    labels = ["PLAN", "APPLY,\nall five", "APPLY,\nSH1"]
    bars = [
        ("starting adapter", CONF["atomic"]["starting adapter"], C_BASE),
        ("atomic control", CONF["atomic"]["atomic control"], C_FAINT),
        ("composition branch", CONF["atomic"]["composition branch"], C_MAIN),
    ]
    w = 0.8 / len(bars)
    for k, (name, vals, colour) in enumerate(bars):
        xs = [i - 0.4 + w / 2 + k * w for i in range(len(labels))]
        ax1.bar(xs, vals, w, color=colour, label=name, edgecolor="white", linewidth=0.4)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylabel("accuracy, percent")
    ax1.set_ylim(0, 126)
    ax1.set_title("(a) confirmatory series", fontsize=8)
    ax1.legend(loc="upper center", fontsize=6.0, handlelength=1.0, labelspacing=0.25)

    night = _night_retention()
    labels2 = ["PLAN", "APPLY,\nSH1 excluded"]
    bars2 = [
        ("starting adapter", [night["base_plan"], night["base_apply"]], C_BASE),
        ("APPLY removed", [night["dropped_plan"], 0.0], C_MAIN),
        ("APPLY restored", [night["plan"], night["apply"]], C_GOOD),
    ]
    w2 = 0.78 / len(bars2)
    for k, (name, vals, colour) in enumerate(bars2):
        xs = [i - 0.39 + w2 / 2 + k * w2 for i in range(len(labels2))]
        ax2.bar(xs, vals, w2, color=colour, label=name, edgecolor="white", linewidth=0.4,
                hatch="///" if name == "APPLY restored" else None)
        # A bar of height zero is the finding, so it is labelled instead of left blank.
        for x, v in zip(xs, vals):
            if v == 0.0:
                ax2.annotate("0.00", (x, 0.0), textcoords="offset points", xytext=(0, 3),
                             ha="center", fontsize=6.2, color=INK)
    ax2.set_xticks(range(len(labels2)))
    ax2.set_xticklabels(labels2, fontsize=7)
    ax2.set_ylim(0, 126)
    ax2.set_title("(b) this series", fontsize=8)
    ax2.legend(loc="upper center", fontsize=6.0, handlelength=1.0, labelspacing=0.25)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig6_atomic.pdf")
    plt.close(fig)


def _night_retention():
    """PLAN and non-SH1 APPLY for the APPLY-restored adapters, plus the dropped arm."""
    import yaml
    got = {"plan": [], "apply": []}
    for run in sorted(RUNS.iterdir()):
        cfg, fm = run / "config.resolved.yaml", run / "final_metrics.json"
        if not (run / "DONE").exists() or not cfg.exists() or not fm.exists():
            continue
        c = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        if not isinstance(c, dict):
            continue
        if not str(c.get("label", "")).startswith("night-retention-applykeep"):
            continue
        env = json.loads((run / "environment.json").read_text(encoding="utf-8"))
        if env.get("torch", {}).get("cuda_available") is not True:
            continue
        stem = Path(str(c["input"]).replace("\\", "/")).stem
        m = json.loads(fm.read_text(encoding="utf-8"))
        if stem == "sft_validation_plan":
            got["plan"].append(m["pass_at_1"])
        elif stem == "sft_validation_apply_non_sh1":
            got["apply"].append(m["pass_at_1"])
    if not got["plan"] or not got["apply"]:
        return None
    forget = json.loads((REPORTS / "ranking_statistics.json").read_text())["atomic_forgetting"]
    tab = forget["arm_table"]
    dropped = [tab[f"{r}|sft_validation_plan"]["pass_at_1"]
               for r in ("oracle0", "oracle1", "oracle2", "nomotif0", "nomotif1", "nomotif2")]
    return {
        "plan": 100.0 * sum(got["plan"]) / len(got["plan"]),
        "apply": 100.0 * sum(got["apply"]) / len(got["apply"]),
        "dropped_plan": 100.0 * sum(dropped) / len(dropped),
        "base_plan": 100.0 * tab["atomic|sft_validation_plan"]["pass_at_1"],
        "base_apply": 100.0 * tab["atomic|sft_validation_apply_non_sh1"]["pass_at_1"],
    }


# ---------------------------------------------------------------- fig 5, diversity curve
ROLE_LABEL = {"pre": "before GRPO", "iidgrpo": "GRPO, iid branch", "prefixgrpo": "GRPO, prefix branch"}
IID_ARMS = ["iid_action@0.7", "temp_mix_action@0.2,0.6,1.0,1.4", "iid_action@2.0", "iid_action@3.0"]


def fig_diversity() -> None:
    table = json.loads((REPORTS / "confirmatory_statistics.json").read_text())["arm_table"]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    for role, colour in (("pre", C_MAIN), ("iidgrpo", C_BASE), ("prefixgrpo", C_ALT)):
        xs = [table[f"{role}|{a}"]["distinct_programs_per_task"] for a in IID_ARMS]
        ys = [table[f"{role}|{a}"]["pass_at_32"] for a in IID_ARMS]
        ax.plot(xs, ys, "o-", color=colour, markersize=3.5, linewidth=1.0, alpha=0.9,
                label=f"{ROLE_LABEL[role]}: iid sweep")
        key = f"{role}|prefix_balanced_action@0.7"
        ax.plot(table[key]["distinct_programs_per_task"], table[key]["pass_at_32"], "*",
                markersize=13, color=colour, markeredgecolor="black", markeredgewidth=0.6,
                label=f"{ROLE_LABEL[role]}: prefix-balanced")
    ax.set_xlabel("distinct programs per task, of $K=32$ candidates")
    ax.set_ylabel("pass@32")
    ax.legend(loc="upper left", fontsize=6.6, handlelength=1.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig7_diversity.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- fig 6, our contrasts
def fig_contrasts() -> None:
    rk = json.loads((REPORTS / "ranking_statistics.json").read_text())["contrasts"]
    ni = json.loads((REPORTS / "night_statistics.json").read_text())["contrasts"]

    def row(label, o, colour):
        ci = o.get("crossed_seed_task_bootstrap_95_ci_pp")
        return label, o["mean_delta_pp"], tuple(ci), colour

    rows = [
        row("known fields", rk["C2_compositions_learnable_in_domain"], C_MAIN),
        row("fields 11, 17", rk["C1_primary_supervision_transfers_to_heldout_fields"], C_MAIN),
        row("equal-budget control", ni["N3b_atomctl_vs_atomic_d3_heldout"], C_BASE),
        row("A, seen motifs", rk["M3a_seen_motif_known_field"], C_MAIN),
        row("B, held motifs", rk["M1_primary_held_motif_known_field"], C_ALT),
        row("D, held motifs", rk["M2_held_motif_new_field"], C_ALT),
        row("B, motifs in training", ni["N2a_oracle_vs_atomic_motif_b"], C_GOOD),
        row("D, motifs in training", ni["N2b_oracle_vs_atomic_motif_d"], C_GOOD),
        row("depth 4", ni["N5a_oracle_vs_atomic_d4_hit32"], C_MAIN),
    ]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.1))
    ys = list(range(len(rows)))[::-1]
    for y, (label, d, (lo, hi), colour) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color=colour, linewidth=1.2, solid_capstyle="butt")
        ax.plot([d], [y], "o", color=colour, markersize=4.5, zorder=3)
        outer, off, align = (hi, 4, "left") if d >= 0 else (lo, -4, "right")
        ax.annotate(f"{d:+.2f}", (outer, y), textcoords="offset points", xytext=(off, -2.2),
                    fontsize=6.6, color=INK, ha=align)
    ax.axvline(0.0, color="black", linewidth=0.8, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.2)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(-42, 88)
    ax.set_xlabel("$\\Delta$ exhaustive Hit@32, pp")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig8_contrasts.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- fig 7, verifier
BETA_ORDER = [
    ("capacity_d3_heldout|oracle0", 0.0),
    ("capacity_d3_heldout|noisy10", 0.10225),
    ("capacity_d3_heldout|noisy25", 0.25225),
    ("capacity_d3_heldout|noisy50", 0.49725),
]


def fig_verifier() -> None:
    import yaml
    arms = dict(json.loads((REPORTS / "ranking_statistics.json").read_text())["arm_table"])
    for tag in ("10", "25", "50"):
        for run in sorted(RUNS.iterdir()):
            cfg, fm = run / "config.resolved.yaml", run / "final_metrics.json"
            if not (run / "DONE").exists() or not cfg.exists() or not fm.exists():
                continue
            c = yaml.safe_load(cfg.read_text(encoding="utf-8"))
            if not isinstance(c, dict):
                continue
            if f"noisy_b{tag}_seed0" in str(c.get("adapter", "")) and \
                    Path(str(c.get("input", "")).replace("\\", "/")).stem == "capacity_d3_heldout":
                arms[f"capacity_d3_heldout|noisy{tag}"] = json.loads(fm.read_text(encoding="utf-8"))
                break
    betas, hits, masses = [], [], []
    for key, beta in BETA_ORDER:
        if key not in arms:
            continue
        betas.append(beta)
        hits.append(arms[key]["hit_at_32"])
        masses.append(arms[key]["mean_correct_mass"])
    base = arms["capacity_d3_heldout|atomic"]

    # Two stacked panels sharing the x axis. A single axes with two y scales
    # would be a dual-axis chart, which misleads about the relative size of the
    # two declines.
    fig, (axh, axm) = plt.subplots(2, 1, figsize=(WIDTH, 3.4), sharex=True,
                                   gridspec_kw={"height_ratios": [1.0, 0.72]})
    axh.plot(betas, hits, "o-", color=C_MAIN, markersize=4, linewidth=1.1,
             label="composition supervision")
    axh.axhline(base["hit_at_32"], color=C_BASE, linestyle="--", linewidth=1.0,
                label="no supervision")
    for beta, hit in zip(betas, hits):
        axh.annotate(f"{hit:.3f}", (beta, hit), textcoords="offset points",
                     xytext=(3, 6), fontsize=6.8, color=INK)
    axh.set_ylabel("exhaustive Hit@32")
    axh.set_ylim(0.0, 1.10)
    axh.legend(loc="lower left", fontsize=7)

    axm.plot(betas, masses, "s-", color=C_ALT, markersize=3.6, linewidth=1.1,
             label="composition supervision")
    axm.axhline(base["mean_correct_mass"], color=C_BASE, linestyle="--", linewidth=1.0)
    axm.set_ylabel("correct-set mass")
    axm.set_xlabel(r"verifier false-accept rate $\beta$, with $\alpha=1$")
    axm.set_ylim(0.0, 0.095)
    fig.savefig(OUT / "fig9_verifier.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        fig_per_run()
        fig_mass()
        fig_transfer()
        fig_atomic()
        fig_diversity()
        fig_contrasts()
        fig_verifier()
    s = companion_summary()
    print("companion series, recomputed from its per-run values:")
    print(f"  control {s['mean_control']:.2f}  branch {s['mean_branch']:.2f}"
          f"  delta {s['mean_delta']:+.2f}")
    print(f"  95% t interval [{s['t_ci'][0]:.3f}, {s['t_ci'][1]:.3f}]"
          f"  t = {s['t_stat']:.3f}  positive {s['positive_runs']}/{s['n']}")
    for f in sorted(OUT.glob("fig*.pdf")):
        print(f"  written: {f}")


if __name__ == "__main__":
    main()
