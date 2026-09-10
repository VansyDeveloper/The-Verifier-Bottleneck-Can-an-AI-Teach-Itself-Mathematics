"""Paired exhaustive-ranking analysis; no ranks are inferred from aggregates."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev

from scipy.stats import t as student_t

from composition_core import enumerate_programs, trajectory
from confirm_stats import exact_sign_flip, seed_t_statistics
from .common import file_hash, read_jsonl

ARMS = ("atomic_control", "composition")
PROVENANCE = ("scorer_id", "tokenizer_hash", "prompt_version")
IDENTITY = ("task_fingerprint", "split", "p", "depth", "degree", "correct_count",
            "sh1_any", "sh1_required")


def input_files(path):
    path = Path(path)
    files = sorted(p for p in path.rglob("*") if p.name.endswith((".jsonl", ".jsonl.gz"))) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"no JSONL files: {path}")
    return files


def read_rows(path):
    for file in input_files(path):
        yield from read_jsonl(file)


def provenance(entry, row):
    sources = (entry, row, row.get("binding", {}), row.get("provenance", {}))
    result = []
    for key in PROVENANCE:
        values = {source[key] for source in sources if source.get(key)}
        if len(values) != 1:
            raise ValueError(f"missing or conflicting {key}; specify it in the manifest")
        result.append(values.pop())
    return tuple(result)


def load_manifest(path, arms=ARMS):
    if len(arms) != 2 or len(set(arms)) != 2 or any(not isinstance(arm, str) or not arm for arm in arms):
        raise ValueError("choose two distinct arm names")
    path = Path(path)
    entries = json.loads(path.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest must be a nonempty JSON list")
    runs, scorer, reference, used_metrics = {}, None, None, set()
    shared_inputs = None
    for entry in entries:
        seed, arm = entry["seed"], entry["arm"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("training seed must be an integer")
        if arm not in arms or (seed, arm) in runs:
            raise ValueError(f"invalid or duplicate arm: {seed}/{arm}")
        files = {file.resolve() for file in input_files(path.parent / entry["metrics"])}
        if files & used_metrics:
            raise ValueError("one metrics file cannot represent multiple independent arms/seeds")
        used_metrics.update(files)
        tasks = {}
        if entry.get("tasks"):
            for task in read_rows(path.parent / entry["tasks"]):
                if task["task_id"] in tasks:
                    raise ValueError("duplicate task ID in task file")
                tasks[task["task_id"]] = task
            if not tasks:
                raise ValueError("task file is empty")
        rows, run_inputs = {}, None
        for raw in read_rows(path.parent / entry["metrics"]):
            row = {**raw, **raw.get("metrics", {})}
            current = provenance(entry, row)
            if scorer is not None and current != scorer:
                raise ValueError("incompatible scorer/tokenizer/prompt provenance")
            scorer = current
            sources = (entry, row, row.get("binding", {}), row.get("provenance", {}))
            inputs = []
            for key in ("base_hash", "data_hash", "model_hash"):
                values = {source[key] for source in sources if source.get(key) is not None}
                if len(values) > 1:
                    raise ValueError(f"conflicting {key} provenance")
                inputs.append(next(iter(values), None))
            if run_inputs is not None and inputs != run_inputs:
                raise ValueError("one run contains different base/data/model provenance")
            run_inputs = inputs
            task_id = row["task_id"]
            if task_id in rows:
                raise ValueError(f"duplicate metric task ID: {task_id}")
            for seed_key in ('seed', 'training_seed'):
                if seed_key in row and row[seed_key] != seed:
                    raise ValueError(f"manifest/metric seed mismatch: {task_id}")
            if tasks:
                task = tasks[task_id]
                for key in ("task_fingerprint", "split", "p", "depth"):
                    if row[key] != task[key]:
                        raise ValueError(f"task/metric identity mismatch: {task_id}/{key}")
                correct = [program for program in enumerate_programs(int(task["depth"]))
                           if trajectory(task["start"], program, task["p"])[-1] == tuple(task["target"])]
                details = {"degree": len(task["start"]) - 1, "correct_count": len(correct),
                           "sh1_any": any("SH1" in program for program in correct),
                           "sh1_required": bool(correct) and all("SH1" in program for program in correct)}
                if any(row.get(key) is not None and row[key] != value for key, value in details.items()):
                    raise ValueError(f"task/metric stratum mismatch: {task_id}")
                row.update(details)
            depth, rank = row["depth"], row["best_rank"]
            if (isinstance(depth, bool) or not isinstance(depth, int) or depth not in (1, 2, 3, 4)
                    or isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= 5 ** depth):
                raise ValueError(f"invalid exhaustive rank/depth: {task_id}")
            if row.get("candidate_count", 5 ** depth) != 5 ** depth:
                raise ValueError(f"candidate space is not exhaustive: {task_id}")
            row["correct_mass"] = row.get("correct_mass", row.get("correct_set_mass"))
            if (isinstance(row["correct_mass"], bool) or not isinstance(row["correct_mass"], (int, float))
                    or not 0 <= row["correct_mass"] <= 1):
                raise ValueError(f"invalid correct mass: {task_id}")
            if not row.get("task_fingerprint") or not row.get("split"):
                raise ValueError(f"missing task identity: {task_id}")
            count = row.get("correct_count")
            if count is not None and (isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 5 ** depth):
                raise ValueError(f"empty/invalid correct set: {task_id}")
            if any(row.get(key) is not None and not isinstance(row[key], bool) for key in ("sh1_any", "sh1_required")):
                raise ValueError(f"SH1 strata must be Boolean: {task_id}")
            if row.get("sh1_required") and row.get("sh1_any") is False:
                raise ValueError(f"inconsistent SH1 strata: {task_id}")
            for k in (1, 8, 16, 32, 64):
                if f"hit@{k}" in row and row[f"hit@{k}"] != int(rank <= k):
                    raise ValueError(f"rank disagrees with hit@{k}: {task_id}")
            rows[task_id] = row
        if not rows:
            raise ValueError("metrics file has no tasks")
        if tasks and rows.keys() != tasks.keys():
            raise ValueError("metrics do not cover the full task file")
        if shared_inputs is not None and run_inputs[:2] != shared_inputs:
            raise ValueError("paired runs have different base/data provenance")
        shared_inputs = run_inputs[:2]
        identities = {task_id: tuple(row.get(key) for key in IDENTITY) for task_id, row in rows.items()}
        if not identities or (reference is not None and identities != reference):
            raise ValueError("paired task IDs/fingerprints/strata differ across arms or seeds")
        reference = identities
        runs[seed, arm] = rows
    seeds = sorted({seed for seed, _ in runs})
    if set(runs) != set(itertools.product(seeds, arms)):
        raise ValueError(f"each seed needs both arms: {arms}")
    return runs, dict(zip(PROVENANCE, scorer))


def interval(values):
    mean = fmean(values)
    if len(values) == 1:
        return mean, None, None
    if len(values) == 6:
        lower, upper = seed_t_statistics(values)["t_ci95"]
    else:
        width = float(student_t.ppf(0.975, len(values) - 1)) * stdev(values) / math.sqrt(len(values))
        lower, upper = mean - width, mean + width
    return mean, lower, upper


def sign_p(numerators, denominator):
    if len(numerators) == 6:
        return exact_sign_flip(numerators, denominator)["p_two_sided"]
    if len(numerators) > 20:
        raise ValueError("exact sign enumeration supports at most 20 seeds")
    observed = abs(sum(numerators))
    return sum(abs(sum(s * n for s, n in zip(signs, numerators))) >= observed
               for signs in itertools.product((-1, 1), repeat=len(numerators))) / 2 ** len(numerators)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(manifest, output, plots=False, arms=ARMS):
    runs, scorer = load_manifest(manifest, arms)
    treatment = "composition" if tuple(arms) == ARMS else "treatment"
    treatment_only = treatment + "_only"
    seeds = sorted({seed for seed, _ in runs})
    first = runs[seeds[0], arms[0]]
    groups = defaultdict(list)
    for task_id, row in first.items():
        groups[row["split"], row["depth"]].append(task_id)
    curves, summaries, pairs, counts, strata, strata_by_seed = [], [], [], [], [], []
    for (split, depth), ids in sorted(groups.items()):
        for k in range(1, 5 ** depth + 1):
            hits = {arm: [sum(runs[seed, arm][task]["best_rank"] <= k for task in ids)
                          for seed in seeds] for arm in arms}
            deltas = [c - a for a, c in zip(hits[arms[0]], hits[arms[1]])]
            for arm in arms:
                for seed, hit in zip(seeds, hits[arm]):
                    curves.append(dict(split=split, depth=depth, seed=seed, arm=arm, k=k,
                                       n_tasks=len(ids), hit=hit / len(ids)))
            mean, low, high = interval([value / len(ids) for value in deltas])
            summaries.append(dict(split=split, depth=depth, k=k, n_seeds=len(seeds), n_tasks=len(ids),
                                  **{arm: fmean(hits[arm]) / len(ids) for arm in arms}, delta=mean,
                                  delta_ci95_low=low, delta_ci95_high=high,
                                  sign_flip_p=sign_p(deltas, len(ids)) if k == 32 else None))
        for seed in seeds:
            table = dict(split=split, depth=depth, seed=seed, k=32, n_tasks=len(ids),
                         both=0, control_only=0, **{treatment_only: 0}, neither=0)
            for task_id in ids:
                a, c = (runs[seed, arm][task_id] for arm in arms)
                success = (a["best_rank"] <= 32, c["best_rank"] <= 32)
                outcome = {(True, True): "both", (True, False): "control_only",
                           (False, True): treatment_only, (False, False): "neither"}[success]
                table[outcome] += 1
                pairs.append(dict(seed=seed, task_id=task_id, **{key: a.get(key) for key in IDENTITY},
                                  control_rank=a["best_rank"], **{treatment + "_rank": c["best_rank"]},
                                  delta_rank=c["best_rank"] - a["best_rank"],
                                  delta_correct_mass=c["correct_mass"] - a["correct_mass"],
                                  delta_mrr=1 / c["best_rank"] - 1 / a["best_rank"],
                                  outcome_at_32=outcome))
            table["delta"] = (table[treatment_only] - table["control_only"]) / len(ids)
            counts.append(table)
        for attribute in ("all", "p", "degree", "correct_count", "sh1_any", "sh1_required"):
            subsets = defaultdict(list)
            for task_id in ids:
                value = "all" if attribute == "all" else first[task_id].get(attribute)
                if attribute == "correct_count" and value is not None:
                    value = str(value) if value < 3 else "3+"
                subsets[str(value) if value is not None else "unknown"].append(task_id)
            for value, subset in sorted(subsets.items()):
                deltas, ranks, masses, mrrs = [], [], [], []
                for seed in seeds:
                    a, c = (runs[seed, arm] for arm in arms)
                    deltas.append(fmean(int(c[t]["best_rank"] <= 32) - int(a[t]["best_rank"] <= 32) for t in subset))
                    ranks.append(fmean(c[t]["best_rank"] - a[t]["best_rank"] for t in subset))
                    masses.append(fmean(c[t]["correct_mass"] - a[t]["correct_mass"] for t in subset))
                    mrrs.append(fmean(1 / c[t]["best_rank"] - 1 / a[t]["best_rank"] for t in subset))
                    strata_by_seed.append(dict(split=split, depth=depth, stratum=attribute, value=value,
                                               seed=seed, n_tasks=len(subset), delta_hit32=deltas[-1],
                                               delta_rank=ranks[-1], delta_correct_mass=masses[-1], delta_mrr=mrrs[-1]))
                mean, low, high = interval(deltas)
                strata.append(dict(split=split, depth=depth, stratum=attribute, value=value,
                                   n_tasks=len(subset), n_seeds=len(seeds), delta_hit32=mean,
                                   delta_ci95_low=low, delta_ci95_high=high, delta_rank=fmean(ranks),
                                   delta_correct_mass=fmean(masses), delta_mrr=fmean(mrrs)))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("hitk_by_seed", curves), ("hitk_summary", summaries),
                       ("paired_task_deltas", pairs), ("table9_counts", counts), ("stratified_errors", strata),
                       ("stratified_errors_by_seed", strata_by_seed)):
        write_csv(output / f"{name}.csv", rows)
    receipt = {**scorer, "manifest_sha256": file_hash(manifest),
               "arms": {"control": arms[0], "treatment": arms[1]},
               "n_seeds": len(seeds), "n_tasks": len(first), "interval": "pointwise paired run t, 95%",
               "sign_flip_endpoint": "Hit@32 only; no tests selected from the full curve",
               "status": "reanalysis; no new training", "unknown_strata": "not available in source metrics/tasks"}
    sources = set()
    for entry in json.loads(Path(manifest).read_text()):
        for key in ("metrics", "tasks"):
            if entry.get(key):
                sources.update(input_files(Path(manifest).parent / entry[key]))
    receipt["sources"] = [{"path": str(path.resolve()), "sha256": file_hash(path)}
                          for path in sorted(sources)]
    (output / "analysis.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if plots:
        plot_curves(summaries, output, arms)
    return summaries


def plot_curves(rows, output, arms=ARMS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    groups = defaultdict(list)
    for row in rows:
        groups[row["split"], row["depth"]].append(row)
    for index, ((split, depth), group) in enumerate(sorted(groups.items())):
        fig, axes = plt.subplots(2, 1, figsize=(5.5, 4.8), sharex=True, layout="constrained")
        ks = [row["k"] for row in group]
        for key, color in zip(arms, ("#0072B2", "#D55E00")):
            label = key.replace("_", " ").capitalize()
            axes[0].plot(ks, [row[key] for row in group], label=label, color=color,
                         linestyle='--' if key == arms[0] else '-')
        title = {'capacity_d3_train': 'Known fields', 'capacity_d3_heldout': 'New fields'}.get(split, split)
        axes[0].set(ylabel="Fraction solved", ylim=(0, 1), title=f"{title}, depth {depth}")
        axes[0].legend(frameon=False)
        axes[1].plot(ks, [row["delta"] for row in group], color="#009E73")
        if group[0]["delta_ci95_low"] is not None:
            axes[1].fill_between(ks, [r["delta_ci95_low"] for r in group],
                                 [r["delta_ci95_high"] for r in group], color="#009E73", alpha=0.2,
                                 label="95% pointwise run t interval")
            axes[1].legend(frameon=False, fontsize=8)
        axes[1].axhline(0, color="0.5", linewidth=0.7)
        axes[1].set(xlabel="K", ylabel="Paired difference", xlim=(1, ks[-1]))
        for extension in ("png", "pdf"):
            fig.savefig(output / f"hitk_{index}_depth{depth}.{extension}", dpi=300)
        plt.close(fig)


def historical_summary(source, output):
    """Recompute published aggregate statistics; this cannot produce H(K)."""
    primary = json.loads(Path(source).read_text())["primary"]
    rows = primary["per_seed"]
    if len({row["replicate_label"] for row in rows}) != 6 or len({row["delta_denominator"] for row in rows}) != 1:
        raise ValueError("historical aggregate requires six distinct runs and a common denominator")
    for row in rows:
        if (sum(row[key] for key in ("both_hit", "control_only", "distill_only", "neither")) != row["delta_denominator"]
                or row["distill_only"] - row["control_only"] != row["delta_numerator"]
                or row["control_hits"] != row["both_hit"] + row["control_only"]
                or row["distill_hits"] != row["both_hit"] + row["distill_only"]):
            raise ValueError("published Table 9 aggregate counts are inconsistent")
    deltas = [row["delta_numerator"] / row["delta_denominator"] for row in rows]
    statistics = seed_t_statistics(deltas)
    for key in ("mean_delta", "sample_sd", "standard_error", "t", "p_two_sided"):
        if not math.isclose(statistics[key], primary["seed_statistics"][key], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"published aggregate statistic differs: {key}")
    if any(not math.isclose(a, b, abs_tol=1e-12)
           for a, b in zip(statistics["t_ci95"], primary["seed_statistics"]["t_ci95"])):
        raise ValueError("published aggregate t interval differs")
    sign_flip = exact_sign_flip([r["delta_numerator"] for r in rows], rows[0]["delta_denominator"])
    if sign_flip["p_two_sided"] != primary["exact_sign_flip"]["p_two_sided"]:
        raise ValueError("published aggregate sign-flip p differs")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "table9_published_aggregates.csv", rows)
    result = {"status": "aggregate arithmetic reproduced; task-level ranks unavailable here",
              "source_sha256": file_hash(source),
              "atomic_control_hit32": fmean(r["control_hits"] / r["delta_denominator"] for r in rows),
              "composition_hit32": fmean(r["distill_hits"] / r["delta_denominator"] for r in rows),
              "seed_statistics": statistics, "exact_sign_flip": sign_flip}
    (output / "aggregate_check.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest", type=Path, help="JSON list: seed, arm, metrics, optional tasks, scorer provenance")
    inputs.add_argument("--historical-summary", type=Path, help="PRIMARY_ANALYSIS.json; aggregate check only")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--arms", nargs=2, default=ARMS, metavar=("CONTROL", "TREATMENT"),
                        help="manifest arm names; differences are treatment minus control")
    args = parser.parse_args()
    if args.manifest:
        analyze(args.manifest, args.output, args.plots, args.arms)
    else:
        historical_summary(args.historical_summary, args.output)


if __name__ == "__main__":
    main()
