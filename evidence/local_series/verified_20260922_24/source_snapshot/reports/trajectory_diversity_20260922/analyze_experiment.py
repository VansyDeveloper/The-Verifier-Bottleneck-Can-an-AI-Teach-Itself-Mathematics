"""Verify new selection experiment and summarize paired held-out outcomes."""

from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path

from scipy.stats import t

from analysis import analyze_ranking, digest, hit_curve, read_jsonl


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO / "artifacts/trajectory_diversity_20260922"
CONFIRM = REPO / "artifacts/stage4_composition_confirm_v1_0p6b"
OUT = HERE / "data/new_experiment"


def paired_stats(differences: list[float]) -> dict:
    if len(differences) != 6 or not all(math.isfinite(value) for value in differences):
        raise ValueError("paired inference requires six finite seed differences")
    n = len(differences)
    mean = sum(differences) / n
    sd = math.sqrt(sum((value - mean) ** 2 for value in differences) / (n - 1))
    margin = t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    extreme = 0
    for signs in itertools.product((-1, 1), repeat=n):
        flipped = sum(sign * value for sign, value in zip(signs, differences)) / n
        if abs(flipped) >= abs(mean) - 1e-12:
            extreme += 1
    return {"n": n, "mean_difference": mean, "sd": sd,
            "ci95_low": mean - margin, "ci95_high": mean + margin,
            "exact_signflip_two_sided_p": extreme / 64,
            "positive_pairs": sum(value > 0 for value in differences),
            "negative_pairs": sum(value < 0 for value in differences)}


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _audit_arm(seed: int, arm: str, holdout_sha: str, freeze_sha: str, adapter_sha: str) -> dict[str, dict]:
    branch = f"seed{seed}_{arm}_holdout"
    base = ROOT / "rankings"
    metric_dir = base / "metrics" / branch
    ranking_dir = base / "rankings" / branch
    status = json.loads((base / "status" / f"{branch}.json").read_text(encoding="utf-8"))
    if status.get("status") != "DONE" or status.get("tasks") != 1000:
        raise ValueError(f"{branch}: incomplete evaluation")
    receipt_paths = sorted(metric_dir.glob("part-*.receipt.json"))
    if len(receipt_paths) != 40:
        raise ValueError(f"{branch}: expected 40 evaluation shards")
    tasks = {}
    for shard in range(40):
        stem = f"part-{shard:05d}"
        receipt = json.loads((metric_dir / f"{stem}.receipt.json").read_text(encoding="utf-8"))
        metric_path = metric_dir / f"{stem}.jsonl"
        ranking_path = ranking_dir / f"{stem}.jsonl.gz"
        binding = receipt["binding"]
        if (receipt.get("status") != "DONE" or binding.get("seed") != seed or binding.get("arm") != arm or
            binding.get("holdout_sha256") != holdout_sha or
            binding.get("selection_freeze_sha256") != freeze_sha or
            binding.get("adapter_tree_sha256") != adapter_sha):
            raise ValueError(f"{branch}/{stem}: binding mismatch")
        if digest(metric_path).lower() != receipt["metrics_sha256"].lower() or digest(ranking_path).lower() != receipt["ranking_sha256"].lower():
            raise ValueError(f"{branch}/{stem}: shard SHA-256 mismatch")
        metrics = list(read_jsonl(metric_path))
        rankings = list(read_jsonl(ranking_path))
        if len(metrics) != len(rankings) or len(metrics) != 25:
            raise ValueError(f"{branch}/{stem}: row count mismatch")
        if [row["task_id"] for row in metrics] != receipt["task_ids"] or [row["task_id"] for row in rankings] != receipt["task_ids"]:
            raise ValueError(f"{branch}/{stem}: task order mismatch")
        for metric, raw in zip(metrics, rankings):
            task_id = metric["task_id"]
            if task_id in tasks or task_id != raw["task_id"] or metric["task_fingerprint"] != raw["task_fingerprint"]:
                raise ValueError(f"{branch}: duplicate or mismatched task")
            if metric["split"] != "final_a" or metric["depth"] != 3 or len(raw["ranking"]) != 125:
                raise ValueError(f"{branch}: wrong task stratum or ranking length")
            derived = analyze_ranking(raw["ranking"])
            if derived["best_rank"] != metric["best_rank"] or abs(derived["correct_mass"] - metric["correct_mass"]) > 1e-8:
                raise ValueError(f"{branch}: saved metric disagrees with raw ranking")
            for k in (1, 8, 16, 32, 64):
                if metric[f"hit@{k}"] != float(derived["best_rank"] <= k):
                    raise ValueError(f"{branch}: hit@{k} mismatch")
            tasks[task_id] = {"task_id": task_id, "task_fingerprint": metric["task_fingerprint"], **derived}
    if len(tasks) != 1000:
        raise ValueError(f"{branch}: not 1000 unique tasks")
    return tasks


def main() -> None:
    freeze_path = ROOT / "SELECTION_FROZEN.json"
    freeze_sha = digest(freeze_path).upper()
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    holdout_path = ROOT / "data/holdout_final_a.jsonl"
    holdout_receipt = json.loads((ROOT / "data/holdout_receipt.json").read_text(encoding="utf-8"))
    if holdout_receipt.get("status") != "DONE" or digest(holdout_path).upper() != holdout_receipt["sha256"]:
        raise ValueError("holdout integrity failure")
    expected_tasks = {row["task_id"] for row in read_jsonl(holdout_path)}
    if len(expected_tasks) != 1000:
        raise ValueError("holdout task IDs are not unique")
    base_atomic = json.loads((CONFIRM / "rankings/replicateshared/atomic/replicateshared_atomic_base.json").read_text(encoding="utf-8"))
    if base_atomic.get("status") != "DONE":
        raise ValueError("frozen atomic baseline absent")
    seed_rows = []
    task_rows = []
    curves = []
    atomic_rows = []
    for seed in range(83000, 83006):
        pair = json.loads((ROOT / "runs" / f"seed{seed}_pair.json").read_text(encoding="utf-8"))
        if pair.get("status") != "DONE" or pair["freeze_sha256"] != freeze_sha:
            raise ValueError(f"seed {seed}: training pair integrity failure")
        arms = {}
        for arm in ("random", "diverse"):
            adapter = ROOT / "adapters" / f"seed{seed}_{arm}"
            if digest(adapter / "training_receipt.json").upper() != pair["adapters"][arm]["receipt_sha256"]:
                raise ValueError(f"seed {seed} {arm}: adapter receipt changed")
            values = _audit_arm(seed, arm, holdout_receipt["sha256"], freeze_sha,
                                pair["adapters"][arm]["tree_sha256"])
            if set(values) != expected_tasks:
                raise ValueError(f"seed {seed} {arm}: evaluated tasks differ from holdout")
            arms[arm] = values
            ordered = [values[key] for key in sorted(values)]
            curve = hit_curve([row["best_rank"] for row in ordered])
            for k, hit in enumerate(curve, 1):
                curves.append({"seed": seed, "arm": arm, "k": k, "hit_fraction": hit})
            seed_rows.append({"seed": seed, "arm": arm, "hit32": curve[31],
                              "entropy_nats": sum(row["entropy_nats"] for row in ordered) / 1000,
                              "effective_programs": sum(row["effective_programs"] for row in ordered) / 1000,
                              "correct_mass": sum(row["correct_mass"] for row in ordered) / 1000,
                              "correct_entropy_nats": sum(row["correct_entropy_nats"] for row in ordered) / 1000})
            for row in ordered:
                task_rows.append({"seed": seed, "arm": arm, **row})
            atomic_path = ROOT / "rankings/atomic" / f"seed{seed}_{arm}_atomic.json"
            atomic = json.loads(atomic_path.read_text(encoding="utf-8"))
            if atomic.get("status") != "DONE" or digest(ROOT / "rankings/atomic" / f"seed{seed}_{arm}_atomic.generations.jsonl").lower() != atomic["generation_sha256"].lower():
                raise ValueError(f"seed {seed} {arm}: atomic evaluation invalid")
            for operation in ("SH1", "AX1", "AC1", "SC2", "REV"):
                atomic_rows.append({"seed": seed, "arm": arm, "operation": operation,
                                    "plan_accuracy": atomic["plan"]["by_operation"][operation],
                                    "apply_accuracy": atomic["apply"]["by_operation"][operation],
                                    "plan_drop_from_base": base_atomic["plan"]["by_operation"][operation] - atomic["plan"]["by_operation"][operation],
                                    "apply_drop_from_base": base_atomic["apply"]["by_operation"][operation] - atomic["apply"]["by_operation"][operation]})
        if set(arms["random"]) != set(arms["diverse"]):
            raise ValueError(f"seed {seed}: evaluation tasks differ between arms")
    differences = []
    for seed in range(83000, 83006):
        control = next(row for row in seed_rows if row["seed"] == seed and row["arm"] == "random")
        diverse = next(row for row in seed_rows if row["seed"] == seed and row["arm"] == "diverse")
        differences.append(diverse["hit32"] - control["hit32"])
    primary = paired_stats(differences)
    OUT.mkdir(parents=True, exist_ok=True)
    _csv(OUT / "task_metrics.csv", task_rows)
    _csv(OUT / "seed_metrics.csv", seed_rows)
    _csv(OUT / "hitk_per_seed.csv", curves)
    _csv(OUT / "atomic_forgetting.csv", atomic_rows)
    mean_curves = []
    for arm in ("random", "diverse"):
        for k in range(1, 126):
            hits = [row["hit_fraction"] for row in curves if row["arm"] == arm and row["k"] == k]
            mean_curves.append({"arm": arm, "k": k, "mean_hit_fraction": sum(hits) / 6,
                                "min_seed": min(hits), "max_seed": max(hits)})
    _csv(OUT / "hitk_mean.csv", mean_curves)
    result = {"schema": "trajectory-diversity.final-analysis.v1", "status": "DONE",
              "study_class": "exploratory-dataset-composition-intervention",
              "holdout_sha256": holdout_receipt["sha256"], "freeze_sha256": freeze_sha,
              "task_rows": len(task_rows), "verified_shards": 480,
              "hit32_differences_by_seed": dict(zip(range(83000, 83006), differences)),
              "primary": primary,
              "limits": ["different source tasks in the two selection arms",
                         "post-hoc entropy and full Hit@K are exploratory",
                         "conditional entropy among 125 admissible programs, not full-vocabulary entropy"],
              "output_sha256": {path.name: digest(path).upper() for path in sorted(OUT.glob("*.csv"))}}
    (OUT / "analysis_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
