"""Audit and summarize the saved exact 125-program final-A rankings."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _softmax(scores: list[float]) -> list[float]:
    shift = max(scores)
    weights = [math.exp(score - shift) for score in scores]
    total = sum(weights)
    return [weight / total for weight in weights]


def _entropy(probs: list[float]) -> float:
    return -sum(value * math.log(value) for value in probs if value > 0)


def analyze_ranking(ranking: list[dict], expected_count: int = 125) -> dict:
    if len(ranking) != expected_count:
        raise ValueError(f"expected {expected_count} programs, found {len(ranking)}")
    programs = [tuple(row["program"]) for row in ranking]
    if len(set(programs)) != expected_count:
        raise ValueError("duplicate program")
    if [row["rank"] for row in ranking] != list(range(1, expected_count + 1)):
        raise ValueError("ranks are not consecutive")
    scores = [float(row["score"]) for row in ranking]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("non-finite score")
    if any(scores[index] < scores[index + 1] for index in range(len(scores) - 1)):
        raise ValueError("score order disagrees with rank")
    correct = [row["correct"] for row in ranking]
    if not all(type(value) is bool for value in correct):
        raise ValueError("non-boolean correctness flag")
    if not any(correct):
        raise ValueError("task has no correct program")
    probs = _softmax(scores)
    mass = sum(value for value, good in zip(probs, correct) if good)
    correct_probs = [value / mass for value, good in zip(probs, correct) if good]
    entropy = _entropy(probs)
    return {
        "best_rank": next(index for index, good in enumerate(correct, 1) if good),
        "correct_programs": sum(correct),
        "entropy_nats": entropy,
        "effective_programs": math.exp(entropy),
        "correct_mass": mass,
        "correct_entropy_nats": _entropy(correct_probs),
    }


def hit_curve(best_ranks: list[int], max_rank: int = 125) -> list[float]:
    if not best_ranks or any(rank < 1 or rank > max_rank for rank in best_ranks):
        raise ValueError("invalid best ranks")
    return [sum(rank <= k for rank in best_ranks) / len(best_ranks) for k in range(1, max_rank + 1)]


def audit_arm(root: Path, replicate: int, arm: str) -> dict[str, dict]:
    branch = f"replicate{replicate}_{arm}_final_a"
    base = root / "rankings" / f"replicate{replicate}"
    metrics_dir = base / "metrics" / branch
    ranking_dir = base / "rankings" / branch
    status = json.loads((base / "status" / f"{branch}.json").read_text(encoding="utf-8"))
    if status.get("status") != "DONE" or status.get("tasks") != 1000:
        raise ValueError(f"{branch}: incomplete status")
    rows = {}
    shard_numbers = list(range(40))
    actual_receipts = sorted(metrics_dir.glob("part-*.receipt.json"))
    if len(actual_receipts) != 40:
        raise ValueError(f"{branch}: expected 40 receipts, found {len(actual_receipts)}")
    for shard in shard_numbers:
        stem = f"part-{shard:05d}"
        receipt = json.loads((metrics_dir / f"{stem}.receipt.json").read_text(encoding="utf-8"))
        metric_path = metrics_dir / f"{stem}.jsonl"
        ranking_path = ranking_dir / f"{stem}.jsonl.gz"
        if receipt.get("status") != "DONE":
            raise ValueError(f"{branch}/{stem}: incomplete receipt")
        if digest(metric_path).lower() != receipt["metrics_sha256"].lower():
            raise ValueError(f"{branch}/{stem}: metrics SHA-256 mismatch")
        if digest(ranking_path).lower() != receipt["ranking_sha256"].lower():
            raise ValueError(f"{branch}/{stem}: ranking SHA-256 mismatch")
        metrics = list(read_jsonl(metric_path))
        rankings = list(read_jsonl(ranking_path))
        if len(metrics) != receipt["metrics_rows"] or len(rankings) != receipt["ranking_rows"]:
            raise ValueError(f"{branch}/{stem}: receipt row count mismatch")
        if [row["task_id"] for row in metrics] != receipt["task_ids"]:
            raise ValueError(f"{branch}/{stem}: task order differs from receipt")
        if [row["task_id"] for row in rankings] != receipt["task_ids"]:
            raise ValueError(f"{branch}/{stem}: ranking tasks differ from receipt")
        for metric, raw in zip(metrics, rankings):
            task_id = metric["task_id"]
            if task_id in rows:
                raise ValueError(f"{branch}: duplicate task {task_id}")
            if metric["task_fingerprint"] != raw["task_fingerprint"]:
                raise ValueError(f"{branch}: task fingerprint mismatch")
            if metric["split"] != raw["split"] or metric["depth"] != raw["depth"]:
                raise ValueError(f"{branch}: split or depth mismatch")
            if metric["split"] != "final_a" or metric["depth"] != 3:
                raise ValueError(f"{branch}: wrong task stratum")
            result = analyze_ranking(raw["ranking"])
            if result["best_rank"] != metric["best_rank"]:
                raise ValueError(f"{branch}: saved best rank mismatch")
            if abs(result["correct_mass"] - metric["correct_mass"]) > 1e-8:
                raise ValueError(f"{branch}: saved correct mass mismatch")
            for k in (1, 8, 16, 32, 64):
                if metric[f"hit@{k}"] != float(result["best_rank"] <= k):
                    raise ValueError(f"{branch}: saved hit@{k} mismatch")
            rows[task_id] = {"task_id": task_id, "task_fingerprint": metric["task_fingerprint"], **result}
    if len(rows) != 1000:
        raise ValueError(f"{branch}: expected 1000 distinct tasks, found {len(rows)}")
    return rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Stage-4 confirmation artifact root")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    published = list(csv.DictReader((args.root / "post_analysis_release_v1" / "tables" / "primary_per_replicate.csv").open(encoding="utf-8")))
    if len(published) != 6:
        raise ValueError("published primary table must have six rows")
    task_rows = []
    curve_rows = []
    seed_rows = []
    for replicate in range(6):
        arms = {arm: audit_arm(args.root, replicate, arm) for arm in ("atomic_control", "composition_distill")}
        if set(arms["atomic_control"]) != set(arms["composition_distill"]):
            raise ValueError(f"replicate {replicate}: arms evaluated on different tasks")
        counts = {}
        for arm, values in arms.items():
            ordered = [values[key] for key in sorted(values)]
            ranks = [row["best_rank"] for row in ordered]
            curve = hit_curve(ranks)
            counts[arm] = sum(rank <= 32 for rank in ranks)
            for k, hit in enumerate(curve, 1):
                curve_rows.append({"replicate": replicate, "arm": arm, "k": k, "hit_fraction": hit})
            seed_rows.append({"replicate": replicate, "arm": arm, "hit32": curve[31],
                              "entropy_nats": sum(row["entropy_nats"] for row in ordered) / len(ordered),
                              "effective_programs": sum(row["effective_programs"] for row in ordered) / len(ordered),
                              "correct_mass": sum(row["correct_mass"] for row in ordered) / len(ordered),
                              "correct_entropy_nats": sum(row["correct_entropy_nats"] for row in ordered) / len(ordered)})
            for row in ordered:
                task_rows.append({"replicate": replicate, "arm": arm, **row})
        frozen = published[replicate]
        if int(frozen["replicate_label"]) != replicate or counts["atomic_control"] != int(frozen["control_hits"]) or counts["composition_distill"] != int(frozen["distill_hits"]):
            raise ValueError(f"replicate {replicate}: hit@32 differs from frozen primary table")
    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "task_metrics.csv", list(task_rows[0]), task_rows)
    _write_csv(args.output / "seed_metrics.csv", list(seed_rows[0]), seed_rows)
    _write_csv(args.output / "hitk_per_seed.csv", list(curve_rows[0]), curve_rows)
    means = []
    for arm in ("atomic_control", "composition_distill"):
        for k in range(1, 126):
            hits = [row["hit_fraction"] for row in curve_rows if row["arm"] == arm and row["k"] == k]
            means.append({"arm": arm, "k": k, "mean_hit_fraction": sum(hits) / 6,
                          "min_seed": min(hits), "max_seed": max(hits)})
    _write_csv(args.output / "hitk_mean.csv", list(means[0]), means)
    summary = {"schema": "verifier-bottleneck.hitk-entropy.v1", "status": "DONE", "replicates": 6,
               "tasks_per_replicate": 1000, "candidate_programs": 125,
               "total_arm_task_rows": len(task_rows), "verified_shards": 6 * 2 * 40,
               "conditional_distribution": "softmax of sum log-probabilities of admissible operation tokens",
               "source_root": str(args.root.resolve()),
               "source_primary_table_sha256": digest(args.root / "post_analysis_release_v1" / "tables" / "primary_per_replicate.csv"),
               "output_sha256": {path.name: digest(path) for path in sorted(args.output.glob("*.csv"))}}
    (args.output / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
