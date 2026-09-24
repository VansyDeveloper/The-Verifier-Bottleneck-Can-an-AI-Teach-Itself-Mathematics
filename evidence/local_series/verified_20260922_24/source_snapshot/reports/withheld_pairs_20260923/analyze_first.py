"""Independent audit of the first withheld-operation-pair block."""

from __future__ import annotations

import csv
import gzip
import hashlib
import itertools
import json
import math
import statistics
import sys
from pathlib import Path

from scipy.stats import t


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO / "artifacts/withheld_pairs_20260923"
CONFIRM = REPO / "artifacts/stage4_composition_confirm_v1_0p6b"
CORE_DIR = REPO / "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/code"
OUT = HERE / "data/first_block"
FREEZE_SHA256 = "99DF70CE2030C18F3E86368E1C4A86EA3704FA729C6C355EE8A90C90D1333DD2"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest().upper()


def same_hash(actual: str, expected: str) -> bool:
    return isinstance(actual, str) and isinstance(expected, str) and bool(actual) and actual.upper() == expected.upper()


def tree_digest(path: Path) -> str:
    value = hashlib.sha256()
    files = sorted((item for item in path.rglob("*") if item.is_file()),
                   key=lambda item: item.relative_to(path).as_posix())
    if not files:
        raise ValueError(f"empty adapter tree: {path}")
    for item in files:
        value.update(item.relative_to(path).as_posix().encode())
        value.update(b"\0")
        value.update(bytes.fromhex(digest(item)))
    return value.hexdigest()


def read_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def validate_ranking(rows: list[dict], *, expected_programs: int = 125) -> int:
    if len(rows) != expected_programs:
        raise ValueError("ranking has missing or extra programs")
    programs = [tuple(row["program"]) for row in rows]
    if len(set(programs)) != expected_programs:
        raise ValueError("ranking repeats a program")
    if expected_programs == 125:
        canonical = set(itertools.product(("SH1", "SC2", "REV", "AC1", "AX1"), repeat=3))
        if set(programs) != canonical:
            raise ValueError("ranking omits an admissible program")
    if [row["rank"] for row in rows] != list(range(1, expected_programs + 1)):
        raise ValueError("ranking has invalid ranks")
    if any(not math.isfinite(float(row["score"])) for row in rows):
        raise ValueError("ranking has non-finite score")
    if any(rows[index]["score"] < rows[index + 1]["score"] for index in range(len(rows) - 1)):
        raise ValueError("ranking is not sorted by score")
    if any(type(row["correct"]) is not bool for row in rows):
        raise ValueError("ranking has non-Boolean correctness")
    correct = [row["rank"] for row in rows if row["correct"]]
    if not correct:
        raise ValueError("ranking has no correct program")
    return min(correct)


def hit_curve(best_ranks: list[int], maximum: int = 125) -> list[float]:
    if not best_ranks or any(rank < 1 or rank > maximum for rank in best_ranks):
        raise ValueError("invalid best ranks")
    return [sum(rank <= k for rank in best_ranks) / len(best_ranks) for k in range(1, maximum + 1)]


def selected_pair_count(source: dict, identities: list[str], pair: tuple[str, str]) -> int:
    return sum(pair in zip(source[identity]["program"], source[identity]["program"][1:])
               for identity in identities)


def paired_stats(differences: list[float]) -> dict:
    if len(differences) != 3 or not all(math.isfinite(value) for value in differences):
        raise ValueError("first-block inference requires three finite seed differences")
    mean = statistics.mean(differences)
    sd = statistics.stdev(differences)
    margin = t.ppf(0.975, 2) * sd / math.sqrt(3)
    extreme = sum(abs(sum(sign * value for sign, value in zip(signs, differences)) / 3)
                  >= abs(mean) - 1e-12 for signs in itertools.product((-1, 1), repeat=3))
    return {"n": 3, "mean_difference": mean, "sd": sd,
            "ci95_low": mean - margin, "ci95_high": mean + margin,
            "exact_signflip_two_sided_p": extreme / 8,
            "positive_pairs": sum(value > 0 for value in differences),
            "negative_pairs": sum(value < 0 for value in differences)}


def derive_atomic_rows(source_rows: list[dict], raw_rows: list[dict]) -> dict:
    if len(source_rows) != len(raw_rows) or not source_rows:
        raise ValueError("atomic raw row count mismatch")
    if str(CORE_DIR) not in sys.path:
        sys.path.insert(0, str(CORE_DIR))
    from composition_core import OPS, format_state, trajectory

    counts = {metric: {} for metric in ("plan", "apply")}
    for source, raw in zip(source_rows, raw_rows):
        operation = source.get("operation") or source["witness"][0]
        if raw.get("task_id") != source["task_id"] or raw.get("operation") != operation:
            raise ValueError("atomic task identity mismatch")
        if "start" in source:
            target = format_state(trajectory(source["start"], [operation], source["p"])[-1])
            if raw.get("target") != target:
                raise ValueError("atomic target differs from source task")
        prediction = raw.get("plan_prediction")
        if prediction not in OPS or type(raw.get("plan_correct")) is not bool or raw["plan_correct"] != (prediction == operation):
            raise ValueError("atomic plan correctness mismatch")
        completion = raw.get("raw_completion")
        if (not isinstance(completion, str) or type(raw.get("correct")) is not bool or
                raw["correct"] != (completion.strip() == raw.get("target"))):
            raise ValueError("atomic apply correctness mismatch")
        for metric, passed in (("plan", raw["plan_correct"]), ("apply", raw["correct"])):
            tally = counts[metric].setdefault(operation, [0, 0])
            tally[0] += int(passed)
            tally[1] += 1
    return {metric: {op: passed / total for op, (passed, total) in groups.items()}
            for metric, groups in counts.items()}


def branch_prefix(k: int, subset: int, seed: int, arm: str) -> str:
    return f"k{k}_s{subset}_seed{seed}_{arm}"


def audit_training(seed: int, arm: str, freeze_sha: str, k: int = 1,
                   subset: int = 1) -> str:
    prefix = branch_prefix(k, subset, seed, arm)
    pair = json.loads((ROOT / "runs" / f"k{k}_s{subset}_seed{seed}_pair.json").read_text(encoding="utf-8"))
    if pair.get("status") != "DONE" or pair.get("freeze_sha256") != freeze_sha:
        raise ValueError(f"incomplete training pair: {seed}")
    adapter = ROOT / "adapters" / prefix
    receipt_path = adapter / "training_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("status") != "DONE" or receipt.get("optimizer_steps") != receipt.get("optimizer_steps_expected")
            or receipt.get("runtime_compute_dtype") != "torch.float16" or not math.isfinite(float(receipt["mean_loss"]))):
        raise ValueError(f"training receipt invalid: {prefix}")
    if not same_hash(digest(receipt_path), pair["adapters"][arm]["receipt_sha256"]):
        raise ValueError(f"training receipt SHA mismatch: {prefix}")
    actual_tree = tree_digest(adapter)
    if actual_tree != pair["adapters"][arm]["tree_sha256"]:
        raise ValueError(f"adapter tree SHA mismatch: {prefix}")
    return actual_tree


def audit_ranking(seed: int, arm: str, split: str, holdout: dict,
                  freeze_sha: str, adapter_tree_sha: str, k: int = 1,
                  subset: int = 1) -> dict:
    if str(CORE_DIR) not in sys.path:
        sys.path.insert(0, str(CORE_DIR))
    from composition_core import trajectory

    branch = f"{branch_prefix(k, subset, seed, arm)}_{split}"
    base = ROOT / "rankings"
    status = json.loads((base / "status" / f"{branch}.json").read_text(encoding="utf-8"))
    summary = json.loads((base / "summaries" / f"{branch}.json").read_text(encoding="utf-8"))
    if status.get("status") != "DONE" or status.get("tasks") != 1000 or summary.get("n_tasks") != 1000:
        raise ValueError(f"incomplete evaluation: {branch}")
    holdout_path = ROOT / holdout["path"]
    if not same_hash(digest(holdout_path), holdout["sha256"]):
        raise ValueError(f"holdout SHA mismatch: {branch}")
    expected = list(read_jsonl(holdout_path))
    expected_ids = [row["task_id"] for row in expected]
    expected_by_id = {row["task_id"]: row for row in expected}
    if len(expected_ids) != 1000 or len(set(expected_ids)) != 1000:
        raise ValueError(f"invalid holdout task IDs: {branch}")
    seen_ids = []
    best_ranks = []
    correct_masses = []
    for shard in range(40):
        stem = f"part-{shard:05d}"
        receipt_path = base / "metrics" / branch / f"{stem}.receipt.json"
        metric_path = base / "metrics" / branch / f"{stem}.jsonl"
        ranking_path = base / "rankings" / branch / f"{stem}.jsonl.gz"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        binding = receipt.get("binding", {})
        if (receipt.get("status") != "DONE" or receipt.get("branch") != branch or
                binding.get("freeze_sha256") != freeze_sha or binding.get("adapter_tree_sha256") != adapter_tree_sha or
                binding.get("holdout_sha256") != holdout["sha256"] or binding.get("seed") != seed or
                binding.get("arm") != arm or binding.get("split") != split):
            raise ValueError(f"evaluation binding mismatch: {branch}/{stem}")
        if not same_hash(digest(metric_path), receipt["metrics_sha256"]) or not same_hash(digest(ranking_path), receipt["ranking_sha256"]):
            raise ValueError(f"evaluation shard SHA mismatch: {branch}/{stem}")
        metrics = list(read_jsonl(metric_path))
        rankings = list(read_jsonl(ranking_path))
        if len(metrics) != len(rankings) or len(metrics) != 25:
            raise ValueError(f"evaluation shard incomplete: {branch}/{stem}")
        if [row["task_id"] for row in metrics] != receipt["task_ids"]:
            raise ValueError(f"evaluation shard task IDs differ from receipt: {branch}/{stem}")
        for metric, raw in zip(metrics, rankings):
            if metric["task_id"] != raw["task_id"] or metric["task_fingerprint"] != raw["task_fingerprint"]:
                raise ValueError(f"evaluation task mismatch: {branch}")
            best = validate_ranking(raw["ranking"])
            if metric["best_rank"] != best or metric["hit@32"] != float(best <= 32):
                raise ValueError(f"evaluation metric mismatch: {branch}")
            source = expected_by_id[metric["task_id"]]
            for candidate in raw["ranking"]:
                is_correct = trajectory(source["start"], candidate["program"], source["p"])[-1] == tuple(source["target"])
                if candidate["correct"] != is_correct:
                    raise ValueError(f"program correctness mismatch: {branch}/{metric['task_id']}")
            scores = [candidate["score"] for candidate in raw["ranking"]]
            peak = max(scores)
            weights = [math.exp(score - peak) for score in scores]
            mass = sum(weight for weight, candidate in zip(weights, raw["ranking"]) if candidate["correct"]) / sum(weights)
            if abs(mass - metric["correct_mass"]) > 1e-10:
                raise ValueError(f"correct probability mass mismatch: {branch}/{metric['task_id']}")
            correct_masses.append(mass)
            seen_ids.append(metric["task_id"])
            best_ranks.append(best)
    if seen_ids != expected_ids:
        raise ValueError(f"evaluation task order differs from frozen holdout: {branch}")
    curve = hit_curve(best_ranks)
    if abs(curve[31] - summary["hit@32"]) > 1e-12:
        raise ValueError(f"summary Hit@32 mismatch: {branch}")
    if abs(statistics.mean(correct_masses) - summary["correct_mass"]) > 1e-10:
        raise ValueError(f"summary correct mass mismatch: {branch}")
    return {"branch": branch, "hit32": curve[31], "curve": curve,
            "correct_mass": summary["correct_mass"], "best_ranks": best_ranks}


def audit_atomic(seed: int, arm: str, freeze_sha: str, adapter_tree_sha: str,
                 k: int = 1, subset: int = 1) -> dict:
    branch = f"{branch_prefix(k, subset, seed, arm)}_atomic"
    path = ROOT / "rankings/atomic" / f"{branch}.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    raw_path = ROOT / "rankings/atomic" / f"{branch}.generations.jsonl"
    binding = result.get("binding", {})
    if (result.get("status") != "DONE" or binding.get("freeze_sha256") != freeze_sha or
            binding.get("adapter_tree_sha256") != adapter_tree_sha or binding.get("seed") != seed or
            binding.get("arm") != arm or not same_hash(result.get("generation_sha256"), digest(raw_path))):
        raise ValueError(f"atomic evaluation invalid: {branch}")
    raw_rows = list(read_jsonl(raw_path))
    source_rows = list(read_jsonl(CONFIRM / "data/confirm_atomic.jsonl"))
    if len(raw_rows) != result["tasks"] or len(source_rows) != result["tasks"]:
        raise ValueError(f"atomic generation count mismatch: {branch}")
    if any(raw.get("binding") != binding or raw.get("branch") != branch or raw.get("seed") != seed
           for raw in raw_rows):
        raise ValueError(f"atomic raw binding mismatch: {branch}")
    derived = derive_atomic_rows(source_rows, raw_rows)
    for metric in ("plan", "apply"):
        stored = result[metric]
        if set(derived[metric]) != set(stored["by_operation"]):
            raise ValueError(f"atomic operation coverage mismatch: {branch}/{metric}")
        for operation, value in derived[metric].items():
            if abs(value - stored["by_operation"][operation]) > 1e-12:
                raise ValueError(f"atomic per-operation metric mismatch: {branch}/{metric}/{operation}")
        overall = sum(int(raw["plan_correct"] if metric == "plan" else raw["correct"]) for raw in raw_rows) / len(raw_rows)
        if abs(overall - stored["overall"]) > 1e-12:
            raise ValueError(f"atomic overall metric mismatch: {branch}/{metric}")
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if digest(ROOT / "FREEZE.json") != FREEZE_SHA256:
        raise ValueError("frozen manifest SHA mismatch")
    freeze = json.loads((ROOT / "FREEZE.json").read_text(encoding="utf-8"))
    entry = next(row for row in freeze["entries"] if row["k"] == row["subset"] == 1)
    base_atomic = json.loads((CONFIRM / "rankings/replicateshared/atomic/replicateshared_atomic_base.json")
                             .read_text(encoding="utf-8"))
    if (base_atomic.get("status") != "DONE" or
            base_atomic.get("binding", {}).get("atomic_export_sha256") != freeze["atomic_export_payload_sha256"] or
            base_atomic.get("binding", {}).get("data_sha256") != digest(CONFIRM / "data/confirm_atomic.jsonl")):
        raise ValueError("frozen atomic baseline missing or bound to different checkpoint/tasks")
    rows = []
    curves = []
    atomic_rows = []
    selection_rows = []
    source = {row["trajectory_id"]: row for row in read_jsonl(
        REPO / "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/data/discover_trajectories.jsonl")}
    pair = tuple(entry["pairs"][0])
    for seed in (85000, 85001, 85002):
        for arm in ("random", "withheld"):
            selection = entry["selections"][str(seed)]
            identities = selection[f"{arm}_ids"]
            if len(identities) != 750 or len(set(identities)) != 750:
                raise ValueError(f"selection incomplete: {seed}/{arm}")
            selection_rows.append({"seed": seed, "arm": arm, "selected_trajectories": len(identities),
                                   "removed_trajectories": len(selection[f"{arm}_removed_ids"]),
                                   "selected_with_excluded_pair": selected_pair_count(source, identities, pair),
                                   "composition_target_tokens_per_epoch": selection["budget"]["target_tokens_per_epoch_each"]})
            adapter_tree = audit_training(seed, arm, FREEZE_SHA256)
            for split in ("withheld", "control"):
                result = audit_ranking(seed, arm, split, entry["holdouts"][split], FREEZE_SHA256, adapter_tree)
                rows.append({"seed": seed, "arm": arm, "split": split, "hit32": result["hit32"],
                             "correct_mass": result["correct_mass"]})
                curves.extend({"seed": seed, "arm": arm, "split": split, "k": k, "hit_k": value}
                              for k, value in enumerate(result["curve"], 1))
            atomic = audit_atomic(seed, arm, FREEZE_SHA256, adapter_tree)
            for metric in ("plan", "apply"):
                for operation, accuracy in atomic[metric]["by_operation"].items():
                    atomic_rows.append({"seed": seed, "arm": arm, "metric": metric, "operation": operation,
                                        "accuracy": accuracy,
                                        "drop_from_frozen_atomic": base_atomic[metric]["by_operation"][operation] - accuracy})
    differences = []
    for seed in (85000, 85001, 85002):
        matched = {row["arm"]: row["hit32"] for row in rows if row["seed"] == seed and row["split"] == "withheld"}
        differences.append(matched["withheld"] - matched["random"])
    analysis = {"schema": "withheld-pairs.first-block-audited.v1", "status": "VALID",
                "freeze_sha256": FREEZE_SHA256, "k": 1, "subset": 1,
                "completed_trainings": 6, "planned_trainings": 120,
                "completed_evaluations": 6, "withheld_hit32_difference": paired_stats(differences),
                "per_seed_differences": dict(zip((85000, 85001, 85002), differences)),
                "interpretation": "first set only; no inference over k or pair subsets"}
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "seed_metrics.csv", rows)
    write_csv(OUT / "selection_integrity.csv", selection_rows)
    write_csv(OUT / "hit_k.csv", curves)
    write_csv(OUT / "atomic_forgetting.csv", atomic_rows)
    (OUT / "analysis_summary.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(analysis, ensure_ascii=False))


if __name__ == "__main__":
    main()
