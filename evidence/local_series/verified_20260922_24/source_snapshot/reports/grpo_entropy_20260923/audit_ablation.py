"""Independent CPU-only audit of the paired entropy-bonus experiment."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import io
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vbexp.verifier import verify  # noqa: E402
from vbexp.io import read_tasks  # noqa: E402

OPS = ("SH1", "SC2", "REV", "AC1", "AX1")
PROGRAMS = set(itertools.product(OPS, repeat=3))
T_CRITICAL_DF2 = 4.302652729911275


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted((item for item in root.rglob("*") if item.is_file()),
                   key=lambda item: item.relative_to(root).as_posix())
    if not files:
        raise ValueError("empty adapter tree")
    for item in files:
        digest.update(item.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest().upper()


def jsonl(path: Path):
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def group_metrics(programs: list[tuple[str, ...]], rewards: list[bool]) -> dict:
    if len(programs) != 8 or len(rewards) != 8 or any(type(x) is not bool for x in rewards):
        raise ValueError("one group requires eight program outcomes")
    counts = Counter(programs)
    correct = Counter(program for program, reward in zip(programs, rewards) if reward)
    total_correct = sum(rewards)
    return {
        "group_class": "all_wrong" if total_correct == 0 else
                       "all_correct" if total_correct == 8 else "mixed",
        "correct_count": total_correct,
        "unique_programs": len(counts),
        "unique_correct_programs": len(correct),
        "sample_entropy_nats": -sum((n / 8) * math.log(n / 8) for n in counts.values()),
        "collision_fraction": sum(n * (n - 1) // 2 for n in counts.values()) / 28,
    }


def _logsumexp(values: list[float]) -> float:
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def audit_ranked_task(task, ranking: list[dict]) -> dict:
    if len(ranking) != 125:
        raise ValueError("ranking does not contain 125 programs")
    seen = []
    scores = []
    correct_scores = []
    best_rank = None
    for index, row in enumerate(ranking, 1):
        program = tuple(row["program"])
        if program not in PROGRAMS:
            raise ValueError("invalid program in ranking")
        seen.append(program)
        score = row["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("non-finite or nonnumeric score")
        scores.append(float(score))
        if row.get("rank") != index:
            raise ValueError("noncontiguous rank")
        if type(row.get("correct")) is not bool:
            raise ValueError("non-Boolean correctness flag")
        expected = verify(task, "PROGRAM: " + " ".join(program))
        if not expected.parse_ok or row["correct"] != expected.is_correct:
            raise ValueError("incorrect correctness flag")
        if row["correct"]:
            correct_scores.append(float(score))
            if best_rank is None:
                best_rank = index
    if set(seen) != PROGRAMS or len(set(seen)) != 125:
        raise ValueError("ranking has missing or duplicate program")
    if list(zip(seen, scores)) != sorted(zip(seen, scores),
                                        key=lambda item: (-item[1], item[0])):
        raise ValueError("ranking is not sorted by score and lexical tie-break")
    if not correct_scores or best_rank is None:
        raise ValueError("task has no correct program")
    log_z = _logsumexp(scores)
    log_correct_z = _logsumexp(correct_scores)
    probabilities = [math.exp(score - log_z) for score in scores]
    correct_probs = [math.exp(score - log_correct_z) for score in correct_scores]
    entropy = -sum(p * math.log(p) for p in probabilities if p)
    correct_entropy = -sum(p * math.log(p) for p in correct_probs if p)
    return {
        "best_rank": best_rank, "hit@32": float(best_rank <= 32),
        "correct_mass": math.exp(log_correct_z - log_z),
        "entropy_nats": entropy, "effective_programs": math.exp(entropy),
        "correct_entropy_nats": correct_entropy,
        "unique_correct_programs": len(correct_scores),
    }


def paired_stats(differences: list[float]) -> dict:
    if len(differences) != 3 or not all(math.isfinite(x) for x in differences):
        raise ValueError("three finite paired differences required")
    mean = statistics.mean(differences)
    sd = statistics.stdev(differences)
    margin = T_CRITICAL_DF2 * sd / math.sqrt(3)
    extreme = sum(abs(sum(sign * value for sign, value in zip(signs, differences)) / 3)
                  >= abs(mean) - 1e-12
                  for signs in itertools.product((-1, 1), repeat=3))
    return {"n": 3, "mean_difference": mean, "sd": sd,
            "ci95_low": mean - margin, "ci95_high": mean + margin,
            "exact_signflip_two_sided_p": extreme / 8,
            "positive_pairs": sum(x > 0 for x in differences),
            "negative_pairs": sum(x < 0 for x in differences)}


def _close(actual, expected, name: str, tolerance: float = 1e-10) -> None:
    if not math.isfinite(float(actual)) or abs(float(actual) - float(expected)) > tolerance:
        raise ValueError(f"{name} differs from independently derived value")


def _run_name(seed: int, arm: str) -> str:
    return f"seed{seed}_{arm}_fp32"


def _read_receipt(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "DONE":
        raise ValueError(f"receipt is not DONE: {path}")
    return value


def audit_training(repo: Path, seed: int, arm: str, gpu_uuid: str,
                   training_tasks: list) -> tuple[dict, list[dict], dict]:
    root = repo / "artifacts/grpo_entropy_20260923"
    run = root / "runs" / _run_name(seed, arm)
    if (run / "FAILED.json").exists():
        raise ValueError(f"failed training cannot enter audit: {run}")
    done = _read_receipt(run / "DONE.json")
    config = done["config"]
    saved_config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if (config != saved_config or done.get("steps") != 150 or done.get("groups") != 150 or
            done.get("answers") != 1200 or config.get("seed") != seed or
            config.get("gpu_uuid") != gpu_uuid or
            config.get("entropy_coef") != (0.0 if arm == "control" else 0.01) or
            config.get("group_size") != 8 or config.get("temperature") != 0.7 or
            config.get("method") != "iid_action" or
            config.get("compute_dtype") != "torch.float32" or
            config.get("reference_dtype") != "torch.float32" or
            config.get("gradient_checkpointing") != "non_reentrant_trainable_model" or
            config.get("input_sha256") != sha256(repo / "artifacts/data/pilot/rl_train.jsonl")):
        raise ValueError(f"training configuration or budget differs: {run}")
    if (sha256(run / "metrics.jsonl") != done["metrics_sha256"] or
            sha256(run / "generations.jsonl") != done["generations_sha256"] or
            tree_sha256(run / "final_adapter") != done["adapter_tree_sha256"]):
        raise ValueError(f"training artifact SHA-256 differs: {run}")
    metrics = list(jsonl(run / "metrics.jsonl"))
    generated = list(jsonl(run / "generations.jsonl"))
    if len(metrics) != 150 or len(generated) != 1200:
        raise ValueError("training metrics or generated-answer count incomplete")
    reached_tasks = set()
    correct_programs = set()
    group_rows = []
    for index, metric in enumerate(metrics):
        step = index + 1
        task = training_tasks[index % len(training_tasks)]
        group = generated[index * 8:(index + 1) * 8]
        if metric.get("step") != step or metric.get("task_id") != task.task_id:
            raise ValueError("training metric task order differs")
        programs = []
        rewards = []
        for candidate_id, row in enumerate(group):
            program = tuple(row["program"])
            if (row.get("step") != step or row.get("task_id") != task.task_id or
                    row.get("candidate_id") != candidate_id or
                    len(program) != task.max_steps or any(op not in OPS for op in program) or
                    type(row.get("correct")) is not bool):
                raise ValueError("generated program identity, length, or reward type differs")
            outcome = verify(task, "PROGRAM: " + " ".join(program))
            if not outcome.parse_ok or row["correct"] != outcome.is_correct:
                raise ValueError("generated reward differs from exact verifier")
            programs.append(program)
            rewards.append(row["correct"])
            if outcome.is_correct:
                reached_tasks.add(task.task_id)
                correct_programs.add(program)
        derived = group_metrics(programs, rewards)
        for key, value in derived.items():
            if isinstance(value, float):
                _close(metric[key], value, key)
            elif metric[key] != value:
                raise ValueError(f"group metric {key} differs")
        _close(metric["mean_reward"], sum(rewards) / 8, "mean_reward")
        if (metric.get("cumulative_tasks_reached") != len(reached_tasks) or
                metric.get("cumulative_unique_correct_programs") != len(correct_programs)):
            raise ValueError("training coverage trajectory differs")
        if any(not math.isfinite(float(metric[key])) for key in
               ("loss", "mean_kl", "mean_policy_entropy_normalized", "wall_seconds")):
            raise ValueError("non-finite training metric")
        group_rows.append({"seed": seed, "arm": arm, "step": step,
                           "task_id": task.task_id, **derived,
                           "mean_reward": metric["mean_reward"],
                           "cumulative_tasks_reached": len(reached_tasks),
                           "cumulative_unique_correct_programs": len(correct_programs)})
    summary = {"seed": seed, "arm": arm, "training_steps": 150,
               "training_groups": 150, "training_answers": 1200,
               "groups_all_wrong": sum(row["group_class"] == "all_wrong" for row in group_rows),
               "groups_mixed": sum(row["group_class"] == "mixed" for row in group_rows),
               "groups_all_correct": sum(row["group_class"] == "all_correct" for row in group_rows),
               "reached_training_tasks": len(reached_tasks),
               "unique_correct_training_programs": len(correct_programs),
               "training_adapter_tree_sha256": done["adapter_tree_sha256"],
               "gpu_uuid": gpu_uuid}
    return summary, group_rows, config


def audit_evaluation(repo: Path, seed: int, arm: str, gpu_uuid: str,
                     holdout: list, adapter_sha: str) -> tuple[dict, list[dict], list[dict]]:
    root = repo / "artifacts/grpo_entropy_20260923"
    eval_dir = root / "runs" / _run_name(seed, arm) / "evaluation"
    if (eval_dir / "FAILED.json").exists():
        raise ValueError("failed evaluation cannot enter audit")
    summary = _read_receipt(eval_dir / "DONE.json")
    if (summary.get("tasks") != 1000 or summary.get("candidate_program_scores") != 125000 or
            summary.get("gpu_uuid") != gpu_uuid or
            summary.get("adapter_tree_sha256") != adapter_sha or
            summary.get("holdout_sha256") != sha256(root / "holdout_1000.jsonl")):
        raise ValueError("evaluation source binding or task count differs")
    task_rows = []
    for shard_index in range(40):
        shard_tasks = holdout[shard_index * 25:(shard_index + 1) * 25]
        path = eval_dir / "shards" / f"part-{shard_index:05d}.jsonl.gz"
        receipt = _read_receipt(eval_dir / "shards" / f"part-{shard_index:05d}.receipt.json")
        if (receipt.get("tasks") != 25 or receipt.get("task_ids") !=
                [task.task_id for task in shard_tasks] or receipt.get("sha256") != sha256(path)):
            raise ValueError("evaluation shard identity or SHA-256 differs")
        rows = list(jsonl(path))
        if len(rows) != 25:
            raise ValueError("evaluation shard is incomplete")
        for task, row in zip(shard_tasks, rows):
            if row.get("task_id") != task.task_id:
                raise ValueError("evaluation task order differs")
            derived = audit_ranked_task(task, row["ranking"])
            for key, value in derived.items():
                if isinstance(value, float):
                    _close(row["metrics"][key], value, key)
                elif row["metrics"][key] != value:
                    raise ValueError(f"evaluation metric {key} differs")
            task_rows.append({"seed": seed, "arm": arm, "task_id": task.task_id,
                              **derived})
    ranks = [row["best_rank"] for row in task_rows]
    curve = [{"seed": seed, "arm": arm, "k": k,
              "hit_k": sum(rank <= k for rank in ranks) / 1000}
             for k in range(1, 126)]
    for row in curve:
        _close(summary["hit@k"][str(row["k"])], row["hit_k"], "hit@k", 1e-12)
    _close(summary["hit@32"], curve[31]["hit_k"], "hit@32", 1e-12)
    for key, source in (("mean_correct_mass", "correct_mass"),
                        ("mean_entropy_nats", "entropy_nats"),
                        ("mean_correct_entropy_nats", "correct_entropy_nats")):
        _close(summary[key], statistics.mean(row[source] for row in task_rows), key)
    return {"seed": seed, "arm": arm, "hit32": curve[31]["hit_k"],
            "mean_correct_mass": summary["mean_correct_mass"],
            "mean_entropy_nats": summary["mean_entropy_nats"],
            "mean_correct_entropy_nats": summary["mean_correct_entropy_nats"],
            "evaluation_tasks": 1000, "candidate_program_scores": 125000}, task_rows, curve


def audit_docker_evidence(repo: Path, seed: int, arm: str, gpu_index: int) -> None:
    evidence = repo / "artifacts/grpo_entropy_20260923/queue_evidence"
    for mode in ("train", "eval"):
        name = f"verifier_grpo_entropy_{_run_name(seed, arm)}"
        if mode == "eval":
            name += "_eval"
        info = json.loads((evidence / f"{name}.inspect.json").read_text(encoding="utf-8"))
        ids = [str(value) for request in info.get("HostConfig", {}).get("DeviceRequests", [])
               for value in request.get("DeviceIDs", [])]
        if (info["State"]["Status"] != "exited" or info["State"]["ExitCode"] != 0 or
                ids != [str(gpu_index)] or not (evidence / f"{name}.stdout.log").is_file() or
                not (evidence / f"{name}.stderr.log").is_file()):
            raise ValueError(f"Docker evidence incomplete or GPU binding differs: {name}")


def _csv_bytes(rows: list[dict]) -> bytes:
    if not rows:
        raise ValueError("empty audit table")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"audit output already exists with different content: {path}")
        return
    path.write_bytes(data)


def run(repo: Path, out: Path) -> dict:
    root = repo / "artifacts/grpo_entropy_20260923"
    duration = json.loads((repo / "reports/grpo_entropy_20260923/DURATION_FREEZE_FP32.json")
                          .read_text(encoding="utf-8"))
    if (duration.get("status") != "FROZEN_BEFORE_FULL_PAIR" or
            duration.get("steps_per_arm") != 150 or
            duration.get("heldout_outcomes_viewed_before_decision") is not False):
        raise ValueError("blind duration freeze is invalid")
    training_path = repo / "artifacts/data/pilot/rl_train.jsonl"
    holdout_path = root / "holdout_1000.jsonl"
    training_tasks = read_tasks(training_path)
    holdout = read_tasks(holdout_path)
    if (len(training_tasks) != 500 or len(holdout) != 1000 or
            len({task.task_id for task in holdout}) != 1000 or
            any(task.max_steps != 3 for task in holdout)):
        raise ValueError("training or held-out task inventory differs")
    seed_rows = []
    group_rows = []
    task_rows = []
    curves = []
    blocks = []
    for gpu_index, seeds in ((1, (0, 2)), (2, (1,))):
        allocation = json.loads((root / f"queue_gpu{gpu_index}_fp32_allocation.json")
                                .read_text(encoding="utf-8"))
        if (allocation.get("seeds") != list(seeds) or allocation.get("gpu_index") != gpu_index or
                allocation.get("steps_per_arm") != 150 or
                allocation.get("duration_freeze_sha256") !=
                sha256(repo / "reports/grpo_entropy_20260923/DURATION_FREEZE_FP32.json")):
            raise ValueError("queue GPU allocation or duration binding differs")
        gpu_uuid = allocation["gpu_uuid"]
        events = list(jsonl(root / f"queue_gpu{gpu_index}_fp32.jsonl"))
        if not events or events[-1].get("status") != "QUEUE_DONE":
            raise ValueError("host-side queue has not completed")
        for seed in seeds:
            configs = {}
            for arm in ("control", "entropy"):
                train_summary, groups, configs[arm] = audit_training(
                    repo, seed, arm, gpu_uuid, training_tasks)
                eval_summary, tasks, curve = audit_evaluation(
                    repo, seed, arm, gpu_uuid, holdout,
                    train_summary["training_adapter_tree_sha256"])
                audit_docker_evidence(repo, seed, arm, gpu_index)
                seed_rows.append({**train_summary, **eval_summary})
                group_rows.extend(groups)
                task_rows.extend(tasks)
                curves.extend(curve)
                for offset in range(0, 150, 25):
                    chunk = groups[offset:offset + 25]
                    blocks.append({"seed": seed, "arm": arm, "start_step": offset + 1,
                                   "end_step": offset + 25,
                                   "fraction_all_wrong": sum(x["group_class"] == "all_wrong" for x in chunk) / 25,
                                   "fraction_mixed": sum(x["group_class"] == "mixed" for x in chunk) / 25,
                                   "fraction_all_correct": sum(x["group_class"] == "all_correct" for x in chunk) / 25,
                                   "mean_group_reward": statistics.mean(x["mean_reward"] for x in chunk),
                                   "mean_unique_programs": statistics.mean(x["unique_programs"] for x in chunk),
                                   "mean_sample_entropy_nats": statistics.mean(x["sample_entropy_nats"] for x in chunk),
                                   "reached_training_tasks": chunk[-1]["cumulative_tasks_reached"],
                                   "unique_correct_training_programs":
                                       chunk[-1]["cumulative_unique_correct_programs"]})
            left = {key: value for key, value in configs["control"].items() if key != "entropy_coef"}
            right = {key: value for key, value in configs["entropy"].items() if key != "entropy_coef"}
            if left != right:
                raise ValueError(f"paired training configurations differ for seed {seed}")
    differences = []
    for seed in (0, 1, 2):
        values = {row["arm"]: row["hit32"] for row in seed_rows if row["seed"] == seed}
        if set(values) != {"control", "entropy"}:
            raise ValueError("incomplete paired seed")
        differences.append(values["entropy"] - values["control"])
    result = {"schema": "grpo-entropy.complete-paired-audit.v1", "status": "VALID",
              "algorithm": "group_normalized_categorical_policy_gradient_without_clipping",
              "duration_freeze_sha256": sha256(repo / "reports/grpo_entropy_20260923/DURATION_FREEZE_FP32.json"),
              "holdout_sha256": sha256(holdout_path), "training_steps_per_arm": 150,
              "completed_trainings": 6, "completed_evaluations": 6,
              "candidate_program_scores": 750000,
              "primary_hit32_entropy_minus_control": paired_stats(differences),
              "per_seed_hit32_differences": dict(zip((0, 1, 2), differences)),
              "signflip_resolution_note": "n=3 gives minimum two-sided exact p=0.25"}
    write_once(out / "seed_metrics.csv", _csv_bytes(seed_rows))
    write_once(out / "group_steps.csv", _csv_bytes(group_rows))
    write_once(out / "group_blocks.csv", _csv_bytes(blocks))
    write_once(out / "task_metrics.csv", _csv_bytes(task_rows))
    write_once(out / "hit_k.csv", _csv_bytes(curves))
    write_once(out / "analysis_summary.json",
               (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.repo, args.out), ensure_ascii=False))


if __name__ == "__main__":
    main()
