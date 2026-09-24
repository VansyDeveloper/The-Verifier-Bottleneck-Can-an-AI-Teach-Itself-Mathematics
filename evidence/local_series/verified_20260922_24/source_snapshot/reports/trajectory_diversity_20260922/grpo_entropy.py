"""Empirical within-group diversity in six saved 400-step GRPO runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


RUNS = {
    "iid": {
        0: "20260801T081451Z_qwen3-0.6b_plan_grpo_iid_action_exact_seed0",
        1: "20260801T093401Z_qwen3-0.6b_plan_grpo_iid_action_exact_seed1",
        2: "20260801T105301Z_qwen3-0.6b_plan_grpo_iid_action_exact_seed2",
    },
    "prefix": {
        0: "20260801T085442Z_qwen3-0.6b_plan_grpo_prefix_balanced_action_exact_seed0",
        1: "20260801T101344Z_qwen3-0.6b_plan_grpo_prefix_balanced_action_exact_seed1",
        2: "20260801T113239Z_qwen3-0.6b_plan_grpo_prefix_balanced_action_exact_seed2",
    },
}
STEPS = 400
GROUP_SIZE = 8
BLOCK_SIZE = 25
ARCHIVE_SHA256 = "6a028af242859bffb8fab89dcaa4b88c063c39fd43bdd327af8540036b2fe61e"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _answer_key(row: dict) -> tuple:
    program = row.get("parsed_program")
    if isinstance(program, list):
        return ("program", *program)
    return ("unparsed", row["text"])


def _sample_entropy(counts: Counter) -> float:
    total = sum(counts.values())
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def group_entropy(rows: list[dict]) -> dict:
    if len(rows) != GROUP_SIZE:
        raise ValueError("GRPO group must have exactly eight answers")
    if not all(type(row.get("is_correct")) is bool for row in rows):
        raise ValueError("non-boolean correctness flag")
    counts = Counter(_answer_key(row) for row in rows)
    correct = Counter(_answer_key(row) for row in rows if row["is_correct"])
    equal_pairs = sum(count * (count - 1) // 2 for count in counts.values())
    return {
        "correct_count": sum(correct.values()),
        "unique_answers": len(counts),
        "sample_entropy_nats": _sample_entropy(counts),
        "collision_fraction": equal_pairs / 28,
        "unique_correct_answers": len(correct),
        "correct_sample_entropy_nats": _sample_entropy(correct) if correct else None,
    }


def _records(archive: zipfile.ZipFile, name: str, member: str) -> list[dict]:
    path = f"400step_raw_logs/{name}/{member}"
    return [json.loads(line) for line in archive.read(path).decode("utf-8").splitlines() if line.strip()]


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.archive).lower() != ARCHIVE_SHA256:
        raise ValueError("GRPO raw archive SHA-256 mismatch")
    rows = []
    with zipfile.ZipFile(args.archive) as archive:
        if archive.testzip() is not None:
            raise ValueError("GRPO ZIP integrity failure")
        for sampler, seed_names in RUNS.items():
            for seed, name in seed_names.items():
                generations = _records(archive, name, "generations.jsonl")
                metrics = _records(archive, name, "metrics.jsonl")
                if len(generations) != STEPS * GROUP_SIZE or len(metrics) != STEPS:
                    raise ValueError(f"{name}: wrong row count")
                by_step = defaultdict(list)
                for item in generations:
                    candidate_id = item["candidate_id"]
                    if type(candidate_id) is not int or not 0 <= candidate_id < STEPS * GROUP_SIZE:
                        raise ValueError(f"{name}: invalid candidate id")
                    by_step[candidate_id // GROUP_SIZE + 1].append(item)
                if sorted(by_step) != list(range(1, STEPS + 1)):
                    raise ValueError(f"{name}: missing step")
                reached = set()
                correct_programs = set()
                for step in range(1, STEPS + 1):
                    group = sorted(by_step[step], key=lambda item: item["candidate_id"])
                    if [item["candidate_id"] for item in group] != list(range((step - 1) * 8, step * 8)):
                        raise ValueError(f"{name}: duplicated or missing candidate")
                    if len({item["task_id"] for item in group}) != 1:
                        raise ValueError(f"{name}: mixed tasks in one group")
                    metric = metrics[step - 1]
                    if metric["step"] != step:
                        raise ValueError(f"{name}: metric step mismatch")
                    value = group_entropy(group)
                    reward = metric["metrics"]["mean_reward"]
                    if abs(reward - value["correct_count"] / GROUP_SIZE) > 1e-8:
                        raise ValueError(f"{name}: reward differs from correctness")
                    task_id = group[0]["task_id"]
                    if value["correct_count"]:
                        reached.add(task_id)
                    correct_programs.update(_answer_key(item) for item in group if item["is_correct"])
                    group_class = ("all_wrong" if value["correct_count"] == 0 else
                                   "all_correct" if value["correct_count"] == 8 else "mixed")
                    rows.append({"sampler": sampler, "seed": seed, "step": step,
                                 "block": (step - 1) // BLOCK_SIZE + 1,
                                 "task_id": task_id, "group_class": group_class,
                                 "mean_reward": reward, "cumulative_tasks_reached": len(reached),
                                 "cumulative_unique_correct_programs": len(correct_programs), **value})
    blocks = []
    for sampler in RUNS:
        for seed in range(3):
            for block in range(1, STEPS // BLOCK_SIZE + 1):
                group = [row for row in rows if row["sampler"] == sampler and row["seed"] == seed and row["block"] == block]
                positive = [row for row in group if row["correct_count"] > 0]
                blocks.append({"sampler": sampler, "seed": seed, "block": block,
                               "step_end": block * BLOCK_SIZE,
                               "sample_entropy_nats": sum(row["sample_entropy_nats"] for row in group) / BLOCK_SIZE,
                               "collision_fraction": sum(row["collision_fraction"] for row in group) / BLOCK_SIZE,
                               "unique_answers": sum(row["unique_answers"] for row in group) / BLOCK_SIZE,
                               "correct_entropy_positive_groups": (sum(row["correct_sample_entropy_nats"] for row in positive) / len(positive)) if positive else None,
                               "positive_groups": len(positive),
                               "all_wrong_fraction": sum(row["group_class"] == "all_wrong" for row in group) / BLOCK_SIZE,
                               "mixed_fraction": sum(row["group_class"] == "mixed" for row in group) / BLOCK_SIZE,
                               "all_correct_fraction": sum(row["group_class"] == "all_correct" for row in group) / BLOCK_SIZE,
                               "mean_reward": sum(row["mean_reward"] for row in group) / BLOCK_SIZE,
                               "cumulative_tasks_reached": group[-1]["cumulative_tasks_reached"],
                               "cumulative_unique_correct_programs": group[-1]["cumulative_unique_correct_programs"]})
    args.output.mkdir(parents=True, exist_ok=True)
    _csv(args.output / "grpo_groups.csv", rows)
    _csv(args.output / "grpo_blocks_per_seed.csv", blocks)
    summary = {"schema": "verifier-bottleneck.grpo-sample-entropy.v1", "status": "DONE",
               "runs": 6, "steps_per_run": STEPS, "group_size": GROUP_SIZE,
               "groups": len(rows), "answers": len(rows) * GROUP_SIZE,
               "archive_sha256": ARCHIVE_SHA256,
               "measure": "empirical entropy of eight sampled answers within one task group; not policy entropy",
               "output_sha256": {path.name: digest(path) for path in sorted(args.output.glob("grpo_*.csv"))}}
    (args.output / "grpo_audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
