"""Count the six prospectively paired FP32 runs without importing diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "artifacts/grpo_entropy_20260923"
OUT = HERE / "data/run_index.csv"


def marker_status(path: Path, training: bool) -> str:
    done = path / "DONE.json"
    failed = path / "FAILED.json"
    running = path / "RUNNING.json"
    if done.exists() and failed.exists():
        raise ValueError(f"contradictory DONE and FAILED markers: {path}")
    if failed.exists():
        if json.loads(failed.read_text(encoding="utf-8")).get("status") != "FAILED":
            raise ValueError(f"invalid FAILED marker: {failed}")
        return "FAILED"
    if done.exists():
        receipt = json.loads(done.read_text(encoding="utf-8"))
        expected = (("steps", 150), ("groups", 150), ("answers", 1200)) if training else (
            ("tasks", 1000), ("candidate_program_scores", 125000))
        if receipt.get("status") != "DONE" or any(receipt.get(key) != value for key, value in expected):
            raise ValueError(f"invalid DONE receipt: {done}")
        return "DONE"
    if running.exists():
        return "STARTED"
    return "NOT_STARTED"


def build_index(root: Path) -> list[dict]:
    rows = []
    for seed in (0, 1, 2):
        for arm in ("control", "entropy"):
            name = f"seed{seed}_{arm}_fp32"
            run = root / "runs" / name
            training = marker_status(run, training=True)
            evaluation = marker_status(run / "evaluation", training=False)
            if training != "DONE" and evaluation != "NOT_STARTED":
                raise ValueError(f"evaluation exists without accepted training: {name}")
            rows.append({"seed": seed, "arm": arm, "run_name": name,
                         "gpu_index": 2 if seed == 1 else 1,
                         "training_status": training,
                         "evaluation_status": evaluation})
    return rows


def main() -> None:
    rows = build_index(ROOT)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"planned_trainings": len(rows),
                      "completed_trainings": sum(row["training_status"] == "DONE" for row in rows),
                      "completed_evaluations": sum(row["evaluation_status"] == "DONE" for row in rows),
                      "failed_trainings": sum(row["training_status"] == "FAILED" for row in rows),
                      "failed_evaluations": sum(row["evaluation_status"] == "FAILED" for row in rows)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
