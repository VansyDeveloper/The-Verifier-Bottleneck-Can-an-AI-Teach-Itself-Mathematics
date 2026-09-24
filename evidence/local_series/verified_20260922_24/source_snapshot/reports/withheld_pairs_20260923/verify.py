"""Independent integrity checks for the withheld-pair study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUTPUT = REPO / "artifacts/withheld_pairs_20260923"


def validate_holdout_file(base: Path, metadata: dict) -> list[dict]:
    path = base / metadata["path"]
    if not path.is_file():
        raise ValueError(f"holdout missing: {path}")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest().upper() != metadata["sha256"].upper():
        raise ValueError(f"holdout SHA-256 mismatch: {path}")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != metadata["rows"]:
        raise ValueError(f"holdout row count mismatch: {path}")
    if len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate holdout task: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    freeze = json.loads((args.output / "FREEZE.json").read_text(encoding="utf-8"))
    task_ids = set()
    task_fingerprints = set()
    feasible = 0
    for entry in freeze["entries"]:
        if entry["status"] != "FEASIBLE":
            continue
        feasible += 1
        for metadata in entry["holdouts"].values():
            rows = validate_holdout_file(args.output, metadata)
            ids = {row["task_id"] for row in rows}
            fps = {row["task_fingerprint"] for row in rows}
            if task_ids & ids or task_fingerprints & fps:
                raise ValueError("duplicate task across frozen holdouts")
            task_ids.update(ids)
            task_fingerprints.update(fps)
    print(json.dumps({"status": "VALID", "feasible_sets": feasible,
                      "holdout_tasks": len(task_ids)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
