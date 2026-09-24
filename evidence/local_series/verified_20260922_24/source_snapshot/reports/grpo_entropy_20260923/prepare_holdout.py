"""Freeze 1,000 legacy-action depth-3 tasks before entropy-ablation training."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from vbexp.generator import GenerationSpec, generate_one


def semantic_key(row: dict) -> tuple:
    depth = row.get("max_steps") or row.get("depth") or row.get("difficulty", {}).get("depth")
    return (int(row["p"]), tuple(row["start"]), tuple(row["target"]), int(depth))


def validate_holdout(rows: list[dict], forbidden: set[tuple], expected_count: int) -> None:
    if len(rows) != expected_count:
        raise ValueError("wrong holdout count")
    keys = [semantic_key(row) for row in rows]
    if len(set(keys)) != len(keys) or len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate holdout task")
    if set(keys) & forbidden:
        raise ValueError("holdout leakage")
    for row in rows:
        if (row.get("mode") != "plan" or row.get("max_steps") != 3 or
                row.get("p") not in (11, 17) or row.get("degree_cap") not in (2, 3) or
                row.get("operations") != ["SH1", "SC2", "REV", "AC1", "AX1"]):
            raise ValueError("holdout task differs from frozen design")


def build_holdout(forbidden: set[tuple], count: int, seed: int) -> list[dict]:
    spec = GenerationSpec(
        split="grpo_entropy_holdout_20260923", mode="plan", primes=(11, 17),
        degree_caps=(2, 3), depths=(3,), require_order_sensitive=True,
        heldout_primes=(11, 17),
    )
    rng = random.Random(seed)
    rows: list[dict] = []
    used = set(forbidden)
    while len(rows) < count:
        row = generate_one(spec, rng).to_dict()
        key = semantic_key(row)
        if key in used:
            continue
        used.add(key)
        rows.append(row)
    validate_holdout(rows, forbidden, count)
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"frozen file differs: {path}")
        return
    path.write_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.repo_root
    sources = sorted((root / "artifacts/data/pilot").glob("*.jsonl"))
    sources += sorted((root / "artifacts/data/true_composition").glob("*.jsonl"))
    sources += [root / "artifacts/trajectory_diversity_20260922/data/holdout_final_a.jsonl"]
    forbidden: set[tuple] = set()
    source_hashes = {}
    for path in sources:
        source_hashes[path.relative_to(root).as_posix()] = sha256(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("target") is not None and row.get("start") is not None:
                forbidden.add(semantic_key(row))
    rows = build_holdout(forbidden, count=1000, seed=20260923)
    output = root / "artifacts/grpo_entropy_20260923/holdout_1000.jsonl"
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                      for row in rows).encode("utf-8")
    write_once(output, payload)
    receipt = {
        "schema": "grpo-entropy.holdout.v1", "status": "FROZEN", "seed": 20260923,
        "tasks": len(rows), "sha256": sha256(output), "source_sha256": source_hashes,
        "design": "depth=3,p=11|17,degree_cap=2|3,order_sensitive,legacy_action_protocol",
    }
    manifest = output.with_name("holdout_manifest.json")
    write_once(manifest, (json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"status": "FROZEN", "path": str(output), "tasks": len(rows),
                      "sha256": receipt["sha256"]}))


if __name__ == "__main__":
    main()
