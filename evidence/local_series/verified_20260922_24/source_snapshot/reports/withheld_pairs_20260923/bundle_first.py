"""Build a compact, hash-verified first-block audit ZIP without model weights."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

from analyze_first import FREEZE_SHA256, audit_training, digest, same_hash


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ART = REPO / "artifacts/withheld_pairs_20260923"
OUT = HERE / "data/first_block/first_block_audit_v2.zip"


def audit_file_candidates(repo: Path) -> list[Path]:
    report = repo / "reports/withheld_pairs_20260923"
    artifact = repo / "artifacts/withheld_pairs_20260923"
    files = []
    for pattern in ("*.py", "*.md", "data/first_block/*", "data/run_index.csv"):
        files.extend(path for path in report.glob(pattern) if path.is_file() and path.suffix not in (".zip", ".sha256"))
    for code_dir in (
        "stage4_distill_v5_0p6b_exploratory_sh1_72p8",
        "stage4_sh1_v5_0p6b",
        "stage4_distill_v3",
        "stage4_distill_v2",
        "stage4_distill_v1",
    ):
        files.extend(path for path in (repo / "artifacts" / code_dir / "code").glob("*.py") if path.is_file())
    files.append(artifact / "FREEZE.json")
    for pattern in ("holdouts/*.jsonl", "recovery/**/*"):
        files.extend(path for path in artifact.glob(pattern) if path.is_file())
    files.extend(path for path in artifact.glob("runs/**/*") if path.is_file() and
                 ("k1_s1_" in path.name or "k1s1_" in path.name or
                  path.name in ("queue_first.jsonl", "queue.stdout.log", "queue.stderr.log")))
    files.extend(path for path in artifact.glob("rankings/**/*") if path.is_file() and
                 "k1_s1_" in path.as_posix())
    files.extend(path for path in artifact.glob("failures/**/*") if path.is_file() and
                 "k1_s1_" in path.as_posix())
    for pattern in ("adapters/k1_s1_*/training_receipt.json", "adapters/k1_s1_*/adapter_config.json"):
        files.extend(path for path in artifact.glob(pattern) if path.is_file())
    files.extend((repo / "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/data/discover_trajectories.jsonl",
                  repo / "reports/trajectory_diversity_20260922/WITHHELD_PAIR_SETS.json",
                  repo / "artifacts/stage4_composition_confirm_v1_0p6b/data/confirm_atomic.jsonl",
                  repo / "artifacts/stage4_composition_confirm_v1_0p6b/rankings/replicateshared/atomic/replicateshared_atomic_base.json"))
    return sorted(set(path for path in files if path.is_file()), key=lambda path: path.relative_to(repo).as_posix())


def require_docker_evidence(artifact: Path) -> None:
    log_dir = artifact / "runs/docker"
    for mode in ("train", "eval"):
        for seed in (85000, 85001, 85002):
            for arm in ("random", "withheld"):
                name = f"verifier_withheld_{mode}_20260923_k1s1_{seed}_{arm}"
                for suffix in ("inspect.json", "stdout.log", "stderr.log"):
                    path = log_dir / f"{name}.{suffix}"
                    if not path.is_file():
                        raise ValueError(f"Docker evidence missing: {path}")


def require_queue_evidence(artifact: Path) -> None:
    for name in ("queue_first.jsonl", "queue.stdout.log", "queue.stderr.log"):
        path = artifact / "runs" / name
        if not path.is_file():
            raise ValueError(f"Queue evidence missing: {path}")


def require_complete() -> dict:
    summary_path = HERE / "data/first_block/analysis_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("status") != "VALID" or summary.get("completed_trainings") != 6 or
            summary.get("completed_evaluations") != 6 or summary.get("freeze_sha256") != FREEZE_SHA256):
        raise ValueError("first-block analysis is not complete")
    if not same_hash(digest(ART / "FREEZE.json"), FREEZE_SHA256):
        raise ValueError("frozen manifest changed")
    freeze = json.loads((ART / "FREEZE.json").read_text(encoding="utf-8"))
    bound_inputs = (("artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/data/discover_trajectories.jsonl",
                     "source_sha256"),
                    ("reports/trajectory_diversity_20260922/WITHHELD_PAIR_SETS.json", "pair_sets_sha256"),
                    ("artifacts/stage4_composition_confirm_v1_0p6b/data/confirm_atomic.jsonl",
                     "atomic_evaluation_sha256"))
    for relative, key in bound_inputs:
        if not same_hash(digest(REPO / relative), freeze[key]):
            raise ValueError(f"frozen input SHA mismatch: {relative}")
    with (HERE / "data/run_index.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 120:
        raise ValueError("run index is not complete")
    first = [row for row in rows if row["k"] == "1" and row["subset"] == "1"]
    if len(first) != 6 or any(any(row[field] != "DONE" for field in
                                  ("training_status", "withheld_evaluation_status",
                                   "control_evaluation_status", "atomic_evaluation_status")) for row in first):
        raise ValueError("first-block run index contains incomplete jobs")
    if len(list((ART / "rankings/summaries").glob("k1_s1_seed*_*.json"))) != 12:
        raise ValueError("first-block ranking summaries incomplete")
    if len(list((ART / "rankings/atomic").glob("k1_s1_seed*_atomic.json"))) != 6:
        raise ValueError("first-block atomic evaluations incomplete")
    require_docker_evidence(ART)
    require_queue_evidence(ART)
    return summary


def checkpoint_inventory() -> list[dict]:
    result = []
    for seed in (85000, 85001, 85002):
        for arm in ("random", "withheld"):
            tree = audit_training(seed, arm, FREEZE_SHA256)
            adapter = ART / "adapters" / f"k1_s1_seed{seed}_{arm}"
            weights = adapter / "adapter_model.safetensors"
            result.append({"seed": seed, "arm": arm, "adapter_path": adapter.relative_to(REPO).as_posix(),
                           "adapter_tree_sha256": tree, "weights_sha256": digest(weights),
                           "weights_bytes": weights.stat().st_size,
                           "weights_in_audit_zip": False})
    return result


def build_bundle() -> Path:
    require_complete()
    inventory = checkpoint_inventory()
    paths = audit_file_candidates(REPO)
    if not paths or any(path.is_symlink() for path in paths):
        raise ValueError("empty or symlinked audit inputs")
    manifest = [{"path": path.relative_to(REPO).as_posix(), "sha256": digest(path),
                 "bytes": path.stat().st_size} for path in paths]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="first-block-audit-", suffix=".zip",
                                         dir=OUT.parent, delete=False) as temp:
            temp_path = Path(temp.name)
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in paths:
                name = path.relative_to(REPO).as_posix()
                info = zipfile.ZipInfo(name, date_time=(2026, 9, 23, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
            for name, payload in (("AUDIT_MANIFEST.json", manifest), ("CHECKPOINT_INVENTORY.json", inventory)):
                info = zipfile.ZipInfo(name, date_time=(2026, 9, 23, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n")
        with zipfile.ZipFile(temp_path) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP CRC check failed")
            for item in manifest:
                payload = archive.read(item["path"])
                if len(payload) != item["bytes"] or hashlib.sha256(payload).hexdigest().upper() != item["sha256"]:
                    raise ValueError(f"ZIP content hash mismatch: {item['path']}")
        if OUT.exists():
            if digest(OUT) != digest(temp_path):
                raise ValueError("existing audit ZIP differs; use a new versioned name")
        else:
            os.link(temp_path, OUT)
        sidecar = OUT.with_suffix(".zip.sha256")
        line = f"{digest(OUT)}  {OUT.name}\n"
        if sidecar.exists():
            if sidecar.read_text(encoding="utf-8") != line:
                raise ValueError("existing audit ZIP sidecar differs")
        else:
            with sidecar.open("x", encoding="utf-8") as handle:
                handle.write(line)
        return OUT
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    target = build_bundle()
    print(json.dumps({"status": "VALID", "archive": str(target), "sha256": digest(target),
                      "bytes": target.stat().st_size}))
