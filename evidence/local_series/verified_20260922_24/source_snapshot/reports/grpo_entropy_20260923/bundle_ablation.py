"""Package a complete paired 150-step entropy ablation without model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest().upper()


def tree_digest(root: Path) -> str:
    value = hashlib.sha256()
    paths = sorted((path for path in root.rglob("*") if path.is_file()),
                   key=lambda path: path.relative_to(root).as_posix())
    if not paths:
        raise ValueError(f"final adapter missing: {root}")
    for path in paths:
        value.update(path.relative_to(root).as_posix().encode())
        value.update(b"\0")
        value.update(bytes.fromhex(digest(path)))
    return value.hexdigest().upper()


def candidates(repo: Path) -> list[Path]:
    report = repo / "reports/grpo_entropy_20260923"
    artifact = repo / "artifacts/grpo_entropy_20260923"
    paths = set()
    for root in (report, artifact):
        paths.update(path for path in root.rglob("*") if path.is_file() and
                     not any(part == "__pycache__" for part in path.parts) and
                     path.suffix not in (".zip", ".sha256", ".safetensors", ".pyc"))
    paths.add(repo / "artifacts/data/pilot/rl_train.jsonl")
    paths.add(repo / "scripts/train_action_grpo.py")
    paths.update(path for path in (repo / "src/vbexp").rglob("*.py") if path.is_file())
    return sorted((path for path in paths if path.is_file()),
                  key=lambda path: path.relative_to(repo).as_posix())


def checkpoint_inventory(repo: Path) -> list[dict]:
    root = repo / "artifacts/grpo_entropy_20260923/runs"
    inventory = []
    for seed in (0, 1, 2):
        for arm in ("control", "entropy"):
            run = root / f"seed{seed}_{arm}_fp32"
            adapter = run / "final_adapter"
            weights = adapter / "adapter_model.safetensors"
            receipt = run / "DONE.json"
            if not weights.is_file() or not receipt.is_file() or weights.is_symlink():
                raise ValueError(f"final adapter missing: {run}")
            done = json.loads(receipt.read_text(encoding="utf-8"))
            actual_tree = tree_digest(adapter)
            if done.get("status") != "DONE" or done.get("adapter_tree_sha256") != actual_tree:
                raise ValueError(f"final adapter tree SHA differs: {run}")
            inventory.append({
                "seed": seed, "arm": arm,
                "adapter_path": adapter.relative_to(repo).as_posix(),
                "adapter_tree_sha256": actual_tree,
                "weights_sha256": digest(weights), "weights_bytes": weights.stat().st_size,
                "weights_in_audit_zip": False,
            })
    return inventory


def validate_binding(repo: Path, summary: dict) -> None:
    holdout = repo / "artifacts/grpo_entropy_20260923/holdout_1000.jsonl"
    duration = repo / "reports/grpo_entropy_20260923/DURATION_FREEZE_FP32.json"
    if digest(holdout) != summary.get("holdout_sha256"):
        raise ValueError("holdout SHA differs from independent audit")
    if digest(duration) != summary.get("duration_freeze_sha256"):
        raise ValueError("duration freeze SHA differs from independent audit")


def require_complete(repo: Path) -> dict:
    report = repo / "reports/grpo_entropy_20260923"
    artifact = repo / "artifacts/grpo_entropy_20260923"
    summary = json.loads((report / "data/paired_ablation/analysis_summary.json").read_text(encoding="utf-8"))
    if (summary.get("status") != "VALID" or summary.get("completed_trainings") != 6 or
            summary.get("completed_evaluations") != 6 or
            summary.get("candidate_program_scores") != 750000):
        raise ValueError("independent audit is incomplete")
    for gpu_index in (1, 2):
        queue = artifact / f"queue_gpu{gpu_index}_fp32.jsonl"
        if not queue.is_file():
            raise ValueError(f"queue receipt missing: {queue}")
        events = [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not events or events[-1].get("status") != "QUEUE_DONE":
            raise ValueError(f"queue did not finish: {queue}")
    evidence = artifact / "queue_evidence"
    for seed in (0, 1, 2):
        gpu_index = 2 if seed == 1 else 1
        for arm in ("control", "entropy"):
            for mode in ("train", "evaluate"):
                name = f"verifier_grpo_entropy_seed{seed}_{arm}_fp32"
                if mode == "evaluate":
                    name += "_eval"
                for suffix in ("inspect.json", "stdout.log", "stderr.log"):
                    path = evidence / f"{name}.{suffix}"
                    if not path.is_file():
                        raise ValueError(f"Docker evidence missing: {path}")
                inspect = json.loads((evidence / f"{name}.inspect.json").read_text(encoding="utf-8"))
                state = inspect.get("State", {})
                ids = [str(device) for request in inspect.get("HostConfig", {}).get("DeviceRequests", [])
                       for device in request.get("DeviceIDs", [])]
                if state.get("Status") != "exited" or state.get("ExitCode") != 0 or ids != [str(gpu_index)]:
                    raise ValueError(f"Docker evidence invalid: {name}")
    validate_binding(repo, summary)
    return summary


def write_archive(repo: Path, target: Path, paths: list[Path], inventory: list[dict]) -> None:
    if not paths or any(not path.is_file() or path.is_symlink() for path in paths):
        raise ValueError("empty or symlinked archive inputs")
    manifest = [{"path": path.relative_to(repo).as_posix(), "sha256": digest(path),
                 "bytes": path.stat().st_size} for path in paths]
    if len({item["path"] for item in manifest}) != len(manifest):
        raise ValueError("duplicate archive member")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix="grpo-audit-", suffix=".zip",
                                         dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6) as handle:
            for path in paths:
                info = zipfile.ZipInfo(path.relative_to(repo).as_posix(),
                                       date_time=(2026, 9, 23, 0, 0, 0))
                handle.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED,
                                compresslevel=6)
            for name, payload in (("AUDIT_MANIFEST.json", manifest),
                                  ("CHECKPOINT_INVENTORY.json", inventory)):
                info = zipfile.ZipInfo(name, date_time=(2026, 9, 23, 0, 0, 0))
                handle.writestr(info, json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n")
        with zipfile.ZipFile(temporary) as handle:
            if handle.testzip() is not None:
                raise ValueError("ZIP CRC check failed")
            for item in manifest:
                payload = handle.read(item["path"])
                if len(payload) != item["bytes"] or hashlib.sha256(payload).hexdigest().upper() != item["sha256"]:
                    raise ValueError(f"ZIP member SHA mismatch: {item['path']}")
        if target.exists():
            if digest(target) != digest(temporary):
                raise ValueError("existing archive differs; use a versioned name")
        else:
            os.link(temporary, target)
        sidecar = target.with_suffix(".zip.sha256")
        line = f"{digest(target)}  {target.name}\n"
        if sidecar.exists():
            if sidecar.read_text(encoding="utf-8") != line:
                raise ValueError("existing archive sidecar differs")
        else:
            with sidecar.open("x", encoding="utf-8") as handle:
                handle.write(line)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_bundle(repo: Path, target: Path | None = None) -> Path:
    require_complete(repo)
    inventory = checkpoint_inventory(repo)
    if target is None:
        target = (repo / "reports/grpo_entropy_20260923/data/paired_ablation/"
                  "grpo_entropy_150step_paired_audit.zip")
    write_archive(repo, target, candidates(repo), inventory)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    target = build_bundle(args.repo, args.out)
    print(json.dumps({"status": "VALID", "archive": str(target),
                      "sha256": digest(target), "bytes": target.stat().st_size}))


if __name__ == "__main__":
    main()
