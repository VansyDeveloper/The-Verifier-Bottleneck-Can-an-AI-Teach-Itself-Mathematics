"""Build a compact audit bundle for a complete post-amendment pair set."""

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


def candidates(repo: Path, k: int, subset: int) -> list[Path]:
    artifact = repo / "artifacts/withheld_pairs_20260923"
    report = repo / "reports/withheld_pairs_20260923"
    key = f"k{k}_s{subset}"
    branch = f"{key}_seed"
    files = set()
    files.update(path for path in report.glob("*.py") if path.is_file())
    files.update(path for path in report.glob("*.md") if path.is_file())
    for code_dir in (
        "stage4_distill_v5_0p6b_exploratory_sh1_72p8",
        "stage4_sh1_v5_0p6b",
        "stage4_distill_v3",
        "stage4_distill_v2",
        "stage4_distill_v1",
    ):
        files.update(path for path in (repo / "artifacts" / code_dir / "code").glob("*.py") if path.is_file())
    files.update(path for path in (report / "data" / key).glob("*")
                 if path.is_file() and path.suffix not in (".zip", ".sha256"))
    files.add(artifact / "FREEZE.json")
    files.update(path for path in (artifact / "holdouts").glob("*.jsonl") if path.is_file())
    for suffix in ("training_receipt.json", "adapter_config.json"):
        files.update(path for path in (artifact / "adapters").glob(f"{branch}*/{suffix}") if path.is_file())
    files.update(path for path in (artifact / "rankings").rglob("*")
                 if path.is_file() and branch in path.as_posix())
    files.update(path for path in (artifact / "runs").rglob("*")
                 if path.is_file() and (
                     branch in path.name or f"k{k}s{subset}_" in path.name or
                     path.name == f"{key}_allocation.json" or
                     path.name.startswith(f"queue_k{k}s{subset}.")))
    for relative in (
        "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/data/discover_trajectories.jsonl",
        "reports/trajectory_diversity_20260922/WITHHELD_PAIR_SETS.json",
        "artifacts/stage4_composition_confirm_v1_0p6b/data/confirm_atomic.jsonl",
        "artifacts/stage4_composition_confirm_v1_0p6b/rankings/replicateshared/atomic/replicateshared_atomic_base.json",
    ):
        files.add(repo / relative)
    return sorted((path for path in files if path.is_file()),
                  key=lambda path: path.relative_to(repo).as_posix())


def validate_docker_inspect(inspect: dict, gpu_index: int) -> None:
    state = inspect.get("State", {})
    if state.get("Status") != "exited" or state.get("ExitCode") != 0:
        raise ValueError("Docker exit state is not successful")
    requests = inspect.get("HostConfig", {}).get("DeviceRequests") or []
    device_ids = [str(device) for request in requests for device in request.get("DeviceIDs", [])]
    if device_ids != [str(gpu_index)]:
        raise ValueError("Docker GPU binding differs from allocation")


def require_evidence(repo: Path, k: int, subset: int) -> None:
    artifact = repo / "artifacts/withheld_pairs_20260923"
    summary_path = repo / "reports/withheld_pairs_20260923/data" / f"k{k}_s{subset}" / "analysis_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("status") != "VALID" or summary.get("k") != k or
            summary.get("subset") != subset or summary.get("completed_trainings") != 6 or
            summary.get("completed_evaluations") != 6):
        raise ValueError("independent set analysis is incomplete")
    queue = artifact / "runs" / f"queue_k{k}s{subset}.jsonl"
    events = [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not events or events[-1].get("status") != "SET_DONE":
        raise ValueError("host queue has not completed")
    log_dir = artifact / "runs/docker"
    inspect_paths = []
    for mode in ("train", "eval"):
        for seed in (85000, 85001, 85002):
            for arm in ("random", "withheld"):
                stem = f"verifier_withheld_{mode}_20260923_k{k}s{subset}_{seed}_{arm}"
                for suffix in ("inspect.json", "stdout.log", "stderr.log"):
                    path = log_dir / f"{stem}.{suffix}"
                    if not path.is_file():
                        raise ValueError(f"Docker evidence missing: {path}")
                inspect_paths.append(log_dir / f"{stem}.inspect.json")
    allocation = json.loads((artifact / "runs" / f"k{k}_s{subset}_allocation.json").read_text(encoding="utf-8"))
    for path in inspect_paths:
        inspect = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(inspect, list):
            if len(inspect) != 1:
                raise ValueError(f"Docker inspect count differs: {path}")
            inspect = inspect[0]
        validate_docker_inspect(inspect, allocation["gpu_index"])


def checkpoint_inventory(repo: Path, k: int, subset: int) -> list[dict]:
    adapters = repo / "artifacts/withheld_pairs_20260923/adapters"
    inventory = []
    for seed in (85000, 85001, 85002):
        for arm in ("random", "withheld"):
            adapter = adapters / f"k{k}_s{subset}_seed{seed}_{arm}"
            weights = adapter / "adapter_model.safetensors"
            receipt = adapter / "training_receipt.json"
            config = adapter / "adapter_config.json"
            if not all(path.is_file() and not path.is_symlink() for path in (weights, receipt, config)):
                raise ValueError(f"checkpoint missing: {adapter}")
            inventory.append({
                "seed": seed, "arm": arm,
                "adapter_path": adapter.relative_to(repo).as_posix(),
                "weights_sha256": digest(weights), "weights_bytes": weights.stat().st_size,
                "training_receipt_sha256": digest(receipt),
                "adapter_config_sha256": digest(config),
                "weights_in_audit_zip": False,
            })
    return inventory


def write_archive(repo: Path, target: Path, paths: list[Path], inventory: list[dict]) -> None:
    if not paths or any(not path.is_file() or path.is_symlink() for path in paths):
        raise ValueError("empty or symlinked audit inputs")
    manifest = [{"path": path.relative_to(repo).as_posix(), "sha256": digest(path),
                 "bytes": path.stat().st_size} for path in paths]
    if len({item["path"] for item in manifest}) != len(manifest):
        raise ValueError("duplicate audit member")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix="set-audit-", suffix=".zip",
                                         dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6) as archive:
            for path in paths:
                name = path.relative_to(repo).as_posix()
                info = zipfile.ZipInfo(name, date_time=(2026, 9, 23, 0, 0, 0))
                archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED,
                                 compresslevel=6)
            for name, payload in (("AUDIT_MANIFEST.json", manifest),
                                  ("CHECKPOINT_INVENTORY.json", inventory)):
                info = zipfile.ZipInfo(name, date_time=(2026, 9, 23, 0, 0, 0))
                archive.writestr(info, json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n")
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP CRC check failed")
            for item in manifest:
                payload = archive.read(item["path"])
                if len(payload) != item["bytes"] or hashlib.sha256(payload).hexdigest().upper() != item["sha256"]:
                    raise ValueError(f"ZIP content hash mismatch: {item['path']}")
        if target.exists():
            if digest(target) != digest(temporary):
                raise ValueError("existing audit ZIP differs; use a new versioned name")
        else:
            os.link(temporary, target)
        sidecar = target.with_suffix(".zip.sha256")
        line = f"{digest(target)}  {target.name}\n"
        if sidecar.exists():
            if sidecar.read_text(encoding="utf-8") != line:
                raise ValueError("existing audit ZIP sidecar differs")
        else:
            with sidecar.open("x", encoding="utf-8") as handle:
                handle.write(line)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_bundle(repo: Path, k: int, subset: int, target: Path | None = None) -> Path:
    require_evidence(repo, k, subset)
    inventory = checkpoint_inventory(repo, k, subset)
    if target is None:
        target = (repo / "reports/withheld_pairs_20260923/data" /
                  f"k{k}_s{subset}/k{k}_s{subset}_audit.zip")
    write_archive(repo, target, candidates(repo, k, subset), inventory)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--subset", type=int, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    target = build_bundle(args.repo, args.k, args.subset, args.out)
    print(json.dumps({"status": "VALID", "archive": str(target), "sha256": digest(target),
                      "bytes": target.stat().st_size}))


if __name__ == "__main__":
    main()
