"""Build a compact, hash-indexed archive without model weight files."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArchiveItem:
    source: Path
    archive_path: str


EXCLUDED_SUFFIXES = {".zip", ".pyc", ".safetensors", ".pt", ".pth", ".bin"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _eligible(path: Path) -> bool:
    if (not path.is_file() or path.suffix.lower() in EXCLUDED_SUFFIXES or
            path.name == "trajectory_diversity_audit.receipt.json"):
        return False
    return not any(part in {"__pycache__", ".pytest_cache"} for part in path.parts)


def selected_files(repo_root: Path, report_dir: Path, artifact_dir: Path) -> list[ArchiveItem]:
    items: dict[str, ArchiveItem] = {}
    for base in (report_dir, artifact_dir):
        for path in sorted(base.rglob("*")):
            if _eligible(path):
                relative = path.relative_to(repo_root).as_posix()
                items[relative] = ArchiveItem(path, relative)
    frozen_sources = (
        "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/code/composition_core.py",
        "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/code/composition_eval.py",
        "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/code/composition_model.py",
    )
    for relative in frozen_sources:
        path = repo_root / relative
        if _eligible(path):
            items[relative] = ArchiveItem(path, relative)
    return [items[key] for key in sorted(items)]


def build_archive(repo_root: Path, report_dir: Path, artifact_dir: Path, output: Path) -> dict:
    items = selected_files(repo_root, report_dir, artifact_dir)
    rows = [
        {"path": item.archive_path, "bytes": item.source.stat().st_size, "sha256": sha256(item.source)}
        for item in items
    ]
    manifest = {
        "schema": "trajectory-diversity.audit-archive.v1",
        "status": "DONE",
        "model_weights_included": False,
        "weight_integrity_source": "reports/trajectory_diversity_20260922/data/new_experiment/integrity_audit.json",
        "files": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for item in items:
            archive.write(item.source, item.archive_path)
        archive.writestr("ARCHIVE_MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(output)
    return {
        "schema": "trajectory-diversity.audit-archive-receipt.v1",
        "status": "DONE",
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "included_files": len(rows),
        "model_weights_included": False,
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=here / "trajectory_diversity_audit.zip")
    args = parser.parse_args()
    receipt = build_archive(repo, here, repo / "artifacts/trajectory_diversity_20260922", args.output.resolve())
    receipt_path = args.output.resolve().with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
