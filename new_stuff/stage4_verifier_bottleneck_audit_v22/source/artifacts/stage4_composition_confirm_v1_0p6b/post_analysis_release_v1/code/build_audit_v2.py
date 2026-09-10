from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


REPO = Path(r"C:\MyProject\verifier_bottleneck_codex_pack")
ROOT = REPO / "artifacts/stage4_composition_confirm_v1_0p6b"
POST = ROOT / "post_analysis_release_v1"
ARCHIVES = POST / "archives"
BASE_ARCHIVE = ARCHIVES / "stage4_verifier_bottleneck_audit_v1.zip"
OUTPUT_ARCHIVE = ARCHIVES / "stage4_verifier_bottleneck_audit_v2.zip"

SOURCE_DIRECTORIES = (
    ROOT / "code",
    ROOT / "tests",
    ROOT / "schemas",
    ROOT / "configs",
    POST / "code",
    REPO / "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/code",
    REPO / "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/tests",
    REPO / "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/schemas",
    REPO / "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/configs",
    REPO / "artifacts/stage4_sh1_v5_0p6b/code",
    REPO / "artifacts/stage4_sh1_v5_0p6b/tests",
    REPO / "artifacts/stage4_sh1_v5_0p6b/configs",
    REPO / "artifacts/stage4_distill_v1/code",
    REPO / "artifacts/stage4_distill_v1/configs",
    REPO / "artifacts/stage4_distill_v2/code",
    REPO / "artifacts/stage4_distill_v2/configs",
    REPO / "artifacts/stage4_distill_v3/code",
    REPO / "artifacts/stage4_distill_v3/configs",
)

SOURCE_FILES = (
    REPO / "pyproject.toml",
    REPO / "README.md",
    ROOT / "PREREGISTRATION.md",
    ROOT / "RECOVERY_AMENDMENT_001.md",
    ROOT / "RECOVERY_AMENDMENT_001.json",
    REPO / "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/PREREGISTRATION.md",
    REPO / "artifacts/stage4_sh1_v5_0p6b/PREREGISTRATION.md",
    REPO / "artifacts/stage4_sh1_v5_0p6b/manifests/environment.json",
    REPO / "artifacts/stage4_distill_v1/manifests/environment.json",
    REPO / "artifacts/stage4_distill_v2/manifests/environment.json",
    REPO / "artifacts/stage4_distill_v3/manifests/environment.json",
)

ALLOWED_SUFFIXES = {".py", ".json", ".md", ".toml", ".yaml", ".yml", ".txt"}
REQUIRED_RUNTIME_FILES = {
    "artifacts/stage4_composition_confirm_v1_0p6b/code/confirm.py",
    "artifacts/stage4_composition_confirm_v1_0p6b/code/confirm_stats.py",
    "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/code/composition_core.py",
    "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/code/composition_eval.py",
    "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/code/composition_model.py",
    "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/code/stage4.py",
    "artifacts/stage4_sh1_v5_0p6b/code/v5_train.py",
    "artifacts/stage4_distill_v1/code/stage4_core.py",
    "artifacts/stage4_distill_v2/code/v2_atomic.py",
    "artifacts/stage4_distill_v3/code/v3_train.py",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def source_paths() -> list[Path]:
    paths: set[Path] = set()
    for directory in SOURCE_DIRECTORIES:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in directory.rglob("*"):
            if (path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES
                    and "__pycache__" not in path.parts):
                paths.add(path.resolve())
    for path in SOURCE_FILES:
        if path.is_file():
            paths.add(path.resolve())
    relative = {path.relative_to(REPO.resolve()).as_posix() for path in paths}
    missing = sorted(REQUIRED_RUNTIME_FILES - relative)
    if missing:
        raise RuntimeError(f"required runtime sources missing: {missing}")
    return sorted(paths, key=lambda item: item.relative_to(REPO.resolve()).as_posix())


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def deterministic_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(2026, 8, 2, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def main() -> None:
    if not BASE_ARCHIVE.is_file():
        raise FileNotFoundError(BASE_ARCHIVE)
    sources = source_paths()
    entries = []
    for path in sources:
        relative = path.relative_to(REPO.resolve()).as_posix()
        entries.append({
            "path": relative,
            "archive_path": f"source/{relative}",
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })

    manifest = {
        "schema": "stage4.audit.source-code-manifest.v2",
        "status": "PASS",
        "base_archive": BASE_ARCHIVE.name,
        "base_archive_sha256": sha256_file(BASE_ARCHIVE),
        "source_file_count": len(entries),
        "source_total_bytes": sum(item["bytes"] for item in entries),
        "required_runtime_files": sorted(REQUIRED_RUNTIME_FILES),
        "files": entries,
        "excluded_heavy_artifacts": [
            "base model weights",
            "LoRA adapters and checkpoints",
            "raw full-program ranking shards",
            "generated training and evaluation datasets",
        ],
    }
    readme = """# Stage 4 compact audit archive v2

This archive contains everything from audit_v1 plus the complete source-code
path used for Stage 4 atomic training, composition distillation, exact ranking,
statistics, release construction, schemas, tests, protocols and preregistration.

Source files retain repository-relative paths below `source/`. See
`audit_v2/SOURCE_CODE_MANIFEST.json` for SHA-256 hashes and the exact inventory.

To keep the archive compact, model weights, adapters/checkpoints, generated
datasets and raw full-ranking shards are intentionally excluded. Those binary
and generated artifacts remain in `stage4_verifier_bottleneck_full_v1.zip`.
The source code is complete; a byte-identical rerun still requires the frozen
model, adapter and data artifacts from the full archive.
""".encode("utf-8")

    temporary = OUTPUT_ARCHIVE.with_suffix(OUTPUT_ARCHIVE.suffix + ".partial")
    existing_names: set[str] = set()
    with zipfile.ZipFile(BASE_ARCHIVE, "r") as source_archive, zipfile.ZipFile(
            temporary, "w", allowZip64=True) as output:
        for member in source_archive.infolist():
            if member.filename in existing_names:
                raise RuntimeError(f"duplicate base member: {member.filename}")
            existing_names.add(member.filename)
            output.writestr(member, source_archive.read(member.filename))
        generated = {
            "audit_v2/README.md": readme,
            "audit_v2/SOURCE_CODE_MANIFEST.json": json_bytes(manifest),
        }
        for name, payload in generated.items():
            if name in existing_names:
                raise RuntimeError(f"duplicate generated member: {name}")
            output.writestr(deterministic_info(name), payload, compresslevel=6)
            existing_names.add(name)
        for path, entry in zip(sources, entries, strict=True):
            name = entry["archive_path"]
            if name in existing_names:
                raise RuntimeError(f"duplicate source member: {name}")
            output.write(path, name, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
            existing_names.add(name)
    temporary.replace(OUTPUT_ARCHIVE)

    with zipfile.ZipFile(OUTPUT_ARCHIVE, "r") as archive:
        bad_member = archive.testzip()
        names = set(archive.namelist())
    if bad_member is not None:
        raise RuntimeError(f"CRC failure: {bad_member}")
    missing_members = [item["archive_path"] for item in entries if item["archive_path"] not in names]
    if missing_members:
        raise RuntimeError(f"missing source members: {missing_members[:10]}")

    digest = sha256_file(OUTPUT_ARCHIVE)
    sidecar = OUTPUT_ARCHIVE.with_suffix(OUTPUT_ARCHIVE.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {OUTPUT_ARCHIVE.name}\n", encoding="ascii")
    receipt = {
        "schema": "stage4.audit.archive-receipt.v2",
        "status": "PASS",
        "path": OUTPUT_ARCHIVE.relative_to(ROOT).as_posix(),
        "bytes": OUTPUT_ARCHIVE.stat().st_size,
        "sha256": digest,
        "members": len(names),
        "source_files": len(entries),
        "source_bytes": manifest["source_total_bytes"],
        "crc": "PASS",
    }
    receipt_path = POST / "manifests/AUDIT_V2_RECEIPT.json"
    receipt_path.write_bytes(json_bytes(receipt))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
