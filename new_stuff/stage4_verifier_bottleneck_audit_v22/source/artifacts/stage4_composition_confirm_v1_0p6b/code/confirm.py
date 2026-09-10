from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent.parent
LEGACY = REPO / "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8"
ATOMIC = REPO / "artifacts/stage4_sh1_v5_0p6b"
LEGACY_CODE = LEGACY / "code"
sys.path.insert(0, str(LEGACY_CODE))
sys.path.insert(0, str(ROOT / "code"))

import composition_core as core
import composition_eval as eval_lib
import composition_model as model_lib
import confirm_atomic_v11
from confirm_stats import analyze_primary, holm, seed_t_statistics


CONFIG_PATH = ROOT / "configs/protocol.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
DATA = ROOT / "data"
RUNS = ROOT / "runs"
ADAPTERS = ROOT / "adapters"
RANKINGS = ROOT / "rankings"
MANIFESTS = ROOT / "manifests"
STATISTICS = ROOT / "statistics"
REPORTS = ROOT / "reports"
ARCHIVES = ROOT / "archives"
EXPORT = REPO / CONFIG["source"]["atomic_export"]
PILOT_DATA = LEGACY / "data"
AMENDMENT_SPEC_PATH = ROOT / "RECOVERY_AMENDMENT_001.json"
AMENDMENT_MD_PATH = ROOT / "RECOVERY_AMENDMENT_001.md"
AMENDMENT_RECEIPT_PATH = MANIFESTS / "recovery_amendment_001_receipt.json"
BLINDNESS_SNAPSHOT_PATH = MANIFESTS / "recovery_blindness_snapshot_001.json"
FAILURE_LEDGER = RUNS / "failures" / "freeze_attempt0"
RECOVERY_FREEZE_STARTED_PATH = RUNS / "recovery" / "freeze_attempt1_STARTED.json"
RECOVERY_FREEZE_DONE_PATH = RUNS / "recovery" / "freeze_attempt1_DONE.json"

# Reuse the audited implementation while redirecting every mutable runtime path
# to this isolated follow-up root. The legacy files themselves remain untouched.
model_lib.ROOT = ROOT
model_lib.PROTOCOL = {"model": CONFIG["model"], "pilot_initial": CONFIG["training"]}
_RUN_LEASE = None


def ensure_roots() -> None:
    for path in (DATA, RUNS, ADAPTERS, RANKINGS, MANIFESTS, STATISTICS, REPORTS, ARCHIVES):
        path.mkdir(parents=True, exist_ok=True)


def acquire_run_lease():
    global _RUN_LEASE
    if _RUN_LEASE is not None:
        return _RUN_LEASE
    path = RUNS / "confirmation.run.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    if handle.seek(0, os.SEEK_END) == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError("another confirmation runner holds the run lease") from exc
    _RUN_LEASE = handle
    return handle


def release_run_lease() -> None:
    global _RUN_LEASE
    handle = _RUN_LEASE
    _RUN_LEASE = None
    if handle is None:
        return
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def activity(phase: str, status: str = "RUNNING", **extra) -> None:
    dump_json(RUNS / "ACTIVE_PROCESS.json",
              {"schema": "stage4.composition.confirmation.activity.v1", "status": status,
               "phase": phase, "pid": os.getpid(), "updated_at_unix": time.time(), **extra})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def hash_value(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                         allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest().upper()


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(path: Path, value, *, exclusive: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return
    temporary = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.partial")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in Path(path).rglob("*") if item.is_file())
    if not files:
        raise RuntimeError(f"empty tree: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest().upper()


def manifest_entry(path: Path, *, rows: int | None = None) -> dict:
    result = {"path": path.resolve().relative_to(REPO.resolve()).as_posix(),
              "bytes": path.stat().st_size, "sha256": sha256(path)}
    if rows is not None:
        result["rows"] = rows
    return result


def runtime_dependency_paths() -> list[Path]:
    roots = [LEGACY / "code", ATOMIC / "code",
             REPO / "artifacts/stage4_distill_v1/code",
             REPO / "artifacts/stage4_distill_v2/code",
             REPO / "artifacts/stage4_distill_v3/code"]
    paths = []
    for root in roots:
        paths.extend(sorted(root.glob("*.py")))
    paths.append(ATOMIC / "configs/protocol.json")
    return sorted({path.resolve() for path in paths})


def runtime_dependency_manifest() -> dict:
    return {path.relative_to(REPO.resolve()).as_posix(): {"bytes": path.stat().st_size,
                                                          "sha256": sha256(path)}
            for path in runtime_dependency_paths()}


def code_manifest() -> dict:
    paths = sorted((ROOT / "code").glob("*.py")) + sorted((ROOT / "tests").glob("*.py"))
    paths += sorted((ROOT / "schemas").glob("*.json"))
    paths += [CONFIG_PATH, ROOT / "PREREGISTRATION.md", AMENDMENT_SPEC_PATH, AMENDMENT_MD_PATH]
    result = {path.relative_to(ROOT).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
              for path in paths}
    result.update({f"external:{name}": entry for name, entry in runtime_dependency_manifest().items()})
    return result


def _recovery_spec() -> dict:
    value = load_json(AMENDMENT_SPEC_PATH)
    if (value.get("schema") != "stage4.composition.confirmation.recovery-amendment-spec.v1" or
            value.get("status") != "PREREGISTERED" or
            value.get("amendment_id") != "RECOVERY_AMENDMENT_001"):
        raise RuntimeError("recovery amendment spec is invalid")
    if (value.get("algorithm", {}).get("name") != confirm_atomic_v11.ALGORITHM or
            int(value.get("algorithm", {}).get("seed", -1)) != int(CONFIG["final_data"]["confirm_atomic_seed"]) or
            int(value.get("algorithm", {}).get("per_operation", -1)) != int(CONFIG["final_data"]["confirm_atomic_per_operation"])):
        raise RuntimeError("recovery amendment algorithm differs from frozen protocol")
    scope = value.get("scope", {})
    if (int(scope.get("maximum_freeze_retries_under_amendment", -1)) != 1 or
            scope.get("same_attempt_deterministic_resume_on_interruption") is not True or
            scope.get("resume_forbidden_after_final_access_or_metrics") is not True):
        raise RuntimeError("recovery amendment retry scope is invalid")
    return value


def _assert_recovery_original_bindings(spec: dict) -> None:
    bindings = spec["original_bindings"]
    paths = {
        "protocol_sha256": CONFIG_PATH,
        "preregistration_sha256": MANIFESTS / "preregistration.json",
        "selection_freeze_sha256": ROOT / "CONFIG_SELECTION_FROZEN.json",
        "budget_manifest_sha256": MANIFESTS / "budget_manifest.json",
        "training_done_sha256": RUNS / "TRAINING_DONE.json",
        "failed_sha256": RUNS / "FAILED.json",
        "failed_stdout_sha256": FAILURE_LEDGER / "confirmation.stdout.log",
        "failed_stderr_sha256": FAILURE_LEDGER / "confirmation.stderr.log",
        "legacy_composition_core_sha256": LEGACY / "code/composition_core.py",
    }
    for key, path in paths.items():
        _assert_hash(path, bindings[key], f"recovery original binding {key}")
    _assert_hash(FAILURE_LEDGER / "FAILED.json", bindings["failed_sha256"], "failure ledger receipt")
    preregistration = load_json(MANIFESTS / "preregistration.json")
    if (preregistration.get("git_commit") != bindings["git_commit"] or
            preregistration.get("code", {}).get("code/confirm.py", {}).get("sha256") != bindings["original_confirm_py_sha256"]):
        raise RuntimeError("original preregistration code/commit binding changed")
    training = load_json(RUNS / "TRAINING_DONE.json")
    for label, expected in bindings["pair_receipts"].items():
        entry = training.get("replicates", {}).get(str(label), {})
        path = ROOT / str(entry.get("path", ""))
        if entry.get("sha256") != expected or not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"recovery pair receipt {label} changed")


def _blindness_inventory() -> dict:
    prohibited = [
        ROOT / "CONFIG_FROZEN.json",
        MANIFESTS / "final_data_manifest.json",
        MANIFESTS / "final_split_audit.json",
        MANIFESTS / "final_generation_receipt.json",
        MANIFESTS / "final_evaluation_access.json",
        MANIFESTS / "confirm_atomic_generation_audit.json",
        RUNS / "PRIMARY_EVALUATION_DONE.json",
        RUNS / "SECONDARY_EVALUATION_DONE.json",
        STATISTICS / "PRIMARY_ANALYSIS.json",
        STATISTICS / "FINAL_ANALYSIS.json",
        RECOVERY_FREEZE_STARTED_PATH,
        RECOVERY_FREEZE_DONE_PATH,
    ]
    final_data_files = sorted(
        path.relative_to(ROOT).as_posix()
        for path in DATA.glob("*.jsonl")
        if path.name.startswith("final_") or path.name == "confirm_atomic.jsonl"
    )
    ranking_files = sorted(path.relative_to(ROOT).as_posix() for path in RANKINGS.rglob("*") if path.is_file())
    metric_files = sorted(path.relative_to(ROOT).as_posix() for path in STATISTICS.rglob("*") if path.is_file())
    report_files = sorted(path.relative_to(ROOT).as_posix() for path in REPORTS.rglob("*") if path.is_file())
    archive_files = sorted(path.relative_to(ROOT).as_posix() for path in ARCHIVES.rglob("*") if path.is_file())
    evaluation_receipts = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (RUNS / "confirm").glob("replicate*/eval_*.json")
        if path.is_file()
    )
    present = sorted(path.relative_to(ROOT).as_posix() for path in prohibited if path.exists())
    value = {
        "schema": "stage4.composition.confirmation.recovery-blindness-snapshot.v1",
        "status": "PASS" if not (present or final_data_files or ranking_files or metric_files or
                                  report_files or archive_files or evaluation_receipts) else "FAIL",
        "prohibited_present": present,
        "final_data_files": final_data_files,
        "ranking_files": ranking_files,
        "metric_files": metric_files,
        "report_files": report_files,
        "archive_files": archive_files,
        "evaluation_receipts": evaluation_receipts,
        "config_frozen_absent": not (ROOT / "CONFIG_FROZEN.json").exists(),
        "final_evaluation_access_absent": not (MANIFESTS / "final_evaluation_access.json").exists(),
        "composition_rows_transient_only": True,
        "model_outputs_consulted": False,
    }
    return value


def _expected_final_data_filenames() -> set[str]:
    families = sorted(str(name).lower() for name in CONFIG["final_data"]["depth3_sizes"])
    names = {f"final_{family}.jsonl" for family in families}
    names.update(f"final_{family}_depth{depth}.jsonl" for family in families for depth in (2, 4))
    names.add("confirm_atomic.jsonl")
    return names


def _recovery_resume_inventory() -> dict:
    """Permit only deterministic partial freeze outputs from the same attempt."""

    prohibited = [
        ROOT / "CONFIG_FROZEN.json",
        MANIFESTS / "final_generation_receipt.json",
        MANIFESTS / "final_evaluation_access.json",
        RUNS / "PRIMARY_EVALUATION_DONE.json",
        RUNS / "SECONDARY_EVALUATION_DONE.json",
        RUNS / "RAW_VALIDATION_DONE.json",
        RUNS / "STAGE4_COMPOSITION_DONE.json",
        RECOVERY_FREEZE_DONE_PATH,
    ]
    present = sorted(path.relative_to(ROOT).as_posix() for path in prohibited if path.exists())
    ranking_files = sorted(path.relative_to(ROOT).as_posix() for path in RANKINGS.rglob("*") if path.is_file())
    metric_files = sorted(path.relative_to(ROOT).as_posix() for path in STATISTICS.rglob("*") if path.is_file())
    report_files = sorted(path.relative_to(ROOT).as_posix() for path in REPORTS.rglob("*") if path.is_file())
    archive_files = sorted(path.relative_to(ROOT).as_posix() for path in ARCHIVES.rglob("*") if path.is_file())
    evaluation_receipts = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (RUNS / "confirm").glob("replicate*/eval_*.json")
        if path.is_file()
    )
    final_data_files = sorted(
        path.relative_to(ROOT).as_posix()
        for path in DATA.glob("*.jsonl")
        if path.name.startswith("final_") or path.name == "confirm_atomic.jsonl"
    )
    expected = _expected_final_data_filenames()
    unexpected_final_data = sorted(
        path.relative_to(ROOT).as_posix()
        for path in DATA.glob("*.jsonl")
        if (path.name.startswith("final_") or path.name == "confirm_atomic.jsonl") and path.name not in expected
    )
    allowed_manifests = {
        "confirm_atomic_generation_audit.json",
        "final_data_manifest.json",
        "final_split_audit.json",
    }
    unexpected_final_manifests = sorted(
        path.relative_to(ROOT).as_posix()
        for path in MANIFESTS.glob("*.json")
        if (path.name.startswith("final_") or path.name.startswith("confirm_atomic_"))
        and path.name not in allowed_manifests
    )
    failures = (present + ranking_files + metric_files + report_files + archive_files + evaluation_receipts
                + unexpected_final_data + unexpected_final_manifests)
    return {
        "schema": "stage4.composition.confirmation.recovery-freeze-resume-inventory.v1",
        "status": "PASS" if not failures else "FAIL",
        "prohibited_present": present,
        "ranking_files": ranking_files,
        "metric_files": metric_files,
        "report_files": report_files,
        "archive_files": archive_files,
        "evaluation_receipts": evaluation_receipts,
        "partial_final_data_files": final_data_files,
        "unexpected_final_data_files": unexpected_final_data,
        "unexpected_final_manifests": unexpected_final_manifests,
    }


def _training_bindings() -> dict:
    training = require_training()
    pairs: dict[str, object] = {}
    for label, entry in sorted(training["replicates"].items(), key=lambda item: int(item[0])):
        pair_path = ROOT / entry["path"]
        pair = load_json(pair_path)
        adapters = {}
        for branch, adapter_entry in sorted(pair["adapters"].items()):
            adapter = ROOT / adapter_entry["path"]
            adapters[branch] = {
                "path": adapter_entry["path"],
                "tree_sha256": tree_sha256(adapter),
                "training_receipt_sha256": sha256(adapter / "training_receipt.json"),
            }
        pairs[label] = {"pair_receipt_sha256": sha256(pair_path), "adapters": adapters}
    return {"training_done_sha256": sha256(RUNS / "TRAINING_DONE.json"), "pairs": pairs}


def _assert_amendment_code_scope(original_code: dict, current_code: dict) -> None:
    removed = sorted(set(original_code) - set(current_code))
    changed = sorted(
        key for key in set(original_code) & set(current_code)
        if original_code[key] != current_code[key]
    )
    added = sorted(set(current_code) - set(original_code))
    allowed_changed = {"code/confirm.py", "schemas/freeze.schema.json"}
    allowed_added = {
        "RECOVERY_AMENDMENT_001.json",
        "RECOVERY_AMENDMENT_001.md",
        "code/confirm_atomic_v11.py",
        "schemas/recovery_amendment.schema.json",
        "tests/test_recovery_amendment.py",
    }
    if removed or set(changed) != allowed_changed or set(added) != allowed_added:
        raise RuntimeError(
            f"recovery code scope differs: removed={removed}, changed={changed}, added={added}"
        )


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def _assert_amendment_commit(receipt: dict) -> None:
    """Bind the receipt to the committed tree that introduced this amendment."""

    commit = str(receipt.get("git_commit", ""))
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise RuntimeError("recovery amendment commit is invalid")
    if subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=REPO,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode != 0:
        raise RuntimeError("recovery amendment commit is missing")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=REPO,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode != 0:
        raise RuntimeError("recovery amendment commit is not an ancestor of HEAD")
    git_paths = []
    for name in receipt["code"]:
        if name.startswith("external:"):
            relative = name.removeprefix("external:")
        else:
            relative = (ROOT / name).resolve().relative_to(REPO.resolve()).as_posix()
        if subprocess.run(
            ["git", "cat-file", "-e", f"{commit}:{relative}"], cwd=REPO,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode != 0:
            raise RuntimeError(f"recovery amendment commit does not contain {relative}")
        git_paths.append(relative)
    if subprocess.run(
        ["git", "diff", "--quiet", commit, "--", *git_paths], cwd=REPO,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode != 0:
        raise RuntimeError("working recovery code differs from amendment commit")


def amend_recovery() -> dict:
    acquire_run_lease()
    try:
        activity("amend_recovery")
        spec = _recovery_spec()
        _assert_recovery_original_bindings(spec)
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
        if status.strip():
            raise RuntimeError(f"recovery amendment requires a clean Git tree:\n{status}")
        preregistration = load_json(MANIFESTS / "preregistration.json")
        current_code = code_manifest()
        _assert_amendment_code_scope(preregistration["code"], current_code)
        blindness = _blindness_inventory()
        if blindness["status"] != "PASS":
            raise RuntimeError(f"recovery blindness guard failed: {blindness}")
        blindness_payload = {**blindness, "checked_at_unix": time.time(),
                             "failure_receipt_sha256": sha256(RUNS / "FAILED.json")}
        if BLINDNESS_SNAPSHOT_PATH.exists():
            existing = load_json(BLINDNESS_SNAPSHOT_PATH)
            stable_existing = {key: value for key, value in existing.items() if key != "checked_at_unix"}
            stable_new = {key: value for key, value in blindness_payload.items() if key != "checked_at_unix"}
            if stable_existing != stable_new:
                raise RuntimeError("recovery blindness snapshot differs")
        else:
            dump_json(BLINDNESS_SNAPSHOT_PATH, blindness_payload, exclusive=True)
        commit = _git_head()
        payload = {
            "schema": "stage4.composition.confirmation.recovery-amendment-receipt.v1",
            "status": "FROZEN",
            "amendment_id": "RECOVERY_AMENDMENT_001",
            "git_commit": commit,
            "created_at_unix": time.time(),
            "spec_sha256": sha256(AMENDMENT_SPEC_PATH),
            "original_bindings": spec["original_bindings"],
            "blindness_snapshot": manifest_entry(BLINDNESS_SNAPSHOT_PATH),
            "training_bindings": _training_bindings(),
            "code": current_code,
            "scope": spec["scope"],
            "algorithm": spec["algorithm"],
        }
        validate_schema(payload, "recovery_amendment.schema.json")
        if AMENDMENT_RECEIPT_PATH.exists():
            existing = load_json(AMENDMENT_RECEIPT_PATH)
            stable_existing = {key: value for key, value in existing.items() if key != "created_at_unix"}
            stable_new = {key: value for key, value in payload.items() if key != "created_at_unix"}
            if stable_existing != stable_new:
                raise RuntimeError("frozen recovery amendment receipt differs")
            activity("amendment_frozen", "DONE", amendment_sha256=sha256(AMENDMENT_RECEIPT_PATH))
            return existing
        dump_json(AMENDMENT_RECEIPT_PATH, payload, exclusive=True)
        activity("amendment_frozen", "DONE", amendment_sha256=sha256(AMENDMENT_RECEIPT_PATH))
        return payload
    finally:
        release_run_lease()


def _assert_recovery_receipt_spec_fields(receipt: dict, spec: dict) -> None:
    for key in ("original_bindings", "scope", "algorithm"):
        if receipt.get(key) != spec.get(key):
            raise RuntimeError(f"recovery amendment receipt differs from spec: {key}")


def require_recovery_amendment() -> dict:
    spec = _recovery_spec()
    _assert_recovery_original_bindings(spec)
    if not AMENDMENT_RECEIPT_PATH.is_file() or not BLINDNESS_SNAPSHOT_PATH.is_file():
        raise RuntimeError("recovery amendment receipt is absent")
    receipt = load_json(AMENDMENT_RECEIPT_PATH)
    validate_schema(receipt, "recovery_amendment.schema.json")
    _assert_recovery_receipt_spec_fields(receipt, spec)
    if (receipt.get("spec_sha256") != sha256(AMENDMENT_SPEC_PATH) or
            receipt.get("code") != code_manifest() or
            receipt.get("training_bindings") != _training_bindings() or
            receipt.get("blindness_snapshot") != manifest_entry(BLINDNESS_SNAPSHOT_PATH)):
        raise RuntimeError("recovery amendment binding changed")
    _assert_amendment_commit(receipt)
    preregistration = load_json(MANIFESTS / "preregistration.json")
    _assert_amendment_code_scope(preregistration["code"], receipt["code"])
    snapshot = load_json(BLINDNESS_SNAPSHOT_PATH)
    if snapshot.get("status") != "PASS" or snapshot.get("failure_receipt_sha256") != spec["original_bindings"]["failed_sha256"]:
        raise RuntimeError("recovery blindness snapshot is invalid")
    return receipt


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", f"Get-Process -Id {pid} -ErrorAction Stop | Out-Null"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return result.returncode == 0
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def _validate_recovery_freeze_started(value: dict, amendment: dict) -> None:
    expected = {
        "schema": "stage4.composition.confirmation.recovery-freeze-attempt.v1",
        "status": "STARTED",
        "attempt_number": 1,
        "maximum_freeze_retries_under_amendment": 1,
        "recovery_amendment_sha256": sha256(AMENDMENT_RECEIPT_PATH),
        "recovery_amendment_git_commit": amendment["git_commit"],
        "recovery_spec_sha256": sha256(AMENDMENT_SPEC_PATH),
        "blindness_snapshot_sha256": sha256(BLINDNESS_SNAPSHOT_PATH),
        "selection_freeze_sha256": sha256(ROOT / "CONFIG_SELECTION_FROZEN.json"),
        "training_done_sha256": sha256(RUNS / "TRAINING_DONE.json"),
        "code_manifest_sha256": hash_value(code_manifest()),
        "model_outputs_consulted_before_start": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise RuntimeError(f"recovery freeze STARTED binding changed: {key}")
    if not isinstance(value.get("owner_pid"), int) or value["owner_pid"] <= 0:
        raise RuntimeError("recovery freeze STARTED owner PID is invalid")
    if not isinstance(value.get("created_at_unix"), (int, float)):
        raise RuntimeError("recovery freeze STARTED timestamp is invalid")


def require_recovery_freeze_started(amendment: dict | None = None) -> dict:
    amendment = amendment or require_recovery_amendment()
    if not RECOVERY_FREEZE_STARTED_PATH.is_file():
        raise RuntimeError("recovery freeze STARTED receipt is absent")
    value = load_json(RECOVERY_FREEZE_STARTED_PATH)
    _validate_recovery_freeze_started(value, amendment)
    return value


def begin_recovery_freeze_attempt() -> tuple[dict, dict, bool]:
    """Start once, or resume the same deterministic pre-access freeze attempt."""

    amendment = require_recovery_amendment()
    if RECOVERY_FREEZE_STARTED_PATH.exists():
        started = require_recovery_freeze_started(amendment)
        if RECOVERY_FREEZE_DONE_PATH.exists():
            raise RuntimeError("recovery freeze DONE exists without CONFIG_FROZEN")
        inventory = _recovery_resume_inventory()
        if inventory["status"] != "PASS":
            raise RuntimeError(f"recovery freeze resume guard failed: {inventory}")
        owner_pid = int(started["owner_pid"])
        if owner_pid != os.getpid() and _pid_is_alive(owner_pid):
            raise RuntimeError(f"recovery freeze attempt is active under PID {owner_pid}")
        return amendment, started, True

    current = _blindness_inventory()
    if current.get("status") != "PASS":
        raise RuntimeError(f"final data/access appeared before recovery freeze: {current}")
    if _git_head() != amendment["git_commit"]:
        raise RuntimeError("recovery freeze must start at the committed amendment HEAD")
    maximum = int(amendment["scope"]["maximum_freeze_retries_under_amendment"])
    if maximum != 1:
        raise RuntimeError("recovery amendment does not authorize exactly one freeze attempt")
    started = {
        "schema": "stage4.composition.confirmation.recovery-freeze-attempt.v1",
        "status": "STARTED",
        "attempt_number": 1,
        "maximum_freeze_retries_under_amendment": maximum,
        "owner_pid": os.getpid(),
        "created_at_unix": time.time(),
        "recovery_amendment_sha256": sha256(AMENDMENT_RECEIPT_PATH),
        "recovery_amendment_git_commit": amendment["git_commit"],
        "recovery_spec_sha256": sha256(AMENDMENT_SPEC_PATH),
        "blindness_snapshot_sha256": sha256(BLINDNESS_SNAPSHOT_PATH),
        "selection_freeze_sha256": sha256(ROOT / "CONFIG_SELECTION_FROZEN.json"),
        "training_done_sha256": sha256(RUNS / "TRAINING_DONE.json"),
        "code_manifest_sha256": hash_value(code_manifest()),
        "model_outputs_consulted_before_start": False,
    }
    dump_json(RECOVERY_FREEZE_STARTED_PATH, started, exclusive=True)
    _validate_recovery_freeze_started(started, amendment)
    return amendment, started, False


def _final_generation_stable_sha256(generation_path: Path) -> str:
    generation = load_json(generation_path)
    if generation.get("evaluation_count") not in (0, 1):
        raise RuntimeError("final generation evaluation count is invalid")
    return hash_value({key: value for key, value in generation.items() if key != "evaluation_count"})


def complete_recovery_freeze_attempt(frozen_path: Path, generation_path: Path) -> dict:
    amendment = require_recovery_amendment()
    started = require_recovery_freeze_started(amendment)
    payload = {
        "schema": "stage4.composition.confirmation.recovery-freeze-attempt.v1",
        "status": "DONE",
        "attempt_number": 1,
        "completed_at_unix": time.time(),
        "started_receipt_sha256": sha256(RECOVERY_FREEZE_STARTED_PATH),
        "config_frozen_sha256": sha256(frozen_path),
        "final_generation_stable_sha256": _final_generation_stable_sha256(generation_path),
        "recovery_amendment_sha256": sha256(AMENDMENT_RECEIPT_PATH),
        "recovery_amendment_git_commit": amendment["git_commit"],
        "started_owner_pid": started["owner_pid"],
    }
    if RECOVERY_FREEZE_DONE_PATH.exists():
        existing = load_json(RECOVERY_FREEZE_DONE_PATH)
        stable_existing = {key: value for key, value in existing.items() if key != "completed_at_unix"}
        stable_payload = {key: value for key, value in payload.items() if key != "completed_at_unix"}
        if stable_existing != stable_payload:
            raise RuntimeError("recovery freeze DONE receipt differs")
        return existing
    dump_json(RECOVERY_FREEZE_DONE_PATH, payload, exclusive=True)
    return payload


def require_recovery_freeze_done(frozen_path: Path, generation_path: Path) -> dict:
    if not RECOVERY_FREEZE_DONE_PATH.is_file():
        raise RuntimeError("recovery freeze DONE receipt is absent")
    value = load_json(RECOVERY_FREEZE_DONE_PATH)
    amendment = require_recovery_amendment()
    started = require_recovery_freeze_started(amendment)
    expected = {
        "schema": "stage4.composition.confirmation.recovery-freeze-attempt.v1",
        "status": "DONE",
        "attempt_number": 1,
        "started_receipt_sha256": sha256(RECOVERY_FREEZE_STARTED_PATH),
        "config_frozen_sha256": sha256(frozen_path),
        "final_generation_stable_sha256": _final_generation_stable_sha256(generation_path),
        "recovery_amendment_sha256": sha256(AMENDMENT_RECEIPT_PATH),
        "recovery_amendment_git_commit": amendment["git_commit"],
        "started_owner_pid": started["owner_pid"],
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise RuntimeError(f"recovery freeze DONE binding changed: {key}")
    if not isinstance(value.get("completed_at_unix"), (int, float)):
        raise RuntimeError("recovery freeze DONE timestamp is invalid")
    return value


def _validate_final_access_receipt(access: dict, frozen: dict) -> None:
    expected = {
        "schema": "stage4.composition.confirmation.final-access.v1",
        "status": "STARTED",
        "config_frozen_sha256": sha256(ROOT / "CONFIG_FROZEN.json"),
        "test_files": frozen["test_files"],
    }
    for key, expected_value in expected.items():
        if access.get(key) != expected_value:
            raise RuntimeError(f"prior final access binding is invalid: {key}")
    if not isinstance(access.get("opened_at_unix"), (int, float)):
        raise RuntimeError("prior final access timestamp is invalid")
    if not isinstance(access.get("owner_pid"), int) or access["owner_pid"] <= 0:
        raise RuntimeError("prior final access owner PID is invalid")
    if not isinstance(access.get("resume_count"), int) or access["resume_count"] < 0:
        raise RuntimeError("prior final access resume count is invalid")


def _reconcile_final_access_state(frozen: dict, *, create_if_absent: bool) -> tuple[dict | None, bool, bool]:
    """Finish the two-file access transaction after an interruption."""

    receipt_path = MANIFESTS / "final_generation_receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError("final generation receipt is absent")
    generation = load_json(receipt_path)
    if (generation.get("config_frozen_sha256") != sha256(ROOT / "CONFIG_FROZEN.json") or
            generation.get("test_files") != frozen["test_files"]):
        raise RuntimeError("final generation binding is invalid")
    count = generation.get("evaluation_count")
    access_path = MANIFESTS / "final_evaluation_access.json"
    if access_path.exists():
        access = load_json(access_path)
        _validate_final_access_receipt(access, frozen)
        if count == 0:
            generation["evaluation_count"] = 1
            dump_json(receipt_path, generation)
            return access, False, True
        if count != 1:
            raise RuntimeError("prior final access evaluation count is invalid")
        return access, False, False
    if count != 0:
        raise RuntimeError("final evaluator was opened without an access receipt")
    if not create_if_absent:
        return None, False, False
    access = {
        "schema": "stage4.composition.confirmation.final-access.v1",
        "status": "STARTED",
        "config_frozen_sha256": sha256(ROOT / "CONFIG_FROZEN.json"),
        "test_files": frozen["test_files"],
        "opened_at_unix": time.time(),
        "owner_pid": os.getpid(),
        "resume_count": 0,
    }
    dump_json(access_path, access, exclusive=True)
    generation["evaluation_count"] = 1
    dump_json(receipt_path, generation)
    return access, True, True


def validate_schema(value: dict, filename: str) -> None:
    from jsonschema import Draft202012Validator
    schema = load_json(ROOT / "schemas" / filename)
    Draft202012Validator(schema).validate(value)


def _assert_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256(path) != expected.upper():
        raise RuntimeError(f"{label} hash mismatch: {path}")


def source_audit() -> dict:
    if CONFIG["model"] != "Qwen/Qwen3-0.6B" or CONFIG["model_size_lock"] != "0.6B":
        raise RuntimeError("Qwen3-0.6B lock failed")
    if "1.7B" not in CONFIG["forbidden_model_sizes"]:
        raise RuntimeError("1.7B prohibition is missing")
    historical = CONFIG["historical_pilot"]
    decision_path = LEGACY / "runs/PILOT_DECISION.json"
    attempt_path = LEGACY / "runs/pilot/seed0_lr1em04_e2_r0p20/attempt.json"
    terminal_path = LEGACY / "runs/STAGE4_FAILED.json"
    pilot_manifest_path = LEGACY / "manifests/pilot_data_manifest.json"
    export_receipt_path = EXPORT / "export_receipt.json"
    checks = {
        "decision_hash": sha256(decision_path) == historical["decision_sha256"],
        "attempt_hash": sha256(attempt_path) == historical["attempt_sha256"],
        "terminal_hash": sha256(terminal_path) == historical["terminal_sha256"],
        "pilot_manifest_hash": sha256(pilot_manifest_path) == historical["pilot_data_manifest_sha256"],
        "export_receipt_hash": sha256(export_receipt_path) == CONFIG["source"]["atomic_export_receipt_sha256"],
        "discover_hash": sha256(PILOT_DATA / "discover_trajectories.jsonl") == CONFIG["source"]["discover_trajectories_sha256"],
    }
    decision = load_json(decision_path)
    attempt = load_json(attempt_path)
    receipt = load_json(export_receipt_path)
    checks.update({
        "historical_result_unchanged": decision.get("status") == historical["historical_result_must_remain"],
        "initial_was_selected": decision.get("initial_attempt") == decision.get("selected_attempt") == historical["selected_attempt"],
        "no_dev_cycle": decision.get("dev_cycle", {}).get("used") is False,
        "pilot_final_absent": decision.get("final_created") is False and not (LEGACY / "CONFIG_FROZEN.json").exists(),
        "pilot_delta_bound": abs(float(attempt["delta_hit32"]) - float(historical["dev_a_delta"])) < 1e-15,
        "pilot_budget_equal": all(attempt["budget_checks"].values()),
        "pilot_leakage_zero": attempt.get("leakage_count") == 0,
        "atomic_export_payload": receipt.get("export_tree_sha256", "").upper() == CONFIG["source"]["atomic_export_tree_sha256"],
        "atomic_export_payload_rederived": model_lib._export_payload_sha(EXPORT).upper() == CONFIG["source"]["atomic_export_tree_sha256"],
        "tokenizer_class_qwen2": load_json(EXPORT / "tokenizer_config.json").get("tokenizer_class") == "Qwen2Tokenizer",
        "tokenizer_policy_frozen": CONFIG.get("tokenizer_regex_policy") == "preserve_frozen_export_default_used_by_pilot",
    })
    if not all(checks.values()):
        raise RuntimeError(f"source audit failed: {checks}")
    result = {"schema": "stage4.composition.confirmation.source-audit.v1", "status": "PASS",
              "model": CONFIG["model"], "checks": checks,
              "historical_pilot": {"decision": manifest_entry(decision_path),
                                   "attempt": manifest_entry(attempt_path),
                                   "terminal": manifest_entry(terminal_path)},
              "atomic_export_payload_sha256": receipt["export_tree_sha256"].upper(),
              "runtime_dependencies": runtime_dependency_manifest(),
              "tokenizer_regex_policy": CONFIG["tokenizer_regex_policy"],
              "pilot_observations_disclosed": {"dev_a_delta": historical["dev_a_delta"],
                                                "dev_b_delta": historical["dev_b_delta"]}}
    return result


def preregister() -> dict:
    ensure_roots()
    validate_schema(CONFIG, "protocol.schema.json")
    path = MANIFESTS / "preregistration.json"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    if not path.exists() and status.strip():
        raise RuntimeError(f"preregistration requires a clean Git tree:\n{status}")
    import peft
    import scipy
    import torch
    import transformers
    payload = {"schema": "stage4.composition.confirmation.preregistration.v1", "status": "FROZEN",
               "study_class": CONFIG["study_class"], "model": CONFIG["model"],
               "git_commit": commit, "created_at_unix": time.time(),
               "protocol_sha256": sha256(CONFIG_PATH),
               "preregistration_md_sha256": sha256(ROOT / "PREREGISTRATION.md"),
               "code": code_manifest(), "source_audit": source_audit(),
               "software": {"python": platform.python_version(), "torch": torch.__version__,
                            "transformers": transformers.__version__, "peft": peft.__version__,
                            "numpy": np.__version__, "scipy": scipy.__version__, "cuda": torch.version.cuda}}
    if path.exists():
        existing = load_json(path)
        stable = {key: value for key, value in payload.items() if key != "created_at_unix"}
        prior = {key: value for key, value in existing.items() if key != "created_at_unix"}
        if prior != stable:
            raise RuntimeError("frozen preregistration differs from current code/config/source")
        return existing
    dump_json(path, payload, exclusive=True)
    return payload


def require_preregistered() -> dict:
    path = MANIFESTS / "preregistration.json"
    if not path.is_file():
        raise RuntimeError("preregistration is absent")
    value = load_json(path)
    if value.get("status") != "FROZEN" or value.get("protocol_sha256") != sha256(CONFIG_PATH):
        raise RuntimeError("preregistration is stale")
    if value.get("source_audit") != source_audit():
        raise RuntimeError("source changed after preregistration")
    if value.get("code") != code_manifest():
        require_recovery_amendment()
    return value


def resolved_config(rep: dict) -> dict:
    value = deepcopy(CONFIG["training"])
    value.update({"schema": "stage4.composition.confirmation.resolved-training.v1",
                  "replicate_label": int(rep["replicate_label"]), "seed": int(rep["rng_seed"])})
    return value


def balanced_atomic_pool(rng_seed: int, per_operation: int = 1000) -> list[dict]:
    import random
    rng = random.Random(rng_seed)
    sh1 = read_jsonl(ATOMIC / "data/full_sh1.jsonl")
    controls = read_jsonl(ATOMIC / "data/control_apply.jsonl")
    by_operation = {"SH1": sh1}
    for operation in core.OPS[1:]:
        by_operation[operation] = [row for row in controls if row["operation"] == operation]
    rows = []
    for operation in core.OPS:
        rows.extend(rng.sample(by_operation[operation], per_operation))
    rng.shuffle(rows)
    return rows


def build_encoded_pair(rep: dict):
    resolved = resolved_config(rep)
    trajectories = read_jsonl(PILOT_DATA / "discover_trajectories.jsonl")
    atomic_pool = balanced_atomic_pool(resolved["seed"] + 1000)
    tokenizer, _ = model_lib.tokenizer_and_ids(EXPORT)
    distill = model_lib.encode_specs(
        tokenizer,
        model_lib.distill_example_specs(trajectories, atomic_pool, resolved["atomic_replay"], resolved["seed"]),
    )
    control_pool = model_lib.encode_specs(tokenizer, model_lib.atomic_example_specs(atomic_pool))
    control, distill, budget = model_lib.pack_control_to_budget(control_pool, distill, resolved["seed"])
    accumulation = resolved["effective_batch"] // resolved["micro_batch"]
    expected_steps = math.ceil(math.ceil(len(distill) / resolved["micro_batch"]) / accumulation) * resolved["epochs"]
    budget.update({"schema": "stage4.composition.confirmation.equal-budget-pair.v1",
                   "replicate_label": resolved["replicate_label"], "rng_seed": resolved["seed"],
                   "control_input_sha256": hash_value(control), "distill_input_sha256": hash_value(distill),
                   "resolved_config_sha256": hash_value(resolved),
                   "runtime_dependencies_sha256": hash_value(runtime_dependency_manifest()),
                   "effective_batch": resolved["effective_batch"], "epochs": resolved["epochs"],
                   "optimizer_steps_expected": expected_steps,
                   "loss_bearing_target_tokens_expected_each": budget["loss_bearing_target_tokens_each"] * resolved["epochs"]})
    return resolved, control, distill, budget


def prepare_budgets() -> dict:
    require_preregistered()
    if (ROOT / "CONFIG_FROZEN.json").exists() or list(DATA.glob("final_*.jsonl")):
        raise RuntimeError("training budget preparation found pre-existing final data")
    entries = {}
    for rep in CONFIG["replicates"]:
        label = int(rep["replicate_label"])
        _, _, _, budget = build_encoded_pair(rep)
        path = RUNS / "confirm" / f"replicate{label}" / "budget.json"
        if path.exists() and load_json(path) != budget:
            raise RuntimeError(f"replicate {label} frozen budget differs")
        if not path.exists():
            dump_json(path, budget, exclusive=True)
        entries[str(label)] = {**manifest_entry(path), "rng_seed": int(rep["rng_seed"]),
                               "optimizer_steps_expected": budget["optimizer_steps_expected"],
                               "loss_bearing_target_tokens_expected_each": budget["loss_bearing_target_tokens_expected_each"]}
    manifest = {"schema": "stage4.composition.confirmation.budget-manifest.v1", "status": "FROZEN",
                "replicates": entries}
    manifest_path = MANIFESTS / "budget_manifest.json"
    if manifest_path.exists() and load_json(manifest_path) != manifest:
        raise RuntimeError("budget manifest changed")
    if not manifest_path.exists():
        dump_json(manifest_path, manifest, exclusive=True)
    selection = {"schema": "stage4.composition.confirmation.selection-freeze.v1", "status": "FROZEN",
                 "model": CONFIG["model"], "study_class": CONFIG["study_class"],
                 "pilot_adapter_reuse": False, "replicates": CONFIG["replicates"],
                 "training": CONFIG["training"], "protocol_sha256": sha256(CONFIG_PATH),
                 "preregistration_sha256": sha256(MANIFESTS / "preregistration.json"),
                 "budget_manifest_sha256": sha256(manifest_path), "code": code_manifest(),
                 "final_data_spec": CONFIG["final_data"], "primary": CONFIG["primary"]}
    selection_path = ROOT / "CONFIG_SELECTION_FROZEN.json"
    if selection_path.exists() and load_json(selection_path) != selection:
        raise RuntimeError("selection freeze changed")
    if not selection_path.exists():
        dump_json(selection_path, selection, exclusive=True)
    return manifest


def require_selection() -> dict:
    require_preregistered()
    path = ROOT / "CONFIG_SELECTION_FROZEN.json"
    if not path.is_file():
        raise RuntimeError("selection freeze is absent")
    value = load_json(path)
    if value.get("protocol_sha256") != sha256(CONFIG_PATH):
        raise RuntimeError("selection freeze is stale")
    if value.get("code") != code_manifest():
        amendment = require_recovery_amendment()
        if amendment["original_bindings"]["selection_freeze_sha256"] != sha256(path):
            raise RuntimeError("selection freeze is not bound by recovery amendment")
    if value.get("budget_manifest_sha256") != sha256(MANIFESTS / "budget_manifest.json"):
        raise RuntimeError("budget manifest changed after selection freeze")
    return value


def train_all() -> dict:
    require_selection()
    results = {}
    for rep in CONFIG["replicates"]:
        label = int(rep["replicate_label"])
        resolved, control, distill, budget = build_encoded_pair(rep)
        budget_path = RUNS / "confirm" / f"replicate{label}" / "budget.json"
        if load_json(budget_path) != budget:
            raise RuntimeError(f"replicate {label} regenerated budget mismatch")
        pair_root = ADAPTERS / "confirm" / f"replicate{label}_seed{resolved['seed']}"
        control_path = pair_root / "atomic_control"
        distill_path = pair_root / "composition_distill"
        control_receipt = model_lib.train_branch(EXPORT, control_path, control, resolved, "atomic_control",
                                                 budget["control_input_sha256"])
        distill_receipt = model_lib.train_branch(EXPORT, distill_path, distill, resolved, "composition_distill",
                                                 budget["distill_input_sha256"])
        checks = {
            "receipts_done": control_receipt.get("status") == distill_receipt.get("status") == "DONE",
            "optimizer_steps_equal": control_receipt["optimizer_steps"] == distill_receipt["optimizer_steps"],
            "optimizer_steps_expected": control_receipt["optimizer_steps"] == distill_receipt["optimizer_steps"] == budget["optimizer_steps_expected"],
            "loss_tokens_equal": control_receipt["loss_bearing_target_tokens_seen"] == distill_receipt["loss_bearing_target_tokens_seen"],
            "loss_tokens_expected": control_receipt["loss_bearing_target_tokens_seen"] == distill_receipt["loss_bearing_target_tokens_seen"] == budget["loss_bearing_target_tokens_expected_each"],
            "effective_batch_equal": control_receipt["effective_batch"] == distill_receipt["effective_batch"] == resolved["effective_batch"],
            "epochs_equal": control_receipt["epochs"] == distill_receipt["epochs"] == resolved["epochs"],
            "fresh_confirmation_stream": resolved["seed"] != 0 and CONFIG["seed0_pilot_adapter_reuse"] is False,
        }
        if not all(checks.values()):
            raise RuntimeError(f"replicate {label} equal-budget validation failed: {checks}")
        pair = {"schema": "stage4.composition.confirmation.training-pair.v1", "status": "DONE",
                "replicate_label": label, "rng_seed": resolved["seed"], "resolved_config": resolved,
                "budget": budget, "checks": checks,
                "adapters": {"atomic_control": {"path": control_path.relative_to(ROOT).as_posix(),
                                                  "tree_sha256": tree_sha256(control_path),
                                                  "receipt_sha256": sha256(control_path / "training_receipt.json")},
                             "composition_distill": {"path": distill_path.relative_to(ROOT).as_posix(),
                                                     "tree_sha256": tree_sha256(distill_path),
                                                     "receipt_sha256": sha256(distill_path / "training_receipt.json")}}}
        pair_receipt = RUNS / "confirm" / f"replicate{label}" / "training_pair.json"
        if pair_receipt.exists() and load_json(pair_receipt) != pair:
            raise RuntimeError(f"replicate {label} training receipt changed")
        if not pair_receipt.exists():
            dump_json(pair_receipt, pair, exclusive=True)
        results[str(label)] = {"path": pair_receipt.relative_to(ROOT).as_posix(),
                               "sha256": sha256(pair_receipt), "checks": checks}
        print(json.dumps({"milestone": "training_pair_done", "replicate": label,
                          "control_minutes": control_receipt["wall_seconds"] / 60,
                          "distill_minutes": distill_receipt["wall_seconds"] / 60}), flush=True)
        del control, distill
        gc.collect()
    receipt = {"schema": "stage4.composition.confirmation.training-series.v1", "status": "DONE",
               "replicates": results, "all_six_complete": len(results) == 6 and all(all(v["checks"].values()) for v in results.values())}
    dump_json(RUNS / "TRAINING_DONE.json", receipt)
    return receipt


def require_training() -> dict:
    path = RUNS / "TRAINING_DONE.json"
    if not path.is_file():
        raise RuntimeError("six-replicate training is incomplete")
    value = load_json(path)
    if value.get("status") != "DONE" or value.get("all_six_complete") is not True:
        raise RuntimeError("training terminal receipt is invalid")
    for label_text, entry in value["replicates"].items():
        receipt = ROOT / entry["path"]
        if not receipt.is_file() or sha256(receipt) != entry["sha256"]:
            raise RuntimeError("training pair receipt changed")
        pair = load_json(receipt)
        label = int(label_text)
        if pair.get("status") != "DONE" or pair.get("replicate_label") != label or not all(pair.get("checks", {}).values()):
            raise RuntimeError(f"replicate {label} frozen training checks failed")
        budget_path = RUNS / "confirm" / f"replicate{label}" / "budget.json"
        if not budget_path.is_file() or load_json(budget_path) != pair.get("budget"):
            raise RuntimeError(f"replicate {label} budget changed after training")
        manifest_entry_value = load_json(MANIFESTS / "budget_manifest.json")["replicates"][str(label)]
        if sha256(budget_path) != manifest_entry_value["sha256"]:
            raise RuntimeError(f"replicate {label} budget manifest binding failed")
        for branch, adapter_entry in pair["adapters"].items():
            adapter = ROOT / adapter_entry["path"]
            training_receipt = adapter / "training_receipt.json"
            if (not adapter.is_dir() or tree_sha256(adapter) != adapter_entry["tree_sha256"] or
                    not training_receipt.is_file() or sha256(training_receipt) != adapter_entry["receipt_sha256"] or
                    load_json(training_receipt).get("status") != "DONE"):
                raise RuntimeError(f"replicate {label} {branch} adapter/receipt binding failed")
    if model_lib._export_payload_sha(EXPORT).upper() != CONFIG["source"]["atomic_export_tree_sha256"]:
        raise RuntimeError("atomic export changed after training")
    return value


def validated_adapter(label: int, branch: str) -> tuple[Path, str, str]:
    series = load_json(RUNS / "TRAINING_DONE.json")
    entry = series["replicates"][str(label)]
    pair_path = ROOT / entry["path"]
    if sha256(pair_path) != entry["sha256"]:
        raise RuntimeError(f"replicate {label} pair receipt changed")
    pair = load_json(pair_path)
    adapter_entry = pair["adapters"][branch]
    adapter = ROOT / adapter_entry["path"]
    receipt = adapter / "training_receipt.json"
    if (tree_sha256(adapter) != adapter_entry["tree_sha256"] or
            sha256(receipt) != adapter_entry["receipt_sha256"] or
            load_json(receipt).get("status") != "DONE"):
        raise RuntimeError(f"replicate {label} evaluated adapter differs from frozen training pair")
    return adapter, adapter_entry["tree_sha256"], sha256(pair_path)


def atomic_reference() -> core.AtomicReference:
    data_manifest = load_json(ATOMIC / "manifests/data_manifest.json")
    corrective_manifest = load_json(ATOMIC / "manifests/corrective_data_manifest.json")
    manifest = {**data_manifest, **corrective_manifest}
    names = ("coordinate_source", "full_sh1", "control_apply", "calibration",
             "corrective_sh1", "corrective_control")
    raw = {name: read_jsonl(ATOMIC / "data" / f"{name}.jsonl") for name in names}
    return core.audit_atomic_reference(manifest, raw, required_splits=("calibration",), strict=True)


def pilot_splits() -> dict[str, list[dict]]:
    return {name: read_jsonl(PILOT_DATA / f"{name}.jsonl") for name in ("train", "dev_a", "dev_b")}


def freeze_final() -> dict:
    selection = require_selection()
    training = require_training()
    frozen_path = ROOT / "CONFIG_FROZEN.json"
    if frozen_path.exists():
        frozen = load_json(frozen_path)
        amendment = require_recovery_amendment()
        require_recovery_freeze_started(amendment)
        for filename, entry in frozen["test_files"].items():
            path = DATA / filename
            if not path.is_file() or sha256(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
                raise RuntimeError(f"frozen test changed: {filename}")
        generation_path = MANIFESTS / "final_generation_receipt.json"
        expected_generation = {
            "schema": "stage4.composition.confirmation.final-generation.v1",
            "status": "DONE",
            "config_frozen_sha256": sha256(frozen_path),
            "test_files": frozen["test_files"],
            "evaluation_count": 0,
            "generated_after_all_training": True,
            "recovery_amendment_sha256": sha256(AMENDMENT_RECEIPT_PATH),
            "recovery_freeze_started_sha256": sha256(RECOVERY_FREEZE_STARTED_PATH),
            "confirm_atomic_generation_audit_sha256": sha256(MANIFESTS / "confirm_atomic_generation_audit.json"),
        }
        if generation_path.exists():
            existing_generation = load_json(generation_path)
            if existing_generation.get("evaluation_count") not in (0, 1):
                raise RuntimeError("recovered final generation evaluation count is invalid")
            stable_existing = {key: value for key, value in existing_generation.items() if key != "evaluation_count"}
            stable_expected = {key: value for key, value in expected_generation.items() if key != "evaluation_count"}
            if stable_existing != stable_expected:
                raise RuntimeError("recovered final generation receipt differs")
        if not generation_path.exists():
            if (MANIFESTS / "final_evaluation_access.json").exists():
                raise RuntimeError("cannot recover generation receipt after final access")
            dump_json(generation_path, expected_generation, exclusive=True)
        _reconcile_final_access_state(frozen, create_if_absent=False)
        complete_recovery_freeze_attempt(frozen_path, generation_path)
        return require_frozen()
    recovery, _recovery_started, _resumed = begin_recovery_freeze_attempt()
    if (MANIFESTS / "final_evaluation_access.json").exists():
        raise RuntimeError("final access marker exists before final freeze")
    selection_sha = sha256(ROOT / "CONFIG_SELECTION_FROZEN.json")
    reference = atomic_reference()
    pilots = pilot_splits()
    spec = CONFIG["final_data"]
    finals, audit = core.generate_final_data(
        reference, pilots, config_frozen=True,
        depth3_sizes=spec["depth3_sizes"],
        depth2_probe_per_split=spec["depth2_probe_per_family"],
        depth4_probe_per_split=spec["depth4_probe_per_family"],
        seed_base=spec["generation_seed_base"],
    )
    confirm_atomic, confirm_atomic_audit = confirm_atomic_v11.generate_confirm_atomic_data_v11(
        reference, pilots, finals,
        per_operation=spec["confirm_atomic_per_operation"], seed=spec["confirm_atomic_seed"],
    )
    finals["confirm_atomic"] = confirm_atomic
    unified = core.audit_pilot_data({**pilots, **finals}, atomic_reference=reference, verify_shortest=True)
    if not audit.get("ok") or not unified.get("ok"):
        raise RuntimeError(f"final split integrity failed: {audit.get('errors')} {unified.get('errors')}")
    confirm_atomic_audit["unified_split_audit_ok"] = unified.get("ok") is True
    confirm_atomic_audit["recovery_amendment_sha256"] = sha256(AMENDMENT_RECEIPT_PATH)
    confirm_atomic_audit["recovery_freeze_started_sha256"] = sha256(RECOVERY_FREEZE_STARTED_PATH)
    confirm_atomic_audit["recovery_freeze_attempt_number"] = 1
    confirm_atomic_audit_path = MANIFESTS / "confirm_atomic_generation_audit.json"
    if confirm_atomic_audit_path.exists() and load_json(confirm_atomic_audit_path) != confirm_atomic_audit:
        raise RuntimeError("recovered confirm-atomic generation audit differs")
    if not confirm_atomic_audit_path.exists():
        dump_json(confirm_atomic_audit_path, confirm_atomic_audit, exclusive=True)
    entries = {}
    for split, rows in sorted(finals.items()):
        path = DATA / f"{split}.jsonl"
        if path.exists() and read_jsonl(path) != rows:
            raise RuntimeError(f"partial final mismatch: {path}")
        if not path.exists():
            write_jsonl(path, rows)
        entries[path.name] = manifest_entry(path, rows=len(rows))
    final_manifest = {"schema": "stage4.composition.confirmation.final-data-manifest.v1", "status": "FROZEN",
                      "selection_freeze_sha256": selection_sha, "files": entries,
                      "recovery_amendment_sha256": sha256(AMENDMENT_RECEIPT_PATH),
                      "recovery_freeze_started_sha256": sha256(RECOVERY_FREEZE_STARTED_PATH),
                      "confirm_atomic_generation_audit_sha256": sha256(confirm_atomic_audit_path)}
    final_manifest_path = MANIFESTS / "final_data_manifest.json"
    audit_path = MANIFESTS / "final_split_audit.json"
    if final_manifest_path.exists() and load_json(final_manifest_path) != final_manifest:
        raise RuntimeError("recovered final data manifest differs")
    if not final_manifest_path.exists():
        dump_json(final_manifest_path, final_manifest, exclusive=True)
    if audit_path.exists() and load_json(audit_path) != unified:
        raise RuntimeError("recovered final split audit differs")
    if not audit_path.exists():
        dump_json(audit_path, unified, exclusive=True)
    frozen = {"schema": "stage4.composition.confirmation.config-freeze.v1", "status": "FROZEN",
              "study_class": CONFIG["study_class"], "model": CONFIG["model"],
              "selection_freeze_sha256": selection_sha,
              "training_done_sha256": sha256(RUNS / "TRAINING_DONE.json"),
              "protocol_sha256": sha256(CONFIG_PATH), "code": code_manifest(),
               "replicates": CONFIG["replicates"], "test_files": entries,
               "final_data_manifest_sha256": sha256(final_manifest_path),
               "final_split_audit_sha256": sha256(MANIFESTS / "final_split_audit.json"),
               "original_preregistration_sha256": recovery["original_bindings"]["preregistration_sha256"],
               "recovery_amendment_sha256": sha256(AMENDMENT_RECEIPT_PATH),
               "recovery_blindness_snapshot_sha256": sha256(BLINDNESS_SNAPSHOT_PATH),
               "recovery_freeze_started_sha256": sha256(RECOVERY_FREEZE_STARTED_PATH),
               "freeze_attempt0_failed_sha256": recovery["original_bindings"]["failed_sha256"],
               "confirm_atomic_generation_audit_sha256": sha256(confirm_atomic_audit_path),
               "primary": CONFIG["primary"], "final_evaluator_opened": False}
    validate_schema(frozen, "freeze.schema.json")
    dump_json(frozen_path, frozen, exclusive=True)
    generation = {"schema": "stage4.composition.confirmation.final-generation.v1", "status": "DONE",
                   "config_frozen_sha256": sha256(frozen_path), "test_files": entries,
                   "evaluation_count": 0, "generated_after_all_training": training["all_six_complete"] is True,
                   "recovery_amendment_sha256": sha256(AMENDMENT_RECEIPT_PATH),
                   "recovery_freeze_started_sha256": sha256(RECOVERY_FREEZE_STARTED_PATH),
                   "confirm_atomic_generation_audit_sha256": sha256(confirm_atomic_audit_path)}
    generation_path = MANIFESTS / "final_generation_receipt.json"
    dump_json(generation_path, generation, exclusive=True)
    complete_recovery_freeze_attempt(frozen_path, generation_path)
    print(json.dumps({"milestone": "final_frozen", "config_sha256": sha256(frozen_path),
                      "files": {name: entry["sha256"] for name, entry in entries.items()}}), flush=True)
    return require_frozen()


def require_frozen() -> dict:
    frozen_path = ROOT / "CONFIG_FROZEN.json"
    if not frozen_path.is_file():
        raise RuntimeError("CONFIG_FROZEN.json is absent")
    frozen = load_json(frozen_path)
    validate_schema(frozen, "freeze.schema.json")
    if frozen.get("status") != "FROZEN" or frozen.get("protocol_sha256") != sha256(CONFIG_PATH):
        raise RuntimeError("frozen configuration is invalid")
    if frozen.get("code") != code_manifest():
        raise RuntimeError("code changed after final freeze")
    if (frozen.get("selection_freeze_sha256") != sha256(ROOT / "CONFIG_SELECTION_FROZEN.json") or
            frozen.get("training_done_sha256") != sha256(RUNS / "TRAINING_DONE.json") or
            frozen.get("final_data_manifest_sha256") != sha256(MANIFESTS / "final_data_manifest.json") or
            frozen.get("final_split_audit_sha256") != sha256(MANIFESTS / "final_split_audit.json") or
            frozen.get("recovery_amendment_sha256") != sha256(AMENDMENT_RECEIPT_PATH) or
            frozen.get("recovery_blindness_snapshot_sha256") != sha256(BLINDNESS_SNAPSHOT_PATH) or
            frozen.get("recovery_freeze_started_sha256") != sha256(RECOVERY_FREEZE_STARTED_PATH) or
            frozen.get("confirm_atomic_generation_audit_sha256") != sha256(MANIFESTS / "confirm_atomic_generation_audit.json")):
        raise RuntimeError("frozen selection/training/final audit chain changed")
    recovery = require_recovery_amendment()
    if (frozen.get("original_preregistration_sha256") != recovery["original_bindings"]["preregistration_sha256"] or
            frozen.get("freeze_attempt0_failed_sha256") != recovery["original_bindings"]["failed_sha256"]):
        raise RuntimeError("frozen recovery provenance changed")
    require_selection()
    if model_lib._export_payload_sha(EXPORT).upper() != CONFIG["source"]["atomic_export_tree_sha256"]:
        raise RuntimeError("frozen atomic export changed")
    for filename, entry in frozen["test_files"].items():
        path = DATA / filename
        if not path.is_file() or sha256(path) != entry["sha256"] or path.stat().st_size != entry["bytes"] or len(read_jsonl(path)) != entry["rows"]:
            raise RuntimeError(f"frozen test validation failed: {filename}")
    require_recovery_freeze_done(frozen_path, MANIFESTS / "final_generation_receipt.json")
    return frozen


def begin_final_access() -> dict:
    frozen = require_frozen()
    access, created, _reconciled = _reconcile_final_access_state(frozen, create_if_absent=True)
    if access is None:
        raise RuntimeError("final access transaction did not produce a receipt")
    if created:
        return access
    access["owner_pid"] = os.getpid()
    access["resume_count"] = int(access["resume_count"]) + 1
    dump_json(MANIFESTS / "final_evaluation_access.json", access)
    return access


def _adapter(label: int, branch: str) -> Path:
    series = load_json(RUNS / "TRAINING_DONE.json")
    pair = load_json(ROOT / series["replicates"][str(label)]["path"])
    return ROOT / pair["adapters"][branch]["path"]


def _eval_name(label: int | str, branch: str, split: str) -> str:
    return f"replicate{label}_{branch}_{split}"


def _binding(frozen: dict, label: int | str, branch: str, split: str, model_sha: str,
             rng_seed: int | None, training_pair_sha256: str | None = None) -> dict:
    value = {"schema": "stage4.composition.confirmation.eval-binding.v1",
             "replicate_label": label, "branch": branch, "split": split,
             "model_sha256": model_sha, "data_sha256": frozen["test_files"][f"{split}.jsonl"]["sha256"],
             "atomic_export_sha256": CONFIG["source"]["atomic_export_tree_sha256"],
             "runtime_dependencies_sha256": hash_value(runtime_dependency_manifest()),
             "config_frozen_sha256": sha256(ROOT / "CONFIG_FROZEN.json"),
             "protocol_sha256": sha256(CONFIG_PATH),
             "core_sha256": sha256(LEGACY_CODE / "composition_core.py"),
             "evaluator_sha256": sha256(LEGACY_CODE / "composition_eval.py")}
    if rng_seed is not None:
        value["rng_seed"] = rng_seed
    if training_pair_sha256 is not None:
        value["training_pair_sha256"] = training_pair_sha256
    return value


def evaluate_branch(label: int | str, branch: str, adapter: Path | None,
                    splits: list[str], *, evaluate_atomic: bool = False) -> dict:
    frozen = require_frozen()
    begin_final_access()
    if adapter is None:
        model_sha = model_lib._export_payload_sha(EXPORT).upper()
        rng_seed = None
        pair_sha = None
    else:
        validated_path, model_sha, pair_sha = validated_adapter(int(label), branch)
        if validated_path.resolve() != adapter.resolve():
            raise RuntimeError("requested adapter path differs from frozen training pair")
        rep = next(item for item in CONFIG["replicates"] if int(item["replicate_label"]) == int(label))
        rng_seed = int(rep["rng_seed"])
    out_root = RANKINGS / f"replicate{label}"
    model, tokenizer, token_ids = model_lib.load_branch(EXPORT, adapter)
    summaries = {}
    try:
        for split in splits:
            rows = read_jsonl(DATA / f"{split}.jsonl")
            name = _eval_name(label, branch, split)
            binding = _binding(frozen, label, branch, split, model_sha, rng_seed, pair_sha)
            summary, _ = eval_lib.evaluate_ranking(model, tokenizer, token_ids, rows, out_root, name,
                                                   binding, batch_size=CONFIG["ranking"]["batch_size"])
            summaries[split] = summary
        atomic = None
        if evaluate_atomic:
            rows = read_jsonl(DATA / "confirm_atomic.jsonl")
            name = f"replicate{label}_{branch}"
            binding = _binding(frozen, label, branch, "confirm_atomic", model_sha, rng_seed, pair_sha)
            atomic = eval_lib.evaluate_atomic(model, tokenizer, token_ids, rows, out_root, name, binding)
    finally:
        eval_lib.unload(model)
    result = {"schema": "stage4.composition.confirmation.branch-eval.v1", "status": "DONE",
              "replicate_label": label, "branch": branch, "model_sha256": model_sha,
              "summaries": summaries, "atomic": atomic}
    path = RUNS / "confirm" / f"replicate{label}" / f"eval_{branch}_{hash_value(splits)[:10]}.json"
    dump_json(path, result)
    return result


def evaluate_primary() -> dict:
    require_frozen()
    require_training()
    begin_final_access()
    results = {}
    for rep in CONFIG["replicates"]:
        label = int(rep["replicate_label"])
        results[str(label)] = {}
        for branch in ("atomic_control", "composition_distill"):
            result = evaluate_branch(label, branch, _adapter(label, branch), ["final_a"])
            results[str(label)][branch] = result["summaries"]["final_a"]
        delta = (results[str(label)]["composition_distill"]["hit@32"] -
                 results[str(label)]["atomic_control"]["hit@32"])
        print(json.dumps({"milestone": "primary_replicate_done", "replicate": label,
                          "control_hit32": results[str(label)]["atomic_control"]["hit@32"],
                          "distill_hit32": results[str(label)]["composition_distill"]["hit@32"],
                          "delta": delta}), flush=True)
    receipt = {"schema": "stage4.composition.confirmation.primary-evaluation.v1", "status": "DONE",
               "replicates": results}
    dump_json(RUNS / "PRIMARY_EVALUATION_DONE.json", receipt)
    return receipt


def _load_metric_map(label: int, branch: str, splits: list[str], metric: str) -> dict[str, float]:
    values = {}
    for split in splits:
        name = _eval_name(label, branch, split)
        directory = RANKINGS / f"replicate{label}" / "metrics" / name
        files = sorted(directory.glob("part-*.jsonl"))
        if not files:
            raise RuntimeError(f"metrics missing: {directory}")
        for path in files:
            for row in read_jsonl(path):
                task_id = f"{split}:{row['task_id']}" if len(splits) > 1 else row["task_id"]
                if task_id in values:
                    raise RuntimeError(f"duplicate metric task: {task_id}")
                values[task_id] = float(row[metric])
    return values


def _metric_matrix(splits: list[str], metric: str) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    control = {}
    distill = {}
    for rep in CONFIG["replicates"]:
        label = int(rep["replicate_label"])
        control[label] = _load_metric_map(label, "atomic_control", splits, metric)
        distill[label] = _load_metric_map(label, "composition_distill", splits, metric)
    return control, distill


def analyze_primary_result() -> dict:
    require_frozen()
    if not (RUNS / "PRIMARY_EVALUATION_DONE.json").is_file():
        raise RuntimeError("primary evaluation is incomplete")
    control, distill = _metric_matrix(["final_a"], "hit@32")
    primary, bootstrap = analyze_primary(control, distill,
                                         CONFIG["statistics"]["bootstrap_repetitions"],
                                         CONFIG["statistics"]["bootstrap_seed"])
    STATISTICS.mkdir(parents=True, exist_ok=True)
    bootstrap_path = STATISTICS / "primary_bootstrap.npy"
    np.save(bootstrap_path, bootstrap, allow_pickle=False)
    primary["seed_task_bootstrap"].update({"replicates_path": bootstrap_path.relative_to(ROOT).as_posix(),
                                           "replicates_sha256": sha256(bootstrap_path)})
    raw_validation = validate_raw("primary")
    training = require_training()
    pair_checks = [load_json(ROOT / entry["path"])["checks"] for entry in training["replicates"].values()]
    integrity = {"raw_rankings_valid": raw_validation.get("status") == "PASS",
                 "equal_budget_valid": all(all(checks.values()) for checks in pair_checks),
                 "leakage_count": len(load_json(MANIFESTS / "final_split_audit.json").get("errors", [])),
                 "same_tasks_across_branches_and_seeds": len({tuple(sorted(values)) for values in list(control.values()) + list(distill.values())}) == 1,
                 "all_six_replicates_complete": len(control) == len(distill) == 6,
                 "budget_manifest_bound": sha256(MANIFESTS / "budget_manifest.json") == load_json(ROOT / "CONFIG_SELECTION_FROZEN.json")["budget_manifest_sha256"]}
    integrity["pass"] = (integrity["raw_rankings_valid"] and integrity["equal_budget_valid"] and
                         integrity["leakage_count"] == 0 and integrity["same_tasks_across_branches_and_seeds"] and
                         integrity["all_six_replicates_complete"] and integrity["budget_manifest_bound"])
    positive = primary["criteria"]["composition_effect_pass"] and integrity["pass"]
    decision = ("POSITIVE_COMPOSITION_CONFIRMATION" if positive else
                "NO_CONFIRMATORY_COMPOSITION_EVIDENCE" if integrity["pass"] else "INVALID_CONFIRMATION_INTEGRITY")
    result = {"schema": "stage4.composition.confirmation.primary-analysis.v1", "status": "DONE",
              "study_class": CONFIG["study_class"], "primary": primary, "integrity": integrity,
              "decision": {"status": decision, "positive_composition_confirmation": positive},
              "bindings": {"preregistration_sha256": sha256(MANIFESTS / "preregistration.json"),
                           "config_frozen_sha256": sha256(ROOT / "CONFIG_FROZEN.json"),
                           "budget_manifest_sha256": sha256(MANIFESTS / "budget_manifest.json"),
                           "final_data_manifest_sha256": sha256(MANIFESTS / "final_data_manifest.json"),
                           "primary_raw_validation_sha256": sha256(MANIFESTS / "raw_validation_primary.json")}}
    dump_json(STATISTICS / "PRIMARY_ANALYSIS.json", result)
    print(json.dumps({"milestone": "primary_analysis_done", "decision": decision,
                      "mean_delta": primary["seed_statistics"]["mean_delta"],
                      "ci95": primary["seed_statistics"]["t_ci95"],
                      "p": primary["seed_statistics"]["p_two_sided"],
                      "positive_replicates": primary["seed_statistics"]["positive_seed_count"]}), flush=True)
    return result


def evaluate_secondary() -> dict:
    require_frozen()
    require_training()
    secondary = ["final_b", "final_c", "final_d",
                 "final_a_depth2", "final_b_depth2", "final_c_depth2", "final_d_depth2",
                 "final_a_depth4", "final_b_depth4", "final_c_depth4", "final_d_depth4"]
    results = {"shared_atomic_base": evaluate_branch("shared", "atomic_base", None,
                                                      ["final_a", *secondary], evaluate_atomic=True)}
    for rep in CONFIG["replicates"]:
        label = int(rep["replicate_label"])
        results[str(label)] = {}
        for branch in ("atomic_control", "composition_distill"):
            results[str(label)][branch] = evaluate_branch(label, branch, _adapter(label, branch),
                                                          secondary, evaluate_atomic=True)
        print(json.dumps({"milestone": "secondary_replicate_done", "replicate": label}), flush=True)
    receipt = {"schema": "stage4.composition.confirmation.secondary-evaluation.v1", "status": "DONE",
               "secondary_splits": secondary, "shared_base_once": True,
               "replicates": {key: {branch: {"model_sha256": value["model_sha256"]}
                                             for branch, value in branches.items()}
                              for key, branches in results.items() if key != "shared_atomic_base"}}
    dump_json(RUNS / "SECONDARY_EVALUATION_DONE.json", receipt)
    return receipt


def _endpoint_stats(splits: list[str]) -> dict:
    control, distill = _metric_matrix(splits, "hit@32")
    deltas = []
    for label in sorted(control):
        tasks = sorted(control[label])
        if tasks != sorted(distill[label]):
            raise RuntimeError("secondary task identity mismatch")
        deltas.append(sum(distill[label][task] - control[label][task] for task in tasks) / len(tasks))
    stats = seed_t_statistics(deltas)
    return {"splits": splits, "seed_deltas": deltas, "seed_statistics": stats}


def finalize_analysis() -> dict:
    full_raw_validation = validate_raw("full")
    primary_receipt = load_json(STATISTICS / "PRIMARY_ANALYSIS.json")
    if not (RUNS / "SECONDARY_EVALUATION_DONE.json").is_file():
        raise RuntimeError("secondary evaluation is incomplete")
    endpoints = {
        "final_b_depth3_hit32": _endpoint_stats(["final_b"]),
        "final_c_depth3_hit32": _endpoint_stats(["final_c"]),
        "final_d_depth3_hit32": _endpoint_stats(["final_d"]),
        "transfer_bcd_depth2_hit32": _endpoint_stats(["final_b_depth2", "final_c_depth2", "final_d_depth2"]),
        "transfer_bcd_depth4_hit32": _endpoint_stats(["final_b_depth4", "final_c_depth4", "final_d_depth4"]),
    }
    adjusted = holm({name: value["seed_statistics"]["p_two_sided"] for name, value in endpoints.items()})
    mass_control, mass_distill = _metric_matrix(["final_a"], "correct_mass")
    mass_deltas = {}
    for label in sorted(mass_control):
        tasks = sorted(mass_control[label])
        mass_deltas[str(label)] = sum(mass_distill[label][task] - mass_control[label][task] for task in tasks) / len(tasks)
    base_atomic_path = RANKINGS / "replicateshared" / "atomic" / "replicateshared_atomic_base.json"
    base_atomic = load_json(base_atomic_path)
    forgetting = {}
    for rep in CONFIG["replicates"]:
        label = int(rep["replicate_label"])
        forgetting[str(label)] = {}
        for branch in ("atomic_control", "composition_distill"):
            post = load_json(RANKINGS / f"replicate{label}" / "atomic" / f"replicate{label}_{branch}.json")
            forgetting[str(label)][branch] = eval_lib.aligned_forgetting(base_atomic, post)
            forgetting[str(label)][branch]["gating"] = False
    result = {"schema": "stage4.composition.confirmation.analysis.v1", "status": "DONE",
              "study_class": CONFIG["study_class"], "primary": primary_receipt["primary"],
              "secondary_holm": {"alpha": 0.05,
                                 "p_source": "two_sided_seed_level_one_sample_t_test",
                                 "family": endpoints, "results": adjusted},
              "descriptive_non_gating": {"atomic_forgetting": forgetting,
                                         "correct_mass_seed_deltas": mass_deltas,
                                         "historical_dev_b_delta": CONFIG["historical_pilot"]["dev_b_delta"]},
              "integrity": {**primary_receipt["integrity"], "full_raw_rankings_valid": True,
                            "full_raw_validation_sha256": sha256(MANIFESTS / "raw_validation_full.json")},
              "decision": primary_receipt["decision"],
              "bindings": primary_receipt["bindings"]}
    validate_schema(result, "analysis.schema.json")
    dump_json(STATISTICS / "FINAL_ANALYSIS.json", result)
    return result


def _expected_ranking_jobs(mode: str, frozen: dict) -> list[dict]:
    if mode not in ("primary", "full"):
        raise ValueError("raw validation mode must be primary or full")
    primary_splits = ["final_a"]
    secondary_splits = ["final_b", "final_c", "final_d",
                        "final_a_depth2", "final_b_depth2", "final_c_depth2", "final_d_depth2",
                        "final_a_depth4", "final_b_depth4", "final_c_depth4", "final_d_depth4"]
    jobs = []
    training = load_json(RUNS / "TRAINING_DONE.json")
    for rep in CONFIG["replicates"]:
        label = int(rep["replicate_label"])
        pair_path = ROOT / training["replicates"][str(label)]["path"]
        pair = load_json(pair_path)
        for branch in ("atomic_control", "composition_distill"):
            model_sha = pair["adapters"][branch]["tree_sha256"]
            for split in (primary_splits if mode == "primary" else [*primary_splits, *secondary_splits]):
                jobs.append({"label": label, "branch": branch, "split": split,
                             "eval_name": _eval_name(label, branch, split),
                             "model_sha": model_sha, "rng_seed": int(rep["rng_seed"]),
                             "pair_sha": sha256(pair_path)})
    if mode == "full":
        for split in [*primary_splits, *secondary_splits]:
            jobs.append({"label": "shared", "branch": "atomic_base", "split": split,
                         "eval_name": _eval_name("shared", "atomic_base", split),
                         "model_sha": CONFIG["source"]["atomic_export_tree_sha256"],
                         "rng_seed": None, "pair_sha": None})
    return jobs


def validate_raw(mode: str = "full") -> dict:
    frozen = require_frozen()
    require_training()
    task_index = {}
    for filename, entry in frozen["test_files"].items():
        if filename == "confirm_atomic.jsonl":
            continue
        for row in read_jsonl(DATA / filename):
            task_index[row["task_id"]] = row
    reports = []
    expected_global_paths = set()
    for job in _expected_ranking_jobs(mode, frozen):
        label, branch, split = job["label"], job["branch"], job["split"]
        rows = read_jsonl(DATA / f"{split}.jsonl")
        expected_binding = _binding(frozen, label, branch, split, job["model_sha"],
                                    job["rng_seed"], job["pair_sha"])
        root = RANKINGS / f"replicate{label}"
        ranking_dir = root / "rankings" / job["eval_name"]
        metric_dir = root / "metrics" / job["eval_name"]
        expected_parts = math.ceil(len(rows) / 25)
        expected_rank_names = {f"part-{index:05d}.jsonl.gz" for index in range(expected_parts)}
        expected_metric_names = {f"part-{index:05d}.jsonl" for index in range(expected_parts)}
        expected_receipt_names = {f"part-{index:05d}.receipt.json" for index in range(expected_parts)}
        if ({path.name for path in ranking_dir.glob("part-*.jsonl.gz")} != expected_rank_names or
                {path.name for path in metric_dir.glob("part-*.jsonl")} != expected_metric_names or
                {path.name for path in metric_dir.glob("part-*.receipt.json")} != expected_receipt_names):
            raise RuntimeError(f"exact shard set mismatch: {job['eval_name']}")
        for index in range(expected_parts):
            shard_rows = rows[index * 25:(index + 1) * 25]
            ranking_path = ranking_dir / f"part-{index:05d}.jsonl.gz"
            metric_path = metric_dir / f"part-{index:05d}.jsonl"
            receipt_path = metric_dir / f"part-{index:05d}.receipt.json"
            expected_global_paths.add(ranking_path.resolve())
            rankings = read_jsonl(ranking_path)
            metrics = read_jsonl(metric_path)
            receipt = load_json(receipt_path)
            checks = {"branch": receipt.get("branch") == job["eval_name"],
                      "binding": receipt.get("binding") == expected_binding,
                      "task_ids": receipt.get("task_ids") == [row["task_id"] for row in shard_rows],
                      "ranking_hash": receipt.get("ranking_sha256", "").upper() == sha256(ranking_path),
                      "metrics_hash": receipt.get("metrics_sha256", "").upper() == sha256(metric_path),
                      "row_count": len(rankings) == len(metrics) == len(shard_rows) == receipt.get("ranking_rows") == receipt.get("metrics_rows")}
            if not all(checks.values()):
                raise RuntimeError(f"ranking shard receipt failed: {ranking_path}: {checks}")
            for task, raw, stored in zip(shard_rows, rankings, metrics):
                if raw.get("branch") != job["eval_name"] or raw.get("binding") != expected_binding:
                    raise RuntimeError(f"raw ranking identity/binding mismatch: {task['task_id']}")
                derived = eval_lib._ranking_metric_from_raw(task, raw, job["eval_name"], expected_binding)
                if derived != stored:
                    raise RuntimeError(f"raw-derived metrics mismatch: {task['task_id']}")
            reports.append({"path": ranking_path.relative_to(ROOT).as_posix(),
                            "rows": len(rankings), "sha256": sha256(ranking_path),
                            "replicate_label": label, "branch": branch, "split": split})
    expected_shards = 480 if mode == "primary" else 2080
    if len(reports) != expected_shards:
        raise RuntimeError(f"exact raw matrix mismatch: {len(reports)} != {expected_shards}")
    if mode == "full":
        actual_global_paths = {path.resolve() for path in RANKINGS.rglob("rankings/*/part-*.jsonl.gz")}
        if actual_global_paths != expected_global_paths:
            raise RuntimeError("full raw ranking tree contains missing or extra shards")
    result = {"schema": "stage4.composition.confirmation.raw-validation.v1", "status": "PASS",
              "mode": mode, "ranking_shards": len(reports), "expected_ranking_shards": expected_shards,
              "branch_split_jobs": len(_expected_ranking_jobs(mode, frozen)), "reports": reports,
              "full_enumeration_rederived": True, "lexicographic_ties_rechecked": True,
              "exact_verifier_rechecked": True}
    dump_json(MANIFESTS / f"raw_validation_{mode}.json", result)
    return result


def accounting() -> dict:
    training_receipts = list(ADAPTERS.rglob("training_receipt.json"))
    eval_receipts = list(RANKINGS.rglob("*.receipt.json"))
    depth_by_task = {}
    for data_path in DATA.glob("*.jsonl"):
        for row in read_jsonl(data_path):
            depth_by_task[row["task_id"]] = int(row["depth"])
    return {"training": {"branches": len(training_receipts),
                         "optimizer_steps": sum(load_json(path)["optimizer_steps"] for path in training_receipts),
                         "loss_bearing_target_tokens": sum(load_json(path)["loss_bearing_target_tokens_seen"] for path in training_receipts),
                         "input_tokens": sum(load_json(path)["input_tokens_seen"] for path in training_receipts),
                         "wall_seconds": sum(load_json(path)["wall_seconds"] for path in training_receipts)},
            "ranking": {"shards": len(eval_receipts),
                        "candidate_program_scores": sum(load_json(path)["ranking_rows"] *
                                                        (5 ** depth_by_task[load_json(path)["task_ids"][0]])
                                                        for path in eval_receipts)},
            "atomic_descriptive": {"evaluations": len(list(RANKINGS.rglob("atomic/*.json")))}}


def validate_json_artifacts() -> dict:
    records = []
    excluded = {ARCHIVES.resolve()}
    self_referential_release_files = {"reports/REPORT_RECEIPT.json", "manifests/validation_manifest.json",
                                      "manifests/archive_manifest.json"}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or any(root == path.resolve() or root in path.resolve().parents for root in excluded):
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative in self_referential_release_files:
            continue
        if path.suffix == ".json":
            load_json(path)
            records.append({"path": relative, "kind": "json", "bytes": path.stat().st_size,
                            "sha256": sha256(path)})
        elif path.suffix == ".jsonl" or str(path).endswith(".jsonl.gz"):
            rows = read_jsonl(path)
            records.append({"path": relative, "kind": "jsonl", "rows": len(rows),
                            "bytes": path.stat().st_size, "sha256": sha256(path)})
    result = {"schema": "stage4.composition.confirmation.validation-manifest.v1", "status": "PASS",
              "files": records, "json_files": sum(row["kind"] == "json" for row in records),
              "jsonl_files": sum(row["kind"] == "jsonl" for row in records),
              "self_referential_release_files_excluded": sorted(self_referential_release_files)}
    dump_json(MANIFESTS / "validation_manifest.json", result)
    return result


def write_run_index() -> dict:
    records = []
    for path in sorted(RUNS.rglob("*.json")):
        value = load_json(path)
        records.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
                        "schema": value.get("schema"), "status": value.get("status"),
                        "bytes": path.stat().st_size})
    result = {"schema": "stage4.composition.confirmation.run-index.v1", "status": "DONE",
              "runs": records, "done": sum(row["status"] == "DONE" for row in records),
              "failed": sum(row["status"] == "FAILED" for row in records)}
    dump_json(MANIFESTS / "run_index.json", result)
    return result


def write_plots(analysis: dict) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["replicate_label"]) for row in analysis["primary"]["per_seed"]]
    deltas = [row["delta_pp"] for row in analysis["primary"]["per_seed"]]
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    colors = ["#17823b" if value > 0 else "#b3261e" for value in deltas]
    axis.bar(labels, deltas, color=colors)
    axis.axhline(5.0, color="#444444", linestyle="--", linewidth=1.2, label="registered +5 pp threshold")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Confirmation replicate")
    axis.set_ylabel("Hit@32 distill - control (percentage points)")
    axis.set_title("Frozen final-A composition effect")
    axis.legend(loc="best")
    fig.tight_layout()
    path = REPORTS / "primary_seed_deltas.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return [path]


def report() -> dict:
    raw = validate_raw()
    analysis = finalize_analysis()
    plots = write_plots(analysis)
    stats = analysis["primary"]["seed_statistics"]
    decision = analysis["decision"]["status"]
    lines = ["# Stage 4 composition-only confirmation", "",
             f"**Decision: {decision}.**", "",
             "The model is Qwen3-0.6B throughout. The historical pilot remains unchanged and formally negative under its old gates.", "",
             "## Pre-freeze failure and recovery amendment", "",
             "Freeze attempt 0 failed before any final file or evaluator access because the legacy non-gating confirm-atomic schedule required field/degree cells with zero remaining disjoint states.",
             "The byte-identical FAILED receipt and logs are preserved. Recovery Amendment 001 was committed while final data, rankings and metrics were absent; it changed only confirm-atomic feasibility handling and retried freeze once.",
             "Final A/B/C/D generation, all seeds and sizes, the primary endpoint and gates, all statistics, and all 12 trained adapters remained byte-identical.", "",
             "## Primary final-A result", "",
             f"- Mean `composition_distill - atomic_control` Hit@32: **{stats['mean_delta_pp']:.3f} pp**.",
             f"- 95% seed t-CI: **[{100*stats['t_ci95'][0]:.3f}, {100*stats['t_ci95'][1]:.3f}] pp**.",
             f"- Two-sided seed t-test: **p={stats['p_two_sided']:.8g}**.",
             f"- Positive replicates: **{stats['positive_seed_count']}/6**.",
             f"- Exact sign-flip sensitivity: **p={analysis['primary']['exact_sign_flip']['p_two_sided']:.8g}**.",
             f"- Crossed seed-by-task bootstrap 95% CI: **[{100*analysis['primary']['seed_task_bootstrap']['ci95'][0]:.3f}, {100*analysis['primary']['seed_task_bootstrap']['ci95'][1]:.3f}] pp**.", "",
             "## Interpretation", "",
             "A positive primary decision supports improved ranking of unseen state/task-disjoint family-A depth-3 compositions in known fields. It does not by itself prove held-motif or new-field transfer.",
             f"The exploratory dev-B transfer delta observed before this preregistration was {100*CONFIG['historical_pilot']['dev_b_delta']:.1f} pp and is disclosed as negative.", "",
             "SH1, PLAN, APPLY and atomic forgetting are reported descriptively and never veto the composition decision.", "",
             "## Per-replicate primary deltas", "",
             "| Replicate | Control hits | Distill hits | Delta (pp) |", "|---:|---:|---:|---:|"]
    for row in analysis["primary"]["per_seed"]:
        lines.append(f"| {row['replicate_label']} | {row['control_hits']} | {row['distill_hits']} | {row['delta_pp']:.3f} |")
    lines += ["", "## Integrity", "",
              f"- Raw ranking shards independently rederived: {raw['ranking_shards']}.",
              "- All programs were exactly enumerated, verifier labels recomputed, and score/tie order checked.",
               "- Within each replicate, control and distill used equal optimizer steps, epochs, effective batch and loss-bearing target tokens.",
               "- Final splits passed state/task disjointness and held-motif audits.",
               f"- Recovery amendment receipt SHA-256: `{sha256(AMENDMENT_RECEIPT_PATH)}`.",
               f"- Freeze-attempt-0 FAILED SHA-256: `{sha256(FAILURE_LEDGER / 'FAILED.json')}`.", ""]
    REPORTS.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS / "FINAL_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    rows_path = REPORTS / "primary_seed_results.csv"
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(analysis["primary"]["per_seed"][0]))
        writer.writeheader()
        writer.writerows(analysis["primary"]["per_seed"])
    run_index = write_run_index()
    validation = validate_json_artifacts()
    receipt = {"schema": "stage4.composition.confirmation.report.v1", "status": "DONE",
               "decision": decision, "report": manifest_entry(report_path),
               "primary_csv": manifest_entry(rows_path), "analysis_sha256": sha256(STATISTICS / "FINAL_ANALYSIS.json"),
               "raw_validation_sha256": sha256(MANIFESTS / "raw_validation_full.json"),
               "plots": [manifest_entry(path) for path in plots],
               "run_index_sha256": sha256(MANIFESTS / "run_index.json"),
                "validation_manifest_sha256": sha256(MANIFESTS / "validation_manifest.json"),
                "recovery_amendment_sha256": sha256(AMENDMENT_RECEIPT_PATH),
                "freeze_attempt0_failed_sha256": sha256(FAILURE_LEDGER / "FAILED.json"),
                "accounting": accounting()}
    dump_json(REPORTS / "REPORT_RECEIPT.json", receipt)
    return receipt


def _archive_items(full: bool) -> list[tuple[Path, str]]:
    excluded_roots = {ARCHIVES.resolve()}
    items = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(root == path.resolve() or root in path.resolve().parents for root in excluded_roots):
            continue
        relative = path.relative_to(ROOT).as_posix()
        keep_failure_log = relative.startswith("runs/failures/") and relative.endswith(".log")
        if ("/__pycache__/" in f"/{relative}/" or relative.endswith((".pyc", ".pid")) or
                (relative.endswith(".log") and not keep_failure_log) or
                relative.startswith(".pytest_cache/")):
            continue
        if not full and (relative.startswith("adapters/") or relative.startswith("rankings/") or
                         relative.endswith(".npy") or relative.endswith(".safetensors")):
            continue
        items.append((path, relative))
    if full:
        for source in runtime_dependency_paths():
            relative = source.relative_to(REPO.resolve()).as_posix()
            items.append((source, f"imports/runtime_dependencies/{relative}"))
    return sorted(items, key=lambda item: item[1])


def _write_zip(path: Path, items: list[tuple[Path, str]]) -> dict:
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for source, archive_name in items:
            archive.write(source, archive_name)
    temporary.replace(path)
    with zipfile.ZipFile(path, "r") as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"archive CRC failed: {bad}")
        names = archive.namelist()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path), "members": len(names)}


def archive() -> dict:
    if not (REPORTS / "REPORT_RECEIPT.json").is_file():
        raise RuntimeError("report must be complete before archiving")
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    experiment_terminal = {"schema": "stage4.composition.confirmation.experiment-terminal.v1",
                           "status": "DONE", "decision": load_json(STATISTICS / "FINAL_ANALYSIS.json")["decision"],
                           "report_sha256": sha256(REPORTS / "FINAL_REPORT.md"),
                           "analysis_sha256": sha256(STATISTICS / "FINAL_ANALYSIS.json"),
                           "raw_validation_sha256": sha256(MANIFESTS / "raw_validation_full.json")}
    dump_json(RUNS / "EXPERIMENT_DONE.json", experiment_terminal)
    write_run_index()
    validate_json_artifacts()
    report_receipt_path = REPORTS / "REPORT_RECEIPT.json"
    report_receipt = load_json(report_receipt_path)
    report_receipt["run_index_sha256"] = sha256(MANIFESTS / "run_index.json")
    report_receipt["validation_manifest_sha256"] = sha256(MANIFESTS / "validation_manifest.json")
    dump_json(report_receipt_path, report_receipt)
    release_content = {"schema": "stage4.composition.confirmation.release-content.v1", "status": "FROZEN",
                       "root_files": [{"path": archive_name, "bytes": source.stat().st_size,
                                       "sha256": sha256(source)}
                                      for source, archive_name in _archive_items(True)],
                       "runtime_dependencies": runtime_dependency_manifest(),
                       "legacy_runtime_code_included_in_full": True,
                       "legacy_model_or_dataset_payload_included": False,
                       "manifest_self_excluded_from_member_list": True}
    dump_json(MANIFESTS / "release_content_manifest.json", release_content)
    compact_path = ARCHIVES / "stage4_composition_confirmation_audit.zip"
    full_path = ARCHIVES / "stage4_composition_confirmation_full.zip"
    compact = _write_zip(compact_path, _archive_items(False))
    full = _write_zip(full_path, _archive_items(True))
    receipt = {"schema": "stage4.composition.confirmation.archives.v1", "status": "DONE",
               "compact": compact, "full": full,
               "legacy_runtime_code_included_in_full": True,
               "legacy_model_or_dataset_payload_included": False,
               "release_content_manifest_sha256": sha256(MANIFESTS / "release_content_manifest.json")}
    dump_json(MANIFESTS / "archive_manifest.json", receipt)
    dump_json(RUNS / "STAGE4_COMPOSITION_DONE.json", {"schema": "stage4.composition.confirmation.terminal.v1",
                                                       "status": "DONE", "decision": load_json(STATISTICS / "FINAL_ANALYSIS.json")["decision"],
                                                       "archive_manifest_sha256": sha256(MANIFESTS / "archive_manifest.json")})
    return receipt


def run_all() -> dict:
    acquire_run_lease()
    try:
        activity("require_preregistered")
        require_preregistered()
        activity("prepare_budgets")
        prepare_budgets()
        activity("train_all")
        train_all()
        activity("freeze_final")
        freeze_final()
        activity("evaluate_primary")
        evaluate_primary()
        activity("analyze_primary")
        analyze_primary_result()
        activity("evaluate_secondary")
        evaluate_secondary()
        activity("report")
        report()
        activity("archive")
        result = archive()
        activity("complete", "DONE", archive_manifest_sha256=sha256(MANIFESTS / "archive_manifest.json"))
        return result
    except Exception as exc:
        activity("failed", "FAILED", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        release_run_lease()


def run_recovery() -> dict:
    """Resume only at freeze after the prospectively frozen v1.1 amendment."""

    acquire_run_lease()
    try:
        activity("require_recovery_amendment")
        require_recovery_amendment()
        activity("freeze_final")
        freeze_final()
        activity("evaluate_primary")
        evaluate_primary()
        activity("analyze_primary")
        analyze_primary_result()
        activity("evaluate_secondary")
        evaluate_secondary()
        activity("report")
        report()
        activity("archive")
        result = archive()
        activity("complete", "DONE", archive_manifest_sha256=sha256(MANIFESTS / "archive_manifest.json"))
        return result
    except Exception as exc:
        activity("failed", "FAILED", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        release_run_lease()


def _write_failure_receipt(failure: dict) -> Path:
    primary = RUNS / "FAILED.json"
    if not primary.exists():
        dump_json(primary, failure, exclusive=True)
        return primary
    failures = RUNS / "failures"
    failures.mkdir(parents=True, exist_ok=True)
    command = str(failure.get("command", "unknown")).replace("/", "_").replace("\\", "_")
    index = 1
    while True:
        path = failures / f"{command}_attempt{index}_FAILED.json"
        if not path.exists():
            dump_json(path, failure, exclusive=True)
            return path
        index += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("source-audit", "preregister", "prepare-budgets", "train",
                                            "freeze", "eval-primary", "analyze-primary", "eval-secondary",
                                            "validate-raw", "analyze", "report", "archive", "amend-recovery",
                                            "recover", "all"))
    args = parser.parse_args()
    ensure_roots()
    commands = {"source-audit": source_audit, "preregister": preregister,
                "prepare-budgets": prepare_budgets, "train": train_all, "freeze": freeze_final,
                "eval-primary": evaluate_primary, "analyze-primary": analyze_primary_result,
                "eval-secondary": evaluate_secondary, "validate-raw": validate_raw,
                "analyze": finalize_analysis, "report": report, "archive": archive,
                "amend-recovery": amend_recovery, "recover": run_recovery, "all": run_all}
    try:
        result = commands[args.command]()
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    except Exception as exc:
        failure = {"schema": "stage4.composition.confirmation.failure.v1", "status": "FAILED",
                   "command": args.command, "pid": os.getpid(), "time_unix": time.time(),
                   "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        if RECOVERY_FREEZE_STARTED_PATH.is_file():
            failure["recovery_freeze_started_sha256"] = sha256(RECOVERY_FREEZE_STARTED_PATH)
        failure_path = _write_failure_receipt(failure)
        print(json.dumps({"failure_receipt": failure_path.relative_to(ROOT).as_posix(),
                          "failure_sha256": sha256(failure_path)}, sort_keys=True), file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
