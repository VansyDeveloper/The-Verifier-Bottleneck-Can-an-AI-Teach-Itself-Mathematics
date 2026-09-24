from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent.parent
ATOMIC_ROOT = REPO / "artifacts/stage4_sh1_v5_0p6b"
DATA = ROOT / "data"
RUNS = ROOT / "runs"
ADAPTERS = ROOT / "adapters"
RANKINGS = ROOT / "rankings"
MANIFESTS = ROOT / "manifests"
REPORTS = ROOT / "reports"
ATOMIC_EXPORT = ROOT / "atomic_export" / "frozen_atomic_0p6b"
sys.path.insert(0, str(ROOT / "code"))

from composition_core import (AtomicReference, DiscoveryThresholds, OPS, audit_atomic_reference,
                              audit_pilot_data, exact_discover, generate_confirm_atomic_data,
                              generate_final_data, generate_pilot_data)

PROTOCOL_PATH = ROOT / "configs/protocol.json"
ATOMIC_AUTHORIZATION_PATH = ROOT / "configs/exploratory_atomic_authorization.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
_FINAL_LEASE_HANDLES = {}


class NoFeasiblePilotAttempt(RuntimeError):
    """Scientific selection outcome, distinct from infrastructure/runtime failure."""


def dump_json(path: Path, value) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def dump_json_exclusive(path: Path, value) -> None:
    """Atomically claim a marker path; never overwrite a competing writer."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())


def dump_json_atomic(path: Path, value) -> None:
    """Durably replace a mutable receipt without exposing truncated JSON."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.partial")
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with partial.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    partial.replace(path)


def pid_alive(pid: int) -> bool:
    if not isinstance(pid, int) or pid <= 0: return False
    if os.name == "nt":
        # On Windows ``os.kill(pid, 0)`` is destructive: Python maps a
        # non-CTRL signal to TerminateProcess.  Query the process handle
        # instead, so a heartbeat can never kill the evaluator it observes.
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            # ERROR_ACCESS_DENIED still proves that the PID exists.
            return ctypes.get_last_error() == 5
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False


def acquire_final_lease() -> None:
    """Hold an OS-released exclusive lease for the entire confirm evaluator."""
    path = MANIFESTS / "final_evaluation.lease.lock"; key = str(path.resolve())
    if key in _FINAL_LEASE_HANDLES: return
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b"); handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"0"); handle.flush(); os.fsync(handle.fileno())
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close(); raise RuntimeError("final evaluation lease is held by another live process") from exc
    _FINAL_LEASE_HANDLES[key] = handle


def release_final_lease() -> None:
    path = MANIFESTS / "final_evaluation.lease.lock"; key = str(path.resolve())
    handle = _FINAL_LEASE_HANDLES.pop(key, None)
    if handle is None: return
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


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n")
    partial.replace(path)


def read_jsonl(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def hash_value(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest().upper()


def tree_sha256(path: Path) -> str:
    path = Path(path); digest = hashlib.sha256(); files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files: raise RuntimeError(f"empty tree: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode()); digest.update(b"\0"); digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest().upper()


def repo_relative(path: Path) -> str:
    return Path(path).resolve().relative_to(REPO.resolve()).as_posix()


def ensure_roots() -> None:
    for path in (DATA, RUNS, ADAPTERS, RANKINGS, MANIFESTS, REPORTS): path.mkdir(parents=True, exist_ok=True)


def preregistration_manifest() -> dict:
    import platform
    import peft
    import torch
    import transformers
    path = MANIFESTS / "preregistration.json"
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    git_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    if not path.exists() and git_status.strip(): raise RuntimeError(f"preregistration requires a clean Git tree:\n{git_status}")
    payload = {"schema": "stage4.composition.v5.preregistration-manifest.v1", "status": "FROZEN",
               "model": PROTOCOL["model"], "git_commit": git_commit, "python": platform.python_version(),
               "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
               "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
               "protocol_sha256": sha256(PROTOCOL_PATH), "preregistration_sha256": sha256(ROOT / "PREREGISTRATION.md"),
               "exploratory_atomic_authorization_sha256": sha256(ATOMIC_AUTHORIZATION_PATH),
               "code": _code_manifest(),
               "schemas": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                           for path in sorted((ROOT / "schemas").glob("*.schema.json"))}}
    if path.exists() and load_json(path) != payload: raise RuntimeError("preregistration manifest changed")
    if not path.exists(): dump_json_atomic(path, payload)
    return payload


def protocol_guard() -> None:
    if PROTOCOL["model"] != "Qwen/Qwen3-0.6B" or PROTOCOL["model_size_lock"] != "0.6B":
        raise RuntimeError("protocol is not locked to Qwen3-0.6B")
    serialized = json.dumps(PROTOCOL)
    if "1.7B" in serialized or "1p7b" in serialized:
        raise RuntimeError("forbidden model size appears in protocol")
    if exploratory_mode() and not (
        PROTOCOL.get("claim_scope") == "exploratory_signal_only"
        and PROTOCOL.get("prepilot_atomic_rescue") is False
        and PROTOCOL.get("source_checkpoint_mutation") is False
        and PROTOCOL.get("execution_scope") == "discover_and_seed0_pilot_only"
        and PROTOCOL.get("registered_stage4_result_remains") == "FAILED_CALIBRATION_GATE"
        and PROTOCOL.get("numeric_runtime", {}).get("atomic_export_storage_dtype") == "float32"
        and PROTOCOL.get("numeric_runtime", {}).get("branch_master_dtype") == "float32"
        and PROTOCOL.get("numeric_runtime", {}).get("branch_compute_dtype") == "bfloat16_autocast"
        and PROTOCOL.get("interpretation", {}).get("confirmatory_stage4_success_allowed") is False
        and PROTOCOL.get("interpretation", {}).get("multi_seed_execution_in_this_protocol") is False
        and PROTOCOL.get("interpretation", {}).get("parent_negative_endpoint_may_be_rewritten") is False
    ):
        raise RuntimeError("exploratory execution/claim lock is invalid")


def atomic_reference_manifest(paths: list[Path], selected_adapter: Path | None = None) -> dict:
    files = []
    unique = {Path(path).resolve() for path in paths}
    adapter_record = None
    if selected_adapter is not None:
        selected_adapter = Path(selected_adapter).resolve()
        adapter_files = sorted(path for path in selected_adapter.rglob("*") if path.is_file())
        if not adapter_files: raise RuntimeError("selected atomic adapter tree is empty")
        unique.update(path.resolve() for path in adapter_files)
        receipt = selected_adapter / "training_receipt.json"
        adapter_record = {"path": repo_relative(selected_adapter), "tree_sha256": tree_sha256(selected_adapter),
                          "receipt_path": repo_relative(receipt), "receipt_sha256": sha256(receipt),
                          "files_count": len(adapter_files), "bytes": sum(path.stat().st_size for path in adapter_files)}
    for path in sorted(unique):
        files.append({"path": repo_relative(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    payload = {"schema": "stage4.composition.v5.atomic-references.v1", "files": files}
    if adapter_record is not None: payload["selected_adapter"] = adapter_record
    dump_json_atomic(MANIFESTS / "atomic_reference.json", payload)
    return payload


def exploratory_mode() -> bool:
    return PROTOCOL.get("study_class") == "EXPLORATORY_FOLLOWUP"


def _repo_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute(): path = REPO / path
    resolved = path.resolve()
    if resolved != REPO.resolve() and REPO.resolve() not in resolved.parents:
        raise RuntimeError(f"exploratory source escapes repository: {value}")
    return resolved


def _require_exploratory_atomic_source() -> dict:
    if not ATOMIC_AUTHORIZATION_PATH.is_file():
        raise RuntimeError("exploratory atomic authorization is missing")
    authorization = load_json(ATOMIC_AUTHORIZATION_PATH)
    if not (
        authorization.get("schema") == "stage4.distill.v5.exploratory-atomic-authorization.v1"
        and authorization.get("status") == "AUTHORIZED_FOR_EXPLORATION"
        and authorization.get("study_class") == "EXPLORATORY_FOLLOWUP"
        and authorization.get("export_authorized") is True
        and authorization.get("registered_atomic_pass") is False
        and authorization.get("old_atomic_final_access_forbidden") is True
        and authorization.get("model") == PROTOCOL["model"]
    ):
        raise RuntimeError("invalid exploratory atomic authorization")
    preregistration_path = MANIFESTS / "preregistration.json"
    if not preregistration_path.is_file():
        raise RuntimeError("exploratory source check requires frozen preregistration")
    preregistration = load_json(preregistration_path)
    if (preregistration.get("protocol_sha256") != sha256(PROTOCOL_PATH) or
            preregistration.get("exploratory_atomic_authorization_sha256") != sha256(ATOMIC_AUTHORIZATION_PATH)):
        raise RuntimeError("exploratory authorization is not bound by preregistration")
    bound_paths: dict[str, Path] = {}
    for name, entry in authorization.get("source_files", {}).items():
        path = _repo_path(entry["path"])
        if not path.is_file() or sha256(path).upper() != str(entry["sha256"]).upper():
            raise RuntimeError(f"exploratory source hash mismatch: {name}")
        bound_paths[name] = path
    required = {"atomic_decision", "calibration_gate", "calibration_generations", "training_receipt",
                "corrective_seal", "atomic_protocol", "data_manifest", "calibration_data",
                "corrective_data_manifest", "corrective_allocation", "corrective_sh1_data",
                "corrective_control_data", "corrective_plan_raw", "plan_raw_audit_receipt"}
    if set(bound_paths) != required:
        raise RuntimeError(f"exploratory source file set mismatch: {sorted(set(bound_paths) ^ required)}")
    decision = load_json(bound_paths["atomic_decision"])
    gate = load_json(bound_paths["calibration_gate"])
    receipt = load_json(bound_paths["training_receipt"])
    seal = load_json(bound_paths["corrective_seal"])
    atomic_protocol = load_json(bound_paths["atomic_protocol"])
    data_manifest = load_json(bound_paths["data_manifest"])
    corrective_manifest = load_json(bound_paths["corrective_data_manifest"])
    plan_audit_receipt = load_json(bound_paths["plan_raw_audit_receipt"])
    selected = _repo_path(authorization["selected_adapter"])
    if selected != (ATOMIC_ROOT / "adapters/corrective").resolve() or not selected.is_dir():
        raise RuntimeError("exploratory source is not the frozen corrective adapter")
    selected_tree = tree_sha256(selected).upper()
    if selected_tree != str(authorization["selected_adapter_tree_sha256"]).upper():
        raise RuntimeError("exploratory corrective adapter tree mismatch")
    if not (
        decision.get("status") == "FAILED_CALIBRATION_GATE"
        and decision.get("composition_unlocked") is False
        and decision.get("final_gate") is None
        and decision.get("selected_adapter") is None
        and decision.get("calibration_gate") == gate
    ):
        raise RuntimeError("registered negative atomic endpoint changed")
    calibration_entry = data_manifest.get("calibration.jsonl", {})
    calibration_rows = read_jsonl(bound_paths["calibration_generations"])
    plan_raw_rows = read_jsonl(bound_paths["corrective_plan_raw"])
    plan_run_receipts = [item for item in plan_audit_receipt.get("runs", [])
                         if item.get("run_id") == "corrective_calibration"]
    plan_run_receipt = plan_run_receipts[0] if len(plan_run_receipts) == 1 else {}
    corrective_rows = {
        "corrective_sh1.jsonl": read_jsonl(bound_paths["corrective_sh1_data"]),
        "corrective_control.jsonl": read_jsonl(bound_paths["corrective_control_data"]),
    }
    corrective_data_checks = {}
    for filename, rows in corrective_rows.items():
        path_key = "corrective_sh1_data" if filename == "corrective_sh1.jsonl" else "corrective_control_data"
        path = bound_paths[path_key]; entry = corrective_manifest.get(filename, {})
        corrective_data_checks[filename] = (
            str(entry.get("sha256", "")).upper() == sha256(path).upper()
            and entry.get("bytes") == path.stat().st_size
            and entry.get("rows") == len(rows)
            and str(seal.get("input_sha256", {}).get(filename, "")).upper() == sha256(path).upper()
        )
    raw_validated_gate = _validate_atomic_gate(bound_paths["calibration_gate"], selected,
                                               "calibration", "corrective_calibration")
    if raw_validated_gate != gate:
        raise RuntimeError("raw-derived corrective gate differs from authorized gate")
    binding = gate.get("binding", {})
    structural_checks = {
        "registered_atomic_pass": gate.get("pass") is True,
        "registered_atomic_failed": gate.get("pass") is False,
        "model_locked_0p6b": atomic_protocol.get("model") == PROTOCOL["model"] == binding.get("model"),
        "adapter_bound": str(binding.get("adapter_tree_sha256", "")).upper() == selected_tree,
        "calibration_bound": (str(binding.get("dataset_sha256", "")).upper() ==
                              str(calibration_entry.get("sha256", "")).upper() ==
                              sha256(bound_paths["calibration_data"]).upper()),
        "generation_bound": (str(gate.get("generation_sha256", "")).upper() ==
                             sha256(bound_paths["calibration_generations"]).upper()
                             and gate.get("generation_rows") == len(calibration_rows)),
        "receipt_done": receipt.get("status") == "DONE" and receipt.get("phase", {}).get("status") == "DONE",
        "seal_bound": (seal.get("phase") == "corrective" and
                       str(seal.get("output_tree_sha256", "")).upper() == selected_tree and
                       str(seal.get("receipt_sha256", "")).upper() == sha256(bound_paths["training_receipt"]).upper() and
                       str(seal.get("protocol_sha256", "")).upper() == sha256(bound_paths["atomic_protocol"]).upper() and
                       str(seal.get("parent_tree_sha256", "")).upper() ==
                       tree_sha256(ATOMIC_ROOT / "adapters/curriculum").upper()),
        "corrective_inputs_bound": all(corrective_data_checks.values()),
        "corrective_allocation_bound": bound_paths["corrective_allocation"].is_file(),
        "raw_gate_rederived": raw_validated_gate == gate,
        "plan_raw_audit_bound": (
            plan_audit_receipt.get("status") == "DONE"
            and plan_audit_receipt.get("final_dataset_accessed") is False
            and plan_run_receipt.get("matches_registered_gate") is True
            and str(plan_run_receipt.get("raw_sha256", "")).upper() == sha256(bound_paths["corrective_plan_raw"]).upper()
            and str(plan_run_receipt.get("gate_sha256", "")).upper() == sha256(bound_paths["calibration_gate"]).upper()
            and plan_run_receipt.get("rows") == len(plan_raw_rows) == gate.get("generation_rows")
        ),
        "registered_plan_check_passed": gate.get("checks", {}).get("plan_ge_095") is True,
        "registered_controls_check_passed": gate.get("checks", {}).get("control_forgetting_le_002") is True,
        "registered_sh1_check_failed": gate.get("checks", {}).get("apply_each_ge_090") is False,
        "registered_spread_check_failed": gate.get("checks", {}).get("spread_le_010") is False,
        "old_atomic_final_unopened": (not (ATOMIC_ROOT / "manifests/final_access_receipt.json").exists()
                                      and not (ATOMIC_ROOT / "runs/generations/atomic_v5_final.jsonl").exists()),
    }
    floor = PROTOCOL["atomic"]["exploratory_floor"]
    controls = [gate["apply"]["by_operation"][operation] for operation in OPS if operation != "SH1"]
    eligibility_checks = {
        "plan_overall_ge_floor": gate["plan"]["overall"] >= floor["plan_overall"],
        "sh1_apply_ge_floor": gate["apply"]["by_operation"]["SH1"] >= floor["sh1_apply"],
        "control_apply_each_ge_floor": min(controls) >= floor["control_apply_each"],
    }
    required_structural = [key for key in structural_checks if key != "registered_atomic_pass"]
    if structural_checks["registered_atomic_pass"] is not False or not all(structural_checks[key] for key in required_structural):
        raise RuntimeError(f"exploratory atomic structural checks failed: {structural_checks}")
    if not all(eligibility_checks.values()):
        raise RuntimeError(f"exploratory atomic floor failed: {eligibility_checks}")
    reference_paths = [ATOMIC_AUTHORIZATION_PATH, *bound_paths.values()]
    for filename in ("coordinate_source.jsonl", "full_sh1.jsonl", "control_apply.jsonl", "calibration.jsonl",
                     "corrective_sh1.jsonl", "corrective_control.jsonl"):
        reference_paths.append(ATOMIC_ROOT / "data" / filename)
    reference_paths.extend([ATOMIC_ROOT / "manifests/corrective_data_manifest.json",
                            ATOMIC_ROOT / "manifests/corrective_allocation.json"])
    atomic_reference_manifest(reference_paths, selected_adapter=selected)
    audit = {"schema": "stage4.composition.v5.exploratory-source-audit.v1", "status": "PASS",
             "study_class": "EXPLORATORY_FOLLOWUP", "registered_atomic_pass": False,
             "registered_stage4_result": decision["status"], "selected_adapter": repo_relative(selected),
             "selected_adapter_tree_sha256": selected_tree, "structural_checks": structural_checks,
             "eligibility_checks": eligibility_checks, "corrective_data_checks": corrective_data_checks,
             "source_metrics": {"plan": gate["plan"], "apply": gate["apply"]},
             "old_atomic_final_accessed": False, "authorization_sha256": sha256(ATOMIC_AUTHORIZATION_PATH)}
    dump_json_atomic(MANIFESTS / "exploratory_source_audit.json", audit)
    return {"decision": authorization, "decision_path": ATOMIC_AUTHORIZATION_PATH,
            "registered_decision": decision, "selected_adapter": selected, "final_gate": None,
            "final_gate_path": None, "access_receipt": None, "source_audit": audit}


def require_atomic_pass() -> dict:
    protocol_guard()
    if exploratory_mode():
        return _require_exploratory_atomic_source()
    decision_path = ATOMIC_ROOT / "runs/ATOMIC_DECISION.json"
    final_gate_path = ATOMIC_ROOT / "runs/atomic_v5_final_gate.json"
    access_path = ATOMIC_ROOT / "manifests/final_access_receipt.json"
    data_manifest_path = ATOMIC_ROOT / "manifests/data_manifest.json"
    atomic_protocol_path = ATOMIC_ROOT / "configs/protocol.json"
    for path in (decision_path, final_gate_path, access_path, data_manifest_path, atomic_protocol_path):
        if not path.is_file(): raise RuntimeError(f"atomic PASS evidence is missing: {path}")
    atomic_protocol = load_json(atomic_protocol_path)
    if atomic_protocol.get("model") != PROTOCOL["model"]: raise RuntimeError("atomic model is not Qwen3-0.6B")
    decision = load_json(decision_path); final_gate = load_json(final_gate_path); access = load_json(access_path)
    if decision.get("status") != "PASS" or decision.get("composition_unlocked") is not True:
        raise RuntimeError("composition remains locked by atomic decision")
    if final_gate.get("pass") is not True or decision.get("final_gate") != final_gate:
        raise RuntimeError("atomic final gate is absent, failed, or differs from decision")
    selected = Path(decision.get("selected_adapter", ""))
    if not selected.is_dir() or ATOMIC_ROOT.resolve() not in selected.resolve().parents:
        raise RuntimeError("selected atomic adapter is missing or outside atomic v5")
    if final_gate.get("binding", {}).get("model") != PROTOCOL["model"]:
        raise RuntimeError("atomic final gate model binding mismatch")
    if final_gate.get("binding", {}).get("adapter_tree_sha256", "").upper() != tree_sha256(selected):
        raise RuntimeError("selected adapter tree differs from atomic final gate")
    data_manifest = load_json(data_manifest_path); final_entry = data_manifest["final.jsonl"]
    if final_gate.get("binding", {}).get("dataset_sha256", "").upper() != final_entry["sha256"].upper():
        raise RuntimeError("atomic final dataset binding mismatch")
    generation_path = ATOMIC_ROOT / "runs/generations/atomic_v5_final.jsonl"
    if (not generation_path.is_file() or final_gate.get("generation_sha256", "").upper() != sha256(generation_path) or
            final_gate.get("generation_rows") != final_entry["rows"]):
        raise RuntimeError("atomic final raw generations are incomplete or mismatched")
    if (access.get("status") != "DONE" or access.get("gate_sha256", "").upper() != sha256(final_gate_path) or
            access.get("generation_sha256", "").upper() != sha256(generation_path)):
        raise RuntimeError("atomic exactly-once final access receipt mismatch")
    selected_receipt = selected / "training_receipt.json"
    if not selected_receipt.is_file(): raise RuntimeError("selected atomic adapter lacks DONE receipt")
    reference_paths = [atomic_protocol_path, data_manifest_path, decision_path, final_gate_path, generation_path,
                       access_path, selected_receipt]
    for filename in ("coordinate_source.jsonl", "full_sh1.jsonl", "control_apply.jsonl", "calibration.jsonl", "final.jsonl"):
        reference_paths.append(ATOMIC_ROOT / "data" / filename)
    if selected.name == "corrective":
        for path in (ATOMIC_ROOT / "manifests/corrective_data_manifest.json",
                     ATOMIC_ROOT / "manifests/corrective_allocation.json",
                     ATOMIC_ROOT / "runs/corrective_calibration_gate.json",
                     ATOMIC_ROOT / "data/corrective_sh1.jsonl", ATOMIC_ROOT / "data/corrective_control.jsonl"):
            if not path.is_file(): raise RuntimeError(f"corrective-selected atomic evidence is missing: {path}")
            reference_paths.append(path)
    atomic_reference_manifest(reference_paths, selected_adapter=selected)
    return {"decision": decision, "decision_path": decision_path, "selected_adapter": selected,
            "final_gate": final_gate, "final_gate_path": final_gate_path, "access_receipt": access}


def build_atomic_reference() -> AtomicReference:
    evidence = require_atomic_pass()
    manifest = load_json(ATOMIC_ROOT / "manifests/data_manifest.json")
    split_files = {
        "coordinate_source": "coordinate_source.jsonl", "full_sh1": "full_sh1.jsonl",
        "control_apply": "control_apply.jsonl", "calibration": "calibration.jsonl",
    }
    if not exploratory_mode(): split_files["final"] = "final.jsonl"
    corrective_manifest = ATOMIC_ROOT / "manifests/corrective_data_manifest.json"
    if evidence["selected_adapter"].name == "corrective":
        manifest = {**manifest, **load_json(corrective_manifest)}
        split_files.update({"corrective_sh1": "corrective_sh1.jsonl", "corrective_control": "corrective_control.jsonl"})
    raw = {name: read_jsonl(ATOMIC_ROOT / "data" / filename) for name, filename in split_files.items()}
    required_splits = ("calibration",) if exploratory_mode() else ("calibration", "final")
    reference = audit_atomic_reference(manifest, raw, required_splits=required_splits, strict=True)
    report = dict(reference.report)
    if exploratory_mode():
        report["source_authorization_sha256"] = sha256(evidence["decision_path"])
        report["registered_atomic_decision_sha256"] = sha256(
            _repo_path(evidence["decision"]["source_files"]["atomic_decision"]["path"]))
        report["registered_atomic_pass"] = False
    else:
        report["atomic_decision_sha256"] = sha256(evidence["decision_path"])
    dump_json_atomic(MANIFESTS / "atomic_reference_audit.json", report)
    return reference


def import_atomic() -> dict:
    ensure_roots(); preregistration_manifest(); evidence = require_atomic_pass()
    from composition_model import export_atomic_checkpoint
    calibration = read_jsonl(ATOMIC_ROOT / "data/calibration.jsonl")
    receipt = export_atomic_checkpoint(evidence["decision_path"], ATOMIC_EXPORT, calibration)
    dump_json_atomic(MANIFESTS / "atomic_import.json", receipt)
    return receipt


def _data_manifest(paths: list[Path]) -> dict:
    return {"schema": "stage4.composition.v5.data-manifest.v1", "status": "FROZEN",
            "files": {path.name: {"rows": len(read_jsonl(path)), "bytes": path.stat().st_size, "sha256": sha256(path)}
                      for path in paths}}


def _data_manifest_files(manifest: dict) -> dict:
    files = manifest.get("files")
    if not isinstance(files, dict): raise RuntimeError("composition data manifest has no files mapping")
    return files


def prepare() -> dict:
    ensure_roots(); preregistration_manifest(); reference = build_atomic_reference()
    if (ROOT / "CONFIG_FROZEN.json").exists() or list(DATA.glob("final_*.jsonl")):
        raise RuntimeError("pilot prepare refuses pre-existing final data/freeze")
    counts = {int(depth): int(count) for depth, count in PROTOCOL["pilot_data"]["train_depth_counts"].items()}
    generation_seeds = PROTOCOL["pilot_data"]["generation_seeds"]
    splits, generation_audit = generate_pilot_data(reference, train_depth_counts=counts,
                                                     dev_a_count=PROTOCOL["pilot_data"]["dev_a"],
                                                     dev_b_count=PROTOCOL["pilot_data"]["dev_b"],
                                                     train_seed=generation_seeds["train"], dev_a_seed=generation_seeds["dev_a"],
                                                     dev_b_seed=generation_seeds["dev_b"])
    audit = audit_pilot_data(splits, atomic_reference=reference, verify_shortest=True)
    if not audit.get("ok"): raise RuntimeError(f"pilot split audit failed: {audit.get('errors')}")
    thresholds = DiscoveryThresholds(PROTOCOL["pilot_data"]["min_unique_trajectories"],
                                     PROTOCOL["pilot_data"]["min_solvable_tasks"],
                                     PROTOCOL["pilot_data"]["min_full_signatures"],
                                     PROTOCOL["pilot_data"]["min_position_count"],
                                     PROTOCOL["pilot_data"]["max_first_operation_spread"])
    trajectories, discover = exact_discover(splits["train"],
                                             target_count=PROTOCOL["pilot_data"]["discover_target_trajectories"],
                                             thresholds=thresholds)
    if not discover.get("passes_minimums"): raise RuntimeError(f"Discover thresholds failed: {discover}")
    paths = []
    for name, rows in (*sorted(splits.items()), ("discover_trajectories", trajectories)):
        path = DATA / f"{name}.jsonl"; write_jsonl(path, rows); paths.append(path)
    manifest = _data_manifest(paths); manifest_files = _data_manifest_files(manifest)
    dump_json_atomic(MANIFESTS / "pilot_data_manifest.json", manifest)
    dump_json_atomic(MANIFESTS / "pilot_split_audit.json", audit)
    dump_json_atomic(MANIFESTS / "pilot_generation_audit.json", generation_audit)
    dump_json_atomic(MANIFESTS / "discover_audit.json", discover)
    result = {"schema": "stage4.composition.v5.prepare.v1", "status": "DONE", "model": PROTOCOL["model"],
              "data": manifest_files, "split_audit": audit, "discover": discover,
              "protocol_sha256": sha256(PROTOCOL_PATH), "final_created": False}
    dump_json_atomic(RUNS / "PREPARE_DONE.json", result); return result


def require_prepared() -> dict:
    receipt_path = RUNS / "PREPARE_DONE.json"
    if not receipt_path.is_file(): raise RuntimeError("pilot data are not prepared")
    receipt = load_json(receipt_path)
    if receipt.get("status") != "DONE" or receipt.get("protocol_sha256") != sha256(PROTOCOL_PATH):
        raise RuntimeError("stale PREPARE_DONE receipt")
    manifest = _data_manifest_files(load_json(MANIFESTS / "pilot_data_manifest.json"))
    for name, entry in manifest.items():
        path = DATA / name
        if not path.is_file() or sha256(path) != entry["sha256"] or path.stat().st_size != entry["bytes"] or len(read_jsonl(path)) != entry["rows"]:
            raise RuntimeError(f"pilot data manifest mismatch: {name}")
    return receipt


def _balanced_atomic_pool(seed: int, per_operation: int = 1000):
    import random
    rng = random.Random(seed)
    sh1 = read_jsonl(ATOMIC_ROOT / "data/full_sh1.jsonl")
    controls = read_jsonl(ATOMIC_ROOT / "data/control_apply.jsonl")
    by_operation = {"SH1": sh1}
    for op in OPS[1:]: by_operation[op] = [row for row in controls if row["operation"] == op]
    pool = []
    for op in OPS:
        candidates = by_operation[op]
        if len(candidates) < per_operation: raise RuntimeError(f"insufficient atomic pool for {op}")
        pool.extend(rng.sample(candidates, per_operation))
    rng.shuffle(pool); return pool


def _resolved_config(*, seed: int, lr: float, epochs: int, replay: float) -> dict:
    resolved = deepcopy(PROTOCOL["pilot_initial"])
    resolved.update({"schema": "stage4.composition.v5.resolved-config.v1", "seed": int(seed), "lr": float(lr),
                     "epochs": int(epochs), "atomic_replay": float(replay)})
    return resolved


def _attempt_id(resolved: dict) -> str:
    lr = f"{resolved['lr']:.0e}".replace("-", "m").replace("+", "p")
    replay = f"{resolved['atomic_replay']:.2f}".replace(".", "p")
    return f"seed{resolved['seed']}_lr{lr}_e{resolved['epochs']}_r{replay}"


def _manifest_entry_for_split(split: str) -> dict:
    filename = f"{split}.jsonl"
    for manifest_path in (MANIFESTS / "pilot_data_manifest.json", MANIFESTS / "final_data_manifest.json"):
        if manifest_path.exists():
            files = _data_manifest_files(load_json(manifest_path))
            if filename in files: return files[filename]
    raise RuntimeError(f"no frozen manifest entry for split {split}")


def _branch_binding(resolved: dict, attempt_id: str, branch: str, split: str, model_sha: str,
                    atomic_split: str = "calibration") -> dict:
    data_entry = _manifest_entry_for_split(split)
    atomic_entry = (load_json(ATOMIC_ROOT / "manifests/data_manifest.json")["calibration.jsonl"]
                    if atomic_split == "calibration" else _manifest_entry_for_split(atomic_split))
    return {"schema": "stage4.composition.v5.eval-binding.v1", "seed": resolved["seed"], "attempt": attempt_id,
            "branch": branch, "split": split, "model_sha256": model_sha, "data_sha256": data_entry["sha256"],
            "atomic_data_sha256": atomic_entry["sha256"],
            "resolved_config_sha256": hash_value(resolved), "protocol_sha256": sha256(PROTOCOL_PATH),
            "core_sha256": sha256(ROOT / "code/composition_core.py"),
            "evaluator_sha256": sha256(ROOT / "code/composition_eval.py")}


def _evaluate_branch(export_dir: Path, adapter: Path | None, resolved: dict, attempt_id: str, branch: str,
                     split: str, rows: list[dict], atomic_rows: list[dict], out_dir: Path,
                     atomic_split: str = "calibration"):
    from composition_eval import aligned_forgetting, evaluate_atomic, evaluate_ranking, unload
    from composition_model import _export_payload_sha, load_branch, tree_sha256 as model_tree_sha
    model_sha = _export_payload_sha(export_dir).upper() if adapter is None else model_tree_sha(adapter).upper()
    binding = _branch_binding(resolved, attempt_id, branch, split, model_sha, atomic_split)
    model, tokenizer, token_ids = load_branch(export_dir, adapter)
    try:
        summary, metrics = evaluate_ranking(model, tokenizer, token_ids, rows, out_dir, branch, binding,
                                            batch_size=PROTOCOL["ranking"]["batch_size"])
        atomic = evaluate_atomic(model, tokenizer, token_ids, atomic_rows, out_dir, branch, binding)
    finally:
        unload(model)
    return summary, metrics, atomic, binding


def _read_attempt_metrics(eval_dir: Path, branch: str):
    metric_dir = eval_dir / "metrics" / branch
    rows = []
    for path in sorted(metric_dir.glob("part-*.jsonl")): rows.extend(read_jsonl(path))
    return rows


def _evaluate_shared_pilot_base(dev_a: list[dict], atomic_rows: list[dict]):
    resolved = _resolved_config(seed=0, lr=PROTOCOL["pilot_initial"]["lr"],
                                epochs=PROTOCOL["pilot_initial"]["epochs"], replay=PROTOCOL["pilot_initial"]["atomic_replay"])
    return _evaluate_branch(ATOMIC_EXPORT, None, resolved, "shared_atomic_base", "atomic_base", "dev_a",
                             dev_a, atomic_rows, RUNS / "pilot/shared_atomic_base/eval")


def _scientific_atomic_view(value: dict) -> dict:
    return {kind: {"overall": value[kind]["overall"], "by_operation": value[kind]["by_operation"]}
            for kind in ("plan", "apply")}


def validate_frozen_source_identity(base_atomic: dict) -> dict:
    """Prove that merge/reload preserved the exact frozen calibration behavior."""
    authorization = load_json(ATOMIC_AUTHORIZATION_PATH)
    source_files = authorization["source_files"]
    gate_path = _repo_path(source_files["calibration_gate"]["path"])
    source_apply_path = _repo_path(source_files["calibration_generations"]["path"])
    source_plan_path = _repo_path(source_files["corrective_plan_raw"]["path"])
    base_path = RUNS / "pilot/shared_atomic_base/eval/atomic/atomic_base.generations.jsonl"
    if not base_path.is_file(): raise RuntimeError("frozen atomic-base raw generations are missing")
    gate = load_json(gate_path)
    expected_metrics = {kind: {"overall": gate[kind]["overall"], "by_operation": gate[kind]["by_operation"]}
                        for kind in ("plan", "apply")}
    actual_metrics = _scientific_atomic_view(base_atomic)
    if actual_metrics != expected_metrics:
        raise RuntimeError(f"frozen atomic-base metrics changed after merge/reload: {actual_metrics}")
    source_apply = {row["task_id"]: row for row in read_jsonl(source_apply_path)}
    source_plan = {row["task_id"]: row for row in read_jsonl(source_plan_path)}
    base_rows = {row["task_id"]: row for row in read_jsonl(base_path)}
    if len(source_apply) != len(source_plan) or len(source_apply) != len(base_rows):
        raise RuntimeError("frozen atomic-base task inventory length mismatch")
    if set(source_apply) != set(source_plan) or set(source_apply) != set(base_rows):
        raise RuntimeError("frozen atomic-base task inventory mismatch")
    mismatches = []
    for task_id in sorted(base_rows):
        expected_apply = source_apply[task_id]; expected_plan = source_plan[task_id]; actual = base_rows[task_id]
        checks = (
            actual.get("operation") == expected_apply.get("operation"),
            actual.get("raw_completion") == expected_apply.get("raw_completion"),
            actual.get("correct") == expected_apply.get("correct"),
            actual.get("plan_prediction") == expected_plan.get("predicted_operation"),
            actual.get("plan_correct") == expected_plan.get("correct"),
        )
        if not all(checks): mismatches.append(task_id)
    if mismatches:
        raise RuntimeError(f"frozen atomic-base per-task behavior changed for {len(mismatches)} tasks")
    receipt = {"schema": "stage4.composition.v5.frozen-source-identity.v1", "status": "PASS",
               "study_class": "EXPLORATORY_FOLLOWUP", "registered_atomic_pass": False,
               "tasks": len(base_rows), "metrics": actual_metrics, "per_task_exact_match": True,
               "source_gate_sha256": sha256(gate_path), "source_apply_sha256": sha256(source_apply_path),
               "source_plan_sha256": sha256(source_plan_path), "base_generation_sha256": sha256(base_path),
               "authorization_sha256": sha256(ATOMIC_AUTHORIZATION_PATH), "old_atomic_final_accessed": False}
    dump_json_atomic(MANIFESTS / "frozen_source_identity.json", receipt)
    return receipt


def _without_runtime_fields(value):
    if isinstance(value, dict):
        return {key: _without_runtime_fields(item) for key, item in value.items()
                if key not in {"wall_seconds", "started_at_unix", "finished_at_unix"}}
    if isinstance(value, list): return [_without_runtime_fields(item) for item in value]
    return value


def _pilot_required_check_names() -> tuple[str, ...]:
    source = "exploratory_source_authorized" if exploratory_mode() else "registered_atomic_pass"
    return (source, "frozen_source_metrics_match", "base_hit32_range", "delta_hit32_ge_008", "forgetting_le_002",
            "correct_mass_growth", "leakage_zero", "equal_budget")


def _pilot_pass(checks: dict) -> bool:
    return all(checks.get(name) is True for name in _pilot_required_check_names())


def _validate_cached_pilot_attempt(prior: dict) -> dict:
    """Re-open every hash-bound cache before it can influence selection/freeze."""
    resolved = prior["resolved_config"]
    provenance = prior.get("provenance", {})
    expected_provenance = {"protocol_sha256": sha256(PROTOCOL_PATH),
                           "core_sha256": sha256(ROOT / "code/composition_core.py"),
                           "trainer_sha256": sha256(ROOT / "code/composition_model.py"),
                           "evaluator_sha256": sha256(ROOT / "code/composition_eval.py")}
    if provenance != expected_provenance: raise RuntimeError("cached pilot code/protocol provenance mismatch")
    attempt_id = prior["attempt_id"]; eval_dir = RUNS / "pilot" / attempt_id / "eval"
    dev_a = read_jsonl(DATA / "dev_a.jsonl"); atomic_rows = read_jsonl(ATOMIC_ROOT / "data/calibration.jsonl")
    from composition_eval import aligned_forgetting
    base_summary, _, base_atomic, _ = _evaluate_shared_pilot_base(dev_a, atomic_rows)
    source_identity = validate_frozen_source_identity(base_atomic)
    summaries = {}; atomics = {}
    branches = (("atomic_base", base_summary, base_atomic, None),
                ("atomic_control", None, None, Path(prior["adapters"]["atomic_control"])),
                ("composition_distill", None, None, Path(prior["adapters"]["composition_distill"])))
    for branch, known_summary, known_atomic, adapter in branches:
        if branch == "atomic_base":
            summary, atomic = known_summary, known_atomic
        else:
            summary, _, atomic, _ = _evaluate_branch(ATOMIC_EXPORT, adapter, resolved, attempt_id, branch,
                                                      "dev_a", dev_a, atomic_rows, eval_dir)
        if _without_runtime_fields(summary) != _without_runtime_fields(prior["summaries"][branch]):
            raise RuntimeError(f"cached pilot summary mismatch: {branch}")
        prior_atomic = prior["atomic"][{"atomic_base": "base", "atomic_control": "control",
                                        "composition_distill": "distill"}[branch]]
        if _without_runtime_fields(atomic) != _without_runtime_fields(prior_atomic):
            raise RuntimeError(f"cached pilot atomic evaluation mismatch: {branch}")
        summaries[branch] = summary; atomics[branch] = atomic
    budget = prior["budget"]
    receipts = {branch: load_json(Path(path) / "training_receipt.json")
                for branch, path in prior["adapters"].items()}
    control_receipt, distill_receipt = receipts["atomic_control"], receipts["composition_distill"]
    from composition_model import _export_payload_sha
    receipt_bindings_ok = (
        control_receipt.get("input_sha256") == budget["control_input_sha256"] and
        distill_receipt.get("input_sha256") == budget["distill_input_sha256"] and
        str(control_receipt.get("resolved_config_sha256", "")).upper() == hash_value(resolved) and
        str(distill_receipt.get("resolved_config_sha256", "")).upper() == hash_value(resolved) and
        control_receipt.get("atomic_export_sha256") == distill_receipt.get("atomic_export_sha256") == _export_payload_sha(ATOMIC_EXPORT))
    budget_checks = {
        "optimizer_steps_equal": control_receipt["optimizer_steps"] == distill_receipt["optimizer_steps"],
        "optimizer_steps_expected": control_receipt["optimizer_steps"] == budget["optimizer_steps_expected"],
        "loss_tokens_equal": control_receipt["loss_bearing_target_tokens_seen"] == distill_receipt["loss_bearing_target_tokens_seen"],
        "loss_tokens_expected": control_receipt["loss_bearing_target_tokens_seen"] == distill_receipt["loss_bearing_target_tokens_seen"] ==
                                budget["loss_bearing_target_tokens_each"] * resolved["epochs"],
        "effective_batch_equal": control_receipt["effective_batch"] == distill_receipt["effective_batch"] == resolved["effective_batch"],
        "epochs_equal": control_receipt["epochs"] == distill_receipt["epochs"] == resolved["epochs"],
        "receipts_done": control_receipt.get("status") == distill_receipt.get("status") == "DONE",
        "receipt_bindings": receipt_bindings_ok,
    }
    if budget_checks != prior.get("budget_checks") or not all(budget_checks.values()):
        raise RuntimeError(f"cached pilot budget mismatch: {budget_checks}")
    control_forgetting = aligned_forgetting(atomics["atomic_base"], atomics["atomic_control"])
    distill_forgetting = aligned_forgetting(atomics["atomic_base"], atomics["composition_distill"])
    delta = summaries["composition_distill"]["hit@32"] - summaries["atomic_control"]["hit@32"]
    mass_delta = summaries["composition_distill"]["correct_mass"] - summaries["atomic_control"]["correct_mass"]
    max_forgetting = max(control_forgetting["max_drop"], distill_forgetting["max_drop"])
    split_audit = load_json(MANIFESTS / "pilot_split_audit.json"); discover_audit = load_json(MANIFESTS / "discover_audit.json")
    leakage_count = (len(split_audit.get("train_heldout_motif_leaks", [])) + sum(split_audit.get("atomic_state_overlap", {}).values()) +
                     sum(split_audit.get("atomic_task_overlap", {}).values()) + len(discover_audit.get("held_motif_leaks", [])))
    gates = PROTOCOL["pilot_gates"]
    scientific_checks = {"atomic_pass": not exploratory_mode(),
                         "registered_atomic_pass": not exploratory_mode(),
                         "exploratory_source_authorized": exploratory_mode(),
                         "frozen_source_metrics_match": source_identity.get("status") == "PASS",
                         "base_hit32_range": gates["base_hit32_min"] <= summaries["atomic_base"]["hit@32"] <= gates["base_hit32_max"],
                         "delta_hit32_ge_008": delta >= gates["delta_hit32_min"],
                         "forgetting_le_002": max_forgetting <= gates["max_atomic_forgetting"] + 1e-12,
                         "correct_mass_growth": mass_delta > gates["correct_mass_delta_min_exclusive"],
                         "leakage_zero": leakage_count == gates["leakage_count"], "equal_budget": all(budget_checks.values())}
    expected_values = {"delta_hit32": delta, "correct_mass_delta": mass_delta, "max_forgetting": max_forgetting,
                       "leakage_count": leakage_count, "checks": scientific_checks,
                       "source_identity_sha256": sha256(MANIFESTS / "frozen_source_identity.json"),
                       "pass": _pilot_pass(scientific_checks)}
    if any(prior.get(key) != value for key, value in expected_values.items()):
        raise RuntimeError("cached pilot scientific decision differs from raw bound artifacts")
    return prior


def run_pilot_attempt(resolved: dict) -> dict:
    require_prepared(); import_atomic()
    from composition_eval import aligned_forgetting
    from composition_model import (_export_payload_sha, atomic_example_specs, distill_example_specs, encode_specs,
                                   pack_control_to_budget, tokenizer_and_ids, train_branch)
    attempt_id = _attempt_id(resolved); attempt_dir = RUNS / "pilot" / attempt_id
    decision_path = attempt_dir / "attempt.json"
    if decision_path.exists():
        prior = load_json(decision_path)
        if prior.get("status") == "DONE" and prior.get("resolved_config") == resolved:
            return _validate_cached_pilot_attempt(prior)
        raise RuntimeError(f"mismatched cached pilot attempt: {attempt_id}")
    trajectories = read_jsonl(DATA / "discover_trajectories.jsonl")
    dev_a = read_jsonl(DATA / "dev_a.jsonl"); atomic_rows = read_jsonl(ATOMIC_ROOT / "data/calibration.jsonl")
    base_summary, _, base_atomic, _ = _evaluate_shared_pilot_base(dev_a, atomic_rows)
    source_identity = validate_frozen_source_identity(base_atomic)
    atomic_pool = _balanced_atomic_pool(resolved["seed"] + 1000)
    tokenizer, _ = tokenizer_and_ids(ATOMIC_EXPORT)
    distill_specs = distill_example_specs(trajectories, atomic_pool, resolved["atomic_replay"], resolved["seed"])
    control_specs = atomic_example_specs(atomic_pool)
    distill_encoded = encode_specs(tokenizer, distill_specs); control_encoded = encode_specs(tokenizer, control_specs)
    control_encoded, distill_encoded, budget = pack_control_to_budget(control_encoded, distill_encoded, resolved["seed"])
    budget.update({"optimizer_steps_expected": math.ceil(math.ceil(len(distill_encoded) / resolved["micro_batch"]) /
                   (resolved["effective_batch"] // resolved["micro_batch"])) * resolved["epochs"],
                   "effective_batch": resolved["effective_batch"], "epochs": resolved["epochs"],
                   "control_input_sha256": hash_value(control_encoded), "distill_input_sha256": hash_value(distill_encoded)})
    attempt_dir.mkdir(parents=True, exist_ok=True); dump_json_atomic(attempt_dir / "resolved_config.json", resolved)
    dump_json_atomic(attempt_dir / "budget.json", budget)
    control_adapter = ADAPTERS / "pilot" / attempt_id / "atomic_control"
    distill_adapter = ADAPTERS / "pilot" / attempt_id / "composition_distill"
    control_receipt = train_branch(ATOMIC_EXPORT, control_adapter, control_encoded, resolved, "atomic_control", budget["control_input_sha256"])
    distill_receipt = train_branch(ATOMIC_EXPORT, distill_adapter, distill_encoded, resolved, "composition_distill", budget["distill_input_sha256"])
    budget_checks = {
        "optimizer_steps_equal": control_receipt["optimizer_steps"] == distill_receipt["optimizer_steps"],
        "optimizer_steps_expected": control_receipt["optimizer_steps"] == budget["optimizer_steps_expected"],
        "loss_tokens_equal": control_receipt["loss_bearing_target_tokens_seen"] == distill_receipt["loss_bearing_target_tokens_seen"],
        "loss_tokens_expected": control_receipt["loss_bearing_target_tokens_seen"] == distill_receipt["loss_bearing_target_tokens_seen"] ==
                                budget["loss_bearing_target_tokens_each"] * resolved["epochs"],
        "effective_batch_equal": control_receipt["effective_batch"] == distill_receipt["effective_batch"] == resolved["effective_batch"],
        "epochs_equal": control_receipt["epochs"] == distill_receipt["epochs"] == resolved["epochs"],
        "receipts_done": control_receipt.get("status") == distill_receipt.get("status") == "DONE",
        "receipt_bindings": (control_receipt.get("input_sha256") == budget["control_input_sha256"] and
                             distill_receipt.get("input_sha256") == budget["distill_input_sha256"] and
                             str(control_receipt.get("resolved_config_sha256", "")).upper() == hash_value(resolved) and
                             str(distill_receipt.get("resolved_config_sha256", "")).upper() == hash_value(resolved) and
                             control_receipt.get("atomic_export_sha256") == distill_receipt.get("atomic_export_sha256") ==
                             _export_payload_sha(ATOMIC_EXPORT)),
    }
    if not all(budget_checks.values()): raise RuntimeError(f"equal-budget training failed: {budget_checks}")
    eval_dir = attempt_dir / "eval"
    control_summary, _, control_atomic, _ = _evaluate_branch(ATOMIC_EXPORT, control_adapter, resolved, attempt_id, "atomic_control", "dev_a", dev_a, atomic_rows, eval_dir)
    distill_summary, _, distill_atomic, _ = _evaluate_branch(ATOMIC_EXPORT, distill_adapter, resolved, attempt_id, "composition_distill", "dev_a", dev_a, atomic_rows, eval_dir)
    control_forgetting = aligned_forgetting(base_atomic, control_atomic)
    distill_forgetting = aligned_forgetting(base_atomic, distill_atomic)
    split_audit = load_json(MANIFESTS / "pilot_split_audit.json"); discover_audit = load_json(MANIFESTS / "discover_audit.json")
    leakage_count = (len(split_audit.get("train_heldout_motif_leaks", [])) + sum(split_audit.get("atomic_state_overlap", {}).values()) +
                     sum(split_audit.get("atomic_task_overlap", {}).values()) + len(discover_audit.get("held_motif_leaks", [])))
    gates = PROTOCOL["pilot_gates"]; delta = distill_summary["hit@32"] - control_summary["hit@32"]
    mass_delta = distill_summary["correct_mass"] - control_summary["correct_mass"]
    max_forgetting = max(control_forgetting["max_drop"], distill_forgetting["max_drop"])
    checks = {
        "atomic_pass": not exploratory_mode(),
        "registered_atomic_pass": not exploratory_mode(),
        "exploratory_source_authorized": exploratory_mode(),
        "frozen_source_metrics_match": source_identity.get("status") == "PASS",
        "base_hit32_range": gates["base_hit32_min"] <= base_summary["hit@32"] <= gates["base_hit32_max"],
        "delta_hit32_ge_008": delta >= gates["delta_hit32_min"],
        "forgetting_le_002": max_forgetting <= gates["max_atomic_forgetting"] + 1e-12,
        "correct_mass_growth": mass_delta > gates["correct_mass_delta_min_exclusive"],
        "leakage_zero": leakage_count == gates["leakage_count"],
        "equal_budget": all(budget_checks.values()),
    }
    result = {"schema": "stage4.composition.v5.pilot-attempt.v1", "status": "DONE", "attempt_id": attempt_id,
              "resolved_config": resolved, "budget": budget, "budget_checks": budget_checks,
              "adapters": {"atomic_control": str(control_adapter), "composition_distill": str(distill_adapter)},
              "summaries": {"atomic_base": base_summary, "atomic_control": control_summary, "composition_distill": distill_summary},
              "atomic": {"base": base_atomic, "control": control_atomic, "distill": distill_atomic,
                         "control_forgetting": control_forgetting, "distill_forgetting": distill_forgetting},
              "delta_hit32": delta, "correct_mass_delta": mass_delta, "max_forgetting": max_forgetting,
              "leakage_count": leakage_count, "checks": checks,
              "source_identity_sha256": sha256(MANIFESTS / "frozen_source_identity.json"),
              "pass": _pilot_pass(checks),
              "pass_scope": "EXPLORATORY_SEED0_PILOT_ONLY",
              "study_class": PROTOCOL.get("study_class", "REGISTERED"),
              "claim_scope": PROTOCOL.get("claim_scope", "confirmatory"),
              "provenance": {"protocol_sha256": sha256(PROTOCOL_PATH),
                             "core_sha256": sha256(ROOT / "code/composition_core.py"),
                             "trainer_sha256": sha256(ROOT / "code/composition_model.py"),
                             "evaluator_sha256": sha256(ROOT / "code/composition_eval.py")}}
    dump_json_atomic(decision_path, result); return result


def _selection_feasible(attempt: dict) -> bool:
    checks = attempt["checks"]
    non_effect = tuple(name for name in _pilot_required_check_names() if name != "delta_hit32_ge_008")
    return all(checks[key] for key in non_effect)


def _select_attempt(attempts: list[dict]) -> dict:
    feasible = [attempt for attempt in attempts if _selection_feasible(attempt)]
    if not feasible: raise NoFeasiblePilotAttempt("no dev-cycle attempt satisfies non-effect gates")
    return min(feasible, key=lambda item: (-item["delta_hit32"],
                                           item["budget"]["loss_bearing_target_tokens_each"] * item["resolved_config"]["epochs"],
                                            item["resolved_config"]["lr"], item["attempt_id"]))


def _cycle_eligible(attempt: dict) -> bool:
    checks = attempt["checks"]
    return (checks.get("delta_hit32_ge_008") is False and
            all(checks[key] for key in _pilot_required_check_names() if key != "delta_hit32_ge_008"))


def _code_manifest() -> dict:
    paths = sorted((ROOT / "code").glob("*.py"))
    return {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in paths}


def freeze_and_generate_final(selected: dict) -> dict:
    if exploratory_mode():
        raise RuntimeError("exploratory pilot cannot create or open final data")
    if not selected.get("pass"): raise RuntimeError("cannot freeze a failed pilot attempt")
    selection_path = ROOT / "CONFIG_SELECTION_FROZEN.json"
    selection = {"schema": "stage4.composition.v5.selection-freeze.v1", "status": "FROZEN",
                 "model": PROTOCOL["model"], "selected_attempt": selected["attempt_id"],
                 "resolved_config": selected["resolved_config"], "pilot_attempt_sha256": sha256(RUNS / "pilot" / selected["attempt_id"] / "attempt.json"),
                 "protocol_sha256": sha256(PROTOCOL_PATH), "code": _code_manifest(),
                 "adapters": {name: {"path": path, "tree_sha256": tree_sha256(Path(path))}
                              for name, path in selected["adapters"].items()}}
    if selection_path.exists():
        if load_json(selection_path) != selection: raise RuntimeError("existing selection freeze differs")
    else: dump_json_atomic(selection_path, selection)
    frozen_path = ROOT / "CONFIG_FROZEN.json"
    if frozen_path.exists():
        frozen = load_json(frozen_path)
        for name, entry in frozen["test_files"].items():
            path = DATA / name
            if not path.is_file() or sha256(path) != entry["sha256"]: raise RuntimeError(f"frozen test mismatch: {name}")
        receipt_path = MANIFESTS / "final_generation_receipt.json"
        access_path = MANIFESTS / "final_evaluation_access.json"
        frozen_sha = sha256(frozen_path)
        access = load_json(access_path) if access_path.exists() else None
        if access is not None and (access.get("status") not in ("STARTED", "DONE") or
                                   access.get("config_frozen_sha256") != frozen_sha or
                                   access.get("test_files") != frozen["test_files"]):
            raise RuntimeError("existing final access marker is not bound to the frozen tests")
        if not receipt_path.exists():
            count = 1 if access is not None else 0
            dump_json_atomic(receipt_path, {"schema": "stage4.composition.v5.final-generation.v1", "status": "DONE",
                                     "config_frozen_sha256": frozen_sha, "test_files": frozen["test_files"],
                                     "evaluation_count": count, "recovered_after_freeze": True})
        else:
            receipt = load_json(receipt_path)
            if (receipt.get("status") != "DONE" or receipt.get("config_frozen_sha256") != frozen_sha or
                    receipt.get("test_files") != frozen["test_files"] or receipt.get("evaluation_count") not in (0, 1)):
                raise RuntimeError("existing final generation receipt is invalid")
            if receipt.get("evaluation_count") == 1 and access is None:
                raise RuntimeError("final generation receipt says opened but access marker is missing")
        return frozen
    reference = build_atomic_reference()
    pilot_splits = {name: read_jsonl(DATA / f"{name}.jsonl") for name in ("train", "dev_a", "dev_b")}
    cfg = PROTOCOL["future_final_data_template_non_executable"]
    depth3 = {"A": cfg["final_a"], "B": cfg["final_b"], "C": cfg["final_c"], "D": cfg["final_d"]}
    final_splits, final_audit = generate_final_data(reference, pilot_splits, config_frozen=True,
                                                     depth3_sizes=depth3,
                                                     depth2_probe_per_split=cfg["depth2_probe_per_split"],
                                                     depth4_probe_per_split=cfg["depth4_probe_per_split"],
                                                     seed_base=cfg["generation_seed_base"])
    confirm_atomic = generate_confirm_atomic_data(reference, pilot_splits, final_splits,
                                                   per_operation=cfg["confirm_atomic_per_operation"],
                                                   seed=cfg["confirm_atomic_seed"])
    final_splits["confirm_atomic"] = confirm_atomic
    unified_audit = audit_pilot_data({**pilot_splits, **final_splits}, atomic_reference=reference,
                                     verify_shortest=True)
    if not unified_audit.get("ok"): raise RuntimeError(f"unified final split audit failed: {unified_audit.get('errors')}")
    expected_names = {f"{name}.jsonl" for name in final_splits}
    existing_names = {path.name for path in DATA.glob("final_*.jsonl")}
    if (DATA / "confirm_atomic.jsonl").exists(): existing_names.add("confirm_atomic.jsonl")
    extras = existing_names - expected_names
    if extras: raise RuntimeError(f"unexpected unbound final files: {sorted(extras)}")
    paths = []
    for name, rows in sorted(final_splits.items()):
        path = DATA / f"{name}.jsonl"
        if path.exists():
            if read_jsonl(path) != rows: raise RuntimeError(f"partial final recovery mismatch: {path.name}")
        else:
            write_jsonl(path, rows)
        paths.append(path)
    manifest = _data_manifest(paths); manifest_files = _data_manifest_files(manifest)
    dump_json_atomic(MANIFESTS / "final_data_manifest.json", manifest)
    dump_json_atomic(MANIFESTS / "final_split_audit.json", unified_audit)
    frozen = {"schema": "stage4.composition.v5.config-freeze.v1", "status": "FROZEN", "model": PROTOCOL["model"],
              "selection_sha256": sha256(selection_path), "selected_attempt": selected["attempt_id"],
              "resolved_config": selected["resolved_config"], "test_files": manifest_files,
              "confirm_seeds": PROTOCOL["future_confirm_template_non_executable"]["seeds"],
              "primary": PROTOCOL["future_confirm_template_non_executable"]["primary"],
              "protocol_sha256": sha256(PROTOCOL_PATH), "code": _code_manifest(), "final_evaluator_opened": False}
    dump_json_atomic(frozen_path, frozen)
    receipt = {"schema": "stage4.composition.v5.final-generation.v1", "status": "DONE",
               "config_frozen_sha256": sha256(frozen_path), "test_files": manifest_files, "evaluation_count": 0}
    dump_json_atomic(MANIFESTS / "final_generation_receipt.json", receipt); return frozen


def evaluate_selected_pilot_dev_b(selected: dict) -> dict:
    """Descriptive held-motif transfer check; never participates in selection."""
    resolved = selected["resolved_config"]; attempt_id = selected["attempt_id"]
    rows = read_jsonl(DATA / "dev_b.jsonl"); atomic_rows = read_jsonl(ATOMIC_ROOT / "data/calibration.jsonl")
    out_dir = RUNS / "pilot" / attempt_id / "dev_b_descriptive_eval"
    results = {}
    for branch, adapter in (("atomic_base", None),
                            ("atomic_control", Path(selected["adapters"]["atomic_control"])),
                            ("composition_distill", Path(selected["adapters"]["composition_distill"]))):
        summary, _, atomic, binding = _evaluate_branch(ATOMIC_EXPORT, adapter, resolved, attempt_id, branch,
                                                       "dev_b", rows, atomic_rows, out_dir)
        results[branch] = {"summary": summary, "atomic": atomic, "binding": binding}
    receipt = {"schema": "stage4.composition.v5.pilot-dev-b.v1", "status": "DONE",
               "selected_attempt": attempt_id, "used_for_selection": False, "branches": results}
    dump_json_atomic(RUNS / "pilot" / attempt_id / "dev_b_descriptive.json", receipt)
    return receipt


def pilot() -> dict:
    require_prepared(); require_atomic_pass()
    initial_cfg = PROTOCOL["pilot_initial"]
    initial = run_pilot_attempt(_resolved_config(seed=0, lr=initial_cfg["lr"], epochs=initial_cfg["epochs"],
                                                  replay=initial_cfg["atomic_replay"]))
    attempts = {initial["attempt_id"]: initial}; cycle = {"used": False, "stages": {}}
    selected = initial
    if not initial["pass"]:
        if not _cycle_eligible(initial):
            selected = initial
        else:
            cycle["used"] = True
            try:
                lr_attempts = []
                for lr in PROTOCOL["pilot_dev_cycle"]["lr"]:
                    attempt = run_pilot_attempt(_resolved_config(seed=0, lr=lr, epochs=2, replay=0.2))
                    attempts[attempt["attempt_id"]] = attempt; lr_attempts.append(attempt)
                lr_selected = _select_attempt(lr_attempts); cycle["stages"]["lr"] = {"attempts": [a["attempt_id"] for a in lr_attempts], "selected": lr_selected["attempt_id"]}
                epoch_attempts = []
                for epochs in PROTOCOL["pilot_dev_cycle"]["epochs"]:
                    attempt = run_pilot_attempt(_resolved_config(seed=0, lr=lr_selected["resolved_config"]["lr"], epochs=epochs, replay=0.2))
                    attempts[attempt["attempt_id"]] = attempt; epoch_attempts.append(attempt)
                epoch_selected = _select_attempt(epoch_attempts); cycle["stages"]["epochs"] = {"attempts": [a["attempt_id"] for a in epoch_attempts], "selected": epoch_selected["attempt_id"]}
                replay_attempts = []
                for replay in PROTOCOL["pilot_dev_cycle"]["atomic_replay"]:
                    attempt = run_pilot_attempt(_resolved_config(seed=0, lr=epoch_selected["resolved_config"]["lr"],
                                                                  epochs=epoch_selected["resolved_config"]["epochs"], replay=replay))
                    attempts[attempt["attempt_id"]] = attempt; replay_attempts.append(attempt)
                selected = _select_attempt(replay_attempts); cycle["stages"]["replay"] = {"attempts": [a["attempt_id"] for a in replay_attempts], "selected": selected["attempt_id"]}
            except NoFeasiblePilotAttempt as exc:
                cycle["selection_error"] = str(exc); selected = initial
    dev_b = evaluate_selected_pilot_dev_b(selected)
    decision = {"schema": "stage4.composition.v5.exploratory-pilot-decision.v1",
                "status": "EXPLORATORY_PASS" if selected["pass"] else "EXPLORATORY_FAILED_PILOT_GATE",
                "scientific_label": ("EXPLORATORY_COMPOSITION_SIGNAL" if selected["pass"] else
                                     "EXPLORATORY_NEGATIVE"),
                "study_class": "EXPLORATORY_FOLLOWUP", "registered_atomic_pass": False,
                "registered_stage4_result_remains": "FAILED_CALIBRATION_GATE", "model": PROTOCOL["model"],
                "initial_attempt": initial["attempt_id"], "selected_attempt": selected["attempt_id"],
                "selected": selected, "dev_cycle": cycle, "attempts": sorted(attempts),
                "final_created": False, "composition_confirm_unlocked": False,
                "confirmatory_claim_allowed": False,
                "dev_b_descriptive_sha256": sha256(RUNS / "pilot" / selected["attempt_id"] / "dev_b_descriptive.json")}
    dump_json_atomic(RUNS / "PILOT_DECISION.json", decision)
    if not selected["pass"]:
        dump_json_atomic(RUNS / "STAGE4_FAILED.json", {"schema": "stage4.composition.v5.terminal.v1", "status": "FAILED",
                                                "phase": "pilot_gate", "pilot_decision_sha256": sha256(RUNS / "PILOT_DECISION.json"),
                                                "final_created": False, "scientific_result": "EXPLORATORY_NEGATIVE",
                                                "registered_stage4_result_remains": "FAILED_CALIBRATION_GATE"})
    return decision


def require_frozen() -> dict:
    frozen_path = ROOT / "CONFIG_FROZEN.json"
    if not frozen_path.is_file(): raise RuntimeError("final evaluator locked: CONFIG_FROZEN.json is absent")
    frozen = load_json(frozen_path)
    if frozen.get("status") != "FROZEN" or frozen.get("model") != PROTOCOL["model"]:
        raise RuntimeError("invalid frozen configuration")
    if frozen.get("protocol_sha256") != sha256(PROTOCOL_PATH): raise RuntimeError("protocol changed after freeze")
    if frozen.get("selection_sha256") != sha256(ROOT / "CONFIG_SELECTION_FROZEN.json"):
        raise RuntimeError("selection freeze changed")
    current_code = _code_manifest()
    if frozen.get("code") != current_code: raise RuntimeError("code changed after freeze")
    for name, entry in frozen["test_files"].items():
        path = DATA / name
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"] or len(read_jsonl(path)) != entry["rows"]:
            raise RuntimeError(f"frozen test file changed: {name}")
    return frozen


def begin_final_evaluation() -> dict:
    acquire_final_lease()
    frozen = require_frozen(); frozen_sha = sha256(ROOT / "CONFIG_FROZEN.json")
    access_path = MANIFESTS / "final_evaluation_access.json"
    generation_path = MANIFESTS / "final_generation_receipt.json"
    generation = load_json(generation_path)
    if (generation.get("status") != "DONE" or generation.get("config_frozen_sha256") != frozen_sha or
            generation.get("test_files") != frozen["test_files"]):
        raise RuntimeError("final generation receipt is stale or mismatched")
    if access_path.exists():
        access = load_json(access_path)
        if access.get("config_frozen_sha256") != frozen_sha or access.get("status") not in ("STARTED", "DONE"):
            raise RuntimeError("mismatched prior final evaluation access")
        owner_pid = access.get("owner_pid")
        if owner_pid != os.getpid() and pid_alive(owner_pid):
            release_final_lease()
            raise RuntimeError(f"final evaluation is already active in PID {owner_pid}")
        if owner_pid != os.getpid():
            access["owner_pid"] = os.getpid(); access["resume_count"] = int(access.get("resume_count", 0)) + 1
            access["resumed_at_unix"] = time.time(); dump_json_atomic(access_path, access)
        count = generation.get("evaluation_count")
        if count == 0:
            generation["evaluation_count"] = 1; dump_json_atomic(generation_path, generation)
        elif count != 1:
            raise RuntimeError(f"invalid final evaluation count: {count}")
        return access
    if generation.get("evaluation_count") != 0: raise RuntimeError("final evaluation was already opened")
    access = {"schema": "stage4.composition.v5.final-access.v1", "status": "STARTED",
              "config_frozen_sha256": frozen_sha, "test_files": frozen["test_files"],
              "opened_at_unix": time.time(), "series": "seeds_0_to_5", "owner_pid": os.getpid(), "resume_count": 0}
    try:
        dump_json_exclusive(access_path, access)
    except FileExistsError:
        return begin_final_evaluation()
    generation["evaluation_count"] = 1; dump_json_atomic(generation_path, generation)
    return access


def _confirm_seed_training(seed: int, frozen: dict) -> dict:
    selected_attempt = frozen["selected_attempt"]; selection = load_json(ROOT / "CONFIG_SELECTION_FROZEN.json")
    if seed == 0:
        selected = _validate_cached_pilot_attempt(load_json(RUNS / "pilot" / selected_attempt / "attempt.json"))
        adapters = {name: Path(entry["path"]) for name, entry in selection["adapters"].items()}
        for name, path in adapters.items():
            if tree_sha256(path) != selection["adapters"][name]["tree_sha256"]: raise RuntimeError(f"pilot seed0 adapter changed: {name}")
        return {"seed": 0, "adapters": {name: str(path) for name, path in adapters.items()},
                "reused_pilot_attempt": selected_attempt,
                "receipts": {name: load_json(path / "training_receipt.json") for name, path in adapters.items()},
                "budget": selected["budget"], "checks": selected["budget_checks"]}
    from composition_model import (_export_payload_sha, atomic_example_specs, distill_example_specs, encode_specs,
                                   pack_control_to_budget, tokenizer_and_ids, train_branch)
    resolved = deepcopy(frozen["resolved_config"]); resolved["seed"] = seed
    trajectories = read_jsonl(DATA / "discover_trajectories.jsonl"); atomic_pool = _balanced_atomic_pool(seed + 1000)
    tokenizer, _ = tokenizer_and_ids(ATOMIC_EXPORT)
    distill = encode_specs(tokenizer, distill_example_specs(trajectories, atomic_pool, resolved["atomic_replay"], seed))
    control_pool = encode_specs(tokenizer, atomic_example_specs(atomic_pool))
    control, distill, budget = pack_control_to_budget(control_pool, distill, seed)
    expected_steps = math.ceil(math.ceil(len(distill) / resolved["micro_batch"]) /
                               (resolved["effective_batch"] // resolved["micro_batch"])) * resolved["epochs"]
    budget.update({"control_input_sha256": hash_value(control), "distill_input_sha256": hash_value(distill),
                   "effective_batch": resolved["effective_batch"], "epochs": resolved["epochs"],
                   "optimizer_steps_expected": expected_steps})
    seed_dir = RUNS / "confirm" / f"seed{seed}"; seed_dir.mkdir(parents=True, exist_ok=True); dump_json_atomic(seed_dir / "budget.json", budget)
    control_adapter = ADAPTERS / "confirm" / f"seed{seed}" / "atomic_control"
    distill_adapter = ADAPTERS / "confirm" / f"seed{seed}" / "composition_distill"
    control_receipt = train_branch(ATOMIC_EXPORT, control_adapter, control, resolved, "atomic_control", budget["control_input_sha256"])
    distill_receipt = train_branch(ATOMIC_EXPORT, distill_adapter, distill, resolved, "composition_distill", budget["distill_input_sha256"])
    target_expected = budget["loss_bearing_target_tokens_each"] * resolved["epochs"]
    checks = {"optimizer_steps_equal": control_receipt["optimizer_steps"] == distill_receipt["optimizer_steps"],
              "optimizer_steps_expected": control_receipt["optimizer_steps"] == distill_receipt["optimizer_steps"] == expected_steps,
              "target_tokens_equal": control_receipt["loss_bearing_target_tokens_seen"] == distill_receipt["loss_bearing_target_tokens_seen"],
              "target_tokens_expected": control_receipt["loss_bearing_target_tokens_seen"] == distill_receipt["loss_bearing_target_tokens_seen"] == target_expected,
              "effective_batch_frozen": control_receipt["effective_batch"] == distill_receipt["effective_batch"] == resolved["effective_batch"],
              "epochs_frozen": control_receipt["epochs"] == distill_receipt["epochs"] == resolved["epochs"],
              "receipts_done": control_receipt.get("status") == distill_receipt.get("status") == "DONE",
              "receipt_bindings": (control_receipt.get("input_sha256") == budget["control_input_sha256"] and
                                   distill_receipt.get("input_sha256") == budget["distill_input_sha256"] and
                                   str(control_receipt.get("resolved_config_sha256", "")).upper() == hash_value(resolved) and
                                   str(distill_receipt.get("resolved_config_sha256", "")).upper() == hash_value(resolved) and
                                   control_receipt.get("atomic_export_sha256") == distill_receipt.get("atomic_export_sha256") ==
                                   _export_payload_sha(ATOMIC_EXPORT))}
    if not all(checks.values()): raise RuntimeError(f"seed {seed} equal-budget failure: {checks}")
    return {"seed": seed, "resolved_config": resolved,
            "adapters": {"atomic_control": str(control_adapter), "composition_distill": str(distill_adapter)},
            "receipts": {"atomic_control": control_receipt, "composition_distill": distill_receipt},
            "budget": budget, "checks": checks}


def _atomic_base_zero_update_run(seed: int, frozen: dict) -> dict:
    """Record the third confirm branch explicitly, with its registered zero budget."""
    from composition_model import _export_payload_sha

    path = RUNS / "confirm" / f"seed{seed}" / "atomic_base" / "run.json"
    resolved = {**frozen["resolved_config"], "seed": seed}
    record = {
        "schema": "stage4.composition.v5.run.v1",
        "run_id": f"confirm-seed-{seed}-atomic-base",
        "status": "DONE",
        "kind": "frozen_no_update",
        "seed": seed,
        "branch": "atomic_base",
        "phase": "confirm",
        "adapter": None,
        "optimizer_steps": 0,
        "optimizer_steps_expected": 0,
        "loss_bearing_target_tokens_seen": 0,
        "loss_bearing_target_tokens_expected": 0,
        "effective_batch": 0,
        "epochs": 0,
        "resolved_config": resolved,
        "resolved_config_sha256": hash_value(resolved),
        "adapter_tree_sha256": _export_payload_sha(ATOMIC_EXPORT).upper(),
        "binding": {
            "config_frozen_sha256": sha256(ROOT / "CONFIG_FROZEN.json"),
            "atomic_export_sha256": _export_payload_sha(ATOMIC_EXPORT).upper(),
            "protocol_sha256": sha256(PROTOCOL_PATH),
        },
    }
    if path.exists():
        if load_json(path) != record: raise RuntimeError(f"atomic_base zero-update receipt changed for seed {seed}")
    else:
        dump_json_atomic(path, record)
    return record


def _confirm_binding(frozen: dict, seed: int, branch: str, split: str, model_sha: str) -> dict:
    return {"schema": "stage4.composition.v5.eval-binding.v1", "seed": seed, "attempt": "confirm_frozen",
            "branch": branch, "split": split, "model_sha256": model_sha,
            "data_sha256": frozen["test_files"][f"{split}.jsonl"]["sha256"],
            "atomic_data_sha256": frozen["test_files"]["confirm_atomic.jsonl"]["sha256"],
            "resolved_config_sha256": hash_value({**frozen["resolved_config"], "seed": seed}),
            "config_frozen_sha256": sha256(ROOT / "CONFIG_FROZEN.json"),
            "core_sha256": sha256(ROOT / "code/composition_core.py"),
            "evaluator_sha256": sha256(ROOT / "code/composition_eval.py")}


def _evaluate_confirm_model(frozen: dict, seed: int, branch: str, adapter: Path | None,
                            splits: dict[str, list[dict]], atomic_rows: list[dict], out_dir: Path):
    from composition_eval import evaluate_atomic, evaluate_ranking, unload
    from composition_model import _export_payload_sha, load_branch, tree_sha256 as model_tree_sha
    model_sha = _export_payload_sha(ATOMIC_EXPORT).upper() if adapter is None else model_tree_sha(adapter).upper()
    model, tokenizer, token_ids = load_branch(ATOMIC_EXPORT, adapter); summaries = {}; metrics = {}
    try:
        for split, rows in sorted(splits.items()):
            eval_branch = f"seed{seed}_{branch}_{split}"; binding = _confirm_binding(frozen, seed, branch, split, model_sha)
            summary, task_metrics = evaluate_ranking(model, tokenizer, token_ids, rows, out_dir, eval_branch, binding,
                                                     batch_size=PROTOCOL["ranking"]["batch_size"])
            summaries[split] = summary; metrics[split] = task_metrics
        atomic_binding = {"schema": "stage4.composition.v5.eval-binding.v1", "seed": seed, "attempt": "confirm_frozen",
                          "branch": branch, "split": "confirm_atomic", "model_sha256": model_sha,
                          "data_sha256": frozen["test_files"]["confirm_atomic.jsonl"]["sha256"],
                          "config_frozen_sha256": sha256(ROOT / "CONFIG_FROZEN.json"),
                          "core_sha256": sha256(ROOT / "code/composition_core.py"),
                          "evaluator_sha256": sha256(ROOT / "code/composition_eval.py")}
        atomic = evaluate_atomic(model, tokenizer, token_ids, atomic_rows, out_dir, f"seed{seed}_{branch}", atomic_binding)
    finally:
        unload(model)
    return {"summaries": summaries, "metrics": metrics, "atomic": atomic, "model_sha256": model_sha}


def _analysis_rows(seed_results: dict, branch: str, split_names: list[str], merged_split: str | None = None):
    rows = []
    for seed in sorted(seed_results):
        for split in split_names:
            for row in seed_results[seed][branch]["metrics"][split]:
                binding = row.get("binding")
                if (not isinstance(binding, dict) or binding.get("seed") != int(seed) or
                        binding.get("branch") != branch or binding.get("split") != split or
                        row.get("seed") != int(seed) or row.get("split") != split):
                    raise RuntimeError(f"confirm metric identity/binding mismatch for seed={seed}, branch={branch}, split={split}")
                copied = dict(row); copied["seed"] = int(seed)
                if merged_split is not None: copied["split"] = merged_split
                rows.append(copied)
    return rows


def _compact_eval_result(result: dict, eval_root: Path, branch: str) -> dict:
    return {"summaries": result["summaries"], "atomic": result["atomic"],
            "model_sha256": result["model_sha256"],
            "raw_artifacts": {"eval_root": eval_root.relative_to(ROOT).as_posix(), "branch": branch}}


def _validate_raw_confirmation_before_terminal() -> dict:
    """Recompute every reported metric from full raw rankings before DONE."""
    from composition_release import (discover_ranking_files, discover_task_files, load_task_index,
                                     metrics_sibling, receipt_sibling, validate_confirm_ranking_matrix,
                                     validate_ranking_file)

    tasks = load_task_index(discover_task_files(ROOT)); reports = []
    for ranking in discover_ranking_files(ROOT):
        metrics_path = metrics_sibling(ranking); shard_receipt = receipt_sibling(metrics_path)
        report = validate_ranking_file(ranking, tasks, metrics_path, shard_receipt)
        report["path"] = ranking.relative_to(ROOT).as_posix(); reports.append(report)
    matrix = validate_confirm_ranking_matrix(ROOT, reports)
    result = {"schema": "stage4.composition.v5.ranking-validation.v1",
              "ranking_files": [{key: value for key, value in report.items() if not key.startswith("_")} for report in reports],
              "ranking_file_count": len(reports), "ranking_rows": sum(report["rows"] for report in reports),
              "confirm_matrix": matrix}
    dump_json_atomic(MANIFESTS / "preterminal_ranking_validation.json", result)
    return result


def _confirmation_evidence_payload() -> dict:
    selection = load_json(ROOT / "CONFIG_SELECTION_FROZEN.json")
    roots = [RUNS / "confirm", RUNS / "pilot" / selection["selected_attempt"],
             ADAPTERS / "confirm", ATOMIC_EXPORT]
    roots.extend(Path(entry["path"]) for entry in selection["adapters"].values())
    explicit = [ROOT / "CONFIG_FROZEN.json", ROOT / "CONFIG_SELECTION_FROZEN.json",
                RUNS / "PREPARE_DONE.json", RUNS / "PILOT_DECISION.json",
                REPORTS / "confirmatory_statistics.json", MANIFESTS / "preterminal_ranking_validation.json"]
    explicit.extend(MANIFESTS / name for name in (
        "preregistration.json", "atomic_reference.json", "atomic_reference_audit.json", "atomic_import.json",
        "pilot_data_manifest.json", "pilot_split_audit.json", "pilot_generation_audit.json", "discover_audit.json",
        "final_data_manifest.json", "final_split_audit.json", "final_generation_receipt.json"))
    missing_explicit = [path for path in explicit if not path.is_file()]
    if missing_explicit: raise RuntimeError(f"confirmation evidence files are missing: {missing_explicit}")
    files = set(explicit)
    for base in roots:
        if not base.is_dir(): raise RuntimeError(f"confirmation evidence root is missing: {base}")
        files.update(path for path in base.rglob("*") if path.is_file() and
                     not path.name.endswith((".partial", ".tmp")))
    for seed in PROTOCOL["future_confirm_template_non_executable"]["seeds"]:
        run_path = RUNS / "confirm" / f"seed{seed}" / "run.json"
        if not run_path.is_file(): raise RuntimeError(f"confirmation seed run is missing: {run_path}")
        run = load_json(run_path)
        if run.get("status") != "DONE" or run.get("seed") != seed:
            raise RuntimeError(f"confirmation seed run is not DONE/bound: {run_path}")
    entries = {}
    for path in sorted(files):
        try: rel = path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError: raise RuntimeError(f"confirmation evidence escaped Stage 4 root: {path}")
        entries[rel] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return {"schema": "stage4.composition.v5.confirm-evidence.v1", "status": "FROZEN",
            "seeds": PROTOCOL["future_confirm_template_non_executable"]["seeds"], "files": entries}


def _validate_completed_confirmation(done_path: Path) -> dict:
    done = load_json(done_path); stats_path = REPORTS / "confirmatory_statistics.json"
    access_path = MANIFESTS / "final_evaluation_access.json"
    if not stats_path.is_file() or not access_path.is_file(): raise RuntimeError("completed confirmation lacks statistics/access")
    access = load_json(access_path); stats = load_json(stats_path)
    evidence_path = MANIFESTS / "confirm_evidence_manifest.json"
    raw_validation_path = MANIFESTS / "preterminal_ranking_validation.json"
    if not evidence_path.is_file() or not raw_validation_path.is_file():
        raise RuntimeError("completed confirmation lacks raw/evidence validation")
    evidence = load_json(evidence_path); expected_evidence = _confirmation_evidence_payload()
    if evidence != expected_evidence: raise RuntimeError("completed confirmation raw evidence changed")
    expected_result = "POSITIVE_CONFIRMED_A" if stats.get("positive_A") is True else "NO_CONFIRMATORY_EVIDENCE_A"
    terminal_core_ok = (done.get("status") == "DONE" and done.get("statistics_sha256") == sha256(stats_path) and
                        done.get("config_frozen_sha256") == sha256(ROOT / "CONFIG_FROZEN.json") and
                        done.get("evidence_manifest_sha256") == sha256(evidence_path) and
                        done.get("raw_validation_sha256") == sha256(raw_validation_path) and
                        done.get("positive_A") == stats.get("positive_A") and
                        done.get("positive_B") == stats.get("positive_B") and
                        done.get("scientific_result") == expected_result and
                        access.get("config_frozen_sha256") == sha256(ROOT / "CONFIG_FROZEN.json"))
    if terminal_core_ok and access.get("status") == "STARTED":
        access.update({"status": "DONE", "completed_at_unix": time.time(),
                       "terminal_result_sha256": sha256(done_path)})
        dump_json_atomic(access_path, access)
    checks = (done.get("status") == "DONE",
              done.get("statistics_sha256") == sha256(stats_path),
              done.get("evidence_manifest_sha256") == sha256(evidence_path),
              done.get("raw_validation_sha256") == sha256(raw_validation_path),
              done.get("config_frozen_sha256") == sha256(ROOT / "CONFIG_FROZEN.json"),
              access.get("status") == "DONE",
              access.get("terminal_result_sha256") == sha256(done_path),
              access.get("config_frozen_sha256") == sha256(ROOT / "CONFIG_FROZEN.json"))
    if not all(checks): raise RuntimeError("completed confirmation terminal bindings do not validate")
    return done


def _confirm_impl() -> dict:
    frozen = require_frozen()
    done_path = RUNS / "STAGE4_DONE.json"
    if done_path.exists(): return _validate_completed_confirmation(done_path)
    access = begin_final_evaluation()
    from composition_eval import aligned_forgetting
    from composition_stats import analyze_confirmation, holm
    split_names = sorted(name[:-6] for name in frozen["test_files"] if name.endswith(".jsonl") and name != "confirm_atomic.jsonl")
    splits = {name: read_jsonl(DATA / f"{name}.jsonl") for name in split_names}
    atomic_rows = read_jsonl(DATA / "confirm_atomic.jsonl")
    seed_results = {}
    for seed in PROTOCOL["future_confirm_template_non_executable"]["seeds"]:
        seed_dir = RUNS / "confirm" / f"seed{seed}"
        base_receipt = _atomic_base_zero_update_run(seed, frozen)
        base_result = _evaluate_confirm_model(frozen, seed, "atomic_base", None, splits, atomic_rows, seed_dir / "eval")
        training = _confirm_seed_training(seed, frozen)
        training["atomic_base_zero_update"] = base_receipt
        control = _evaluate_confirm_model(frozen, seed, "atomic_control", Path(training["adapters"]["atomic_control"]),
                                          splits, atomic_rows, seed_dir / "eval")
        distill = _evaluate_confirm_model(frozen, seed, "composition_distill", Path(training["adapters"]["composition_distill"]),
                                          splits, atomic_rows, seed_dir / "eval")
        control_forgetting = aligned_forgetting(base_result["atomic"], control["atomic"])
        distill_forgetting = aligned_forgetting(base_result["atomic"], distill["atomic"])
        seed_result = {"schema": "stage4.composition.v5.seed-result.v1", "run_id": f"confirm-seed-{seed}",
                       "status": "DONE", "seed": seed, "training": training, "atomic_base": base_result,
                       "atomic_control": control, "composition_distill": distill,
                       "control_forgetting": control_forgetting, "distill_forgetting": distill_forgetting}
        run_record = {key: value for key, value in seed_result.items()
                      if key not in {"atomic_base", "atomic_control", "composition_distill"}}
        run_record.update({"atomic_base": _compact_eval_result(base_result, seed_dir / "eval", "atomic_base"),
                           "atomic_control": _compact_eval_result(control, seed_dir / "eval", "atomic_control"),
                           "composition_distill": _compact_eval_result(distill, seed_dir / "eval", "composition_distill")})
        dump_json_atomic(seed_dir / "run.json", run_record); seed_results[seed] = seed_result
    raw_validation = _validate_raw_confirmation_before_terminal()
    worst_forgetting = max(max(result["control_forgetting"]["max_drop"], result["distill_forgetting"]["max_drop"])
                           for result in seed_results.values())
    analysis_groups = {
        "final_a_depth3": (["final_a"], None), "final_b_depth3": (["final_b"], None),
        "final_c_depth3": (["final_c"], None), "final_d_depth3": (["final_d"], None),
        "depth2_transfer": ([f"final_{family}_depth2" for family in "abcd"], "depth2_transfer"),
        "depth4_transfer": ([f"final_{family}_depth4" for family in "abcd"], "depth4_transfer"),
    }
    analyses = {}; flat_rows = {}
    for name, (names, merged) in analysis_groups.items():
        control_rows = _analysis_rows(seed_results, "atomic_control", names, merged)
        distill_rows = _analysis_rows(seed_results, "composition_distill", names, merged)
        flat_rows[name] = {"control": control_rows, "distill": distill_rows}
        analyses[name] = analyze_confirmation(control_rows, distill_rows, forgetting=worst_forgetting,
                                              split=merged or names[0], bootstrap_repetitions=PROTOCOL["future_confirm_template_non_executable"]["bootstrap_repetitions"],
                                              bootstrap_seed=76000 + len(analyses))
    secondary_names = PROTOCOL["future_confirm_template_non_executable"]["secondary_holm_family"]
    secondary_p = {name: analyses[name]["seed_statistics"]["p_two_sided"] for name in secondary_names}
    holm_result = holm(secondary_p)
    for name in secondary_names: analyses[name]["holm"] = holm_result[name]
    positive_b_raw = analyses["final_b_depth3"]["criteria"]["pass"]
    positive_b_holm = holm_result["final_b_depth3"] < 0.05
    stats = {"schema": "stage4.composition.v5.confirmatory-statistics.v1", "model": PROTOCOL["model"],
             "worst_aligned_forgetting": worst_forgetting, "analyses": analyses, "secondary_holm": holm_result,
             "positive_A": analyses["final_a_depth3"]["criteria"]["pass"],
             "positive_B_raw": positive_b_raw, "positive_B_holm": positive_b_holm,
             "positive_B": positive_b_raw and positive_b_holm}
    REPORTS.mkdir(parents=True, exist_ok=True); dump_json_atomic(REPORTS / "confirmatory_statistics.json", stats)
    for name, branches in flat_rows.items():
        write_jsonl(RUNS / "confirm/analysis" / f"{name}.atomic_control.jsonl", branches["control"])
        write_jsonl(RUNS / "confirm/analysis" / f"{name}.composition_distill.jsonl", branches["distill"])
    evidence = _confirmation_evidence_payload()
    dump_json_atomic(MANIFESTS / "confirm_evidence_manifest.json", evidence)
    result = {"schema": "stage4.composition.v5.terminal.v1", "run_id": "stage4-confirmatory-series",
              "status": "DONE", "scientific_result": "POSITIVE_CONFIRMED_A" if stats["positive_A"] else "NO_CONFIRMATORY_EVIDENCE_A",
              "positive_A": stats["positive_A"], "positive_B": stats["positive_B"],
              "statistics_sha256": sha256(REPORTS / "confirmatory_statistics.json"),
              "raw_validation_sha256": sha256(MANIFESTS / "preterminal_ranking_validation.json"),
              "evidence_manifest_sha256": sha256(MANIFESTS / "confirm_evidence_manifest.json"),
              "config_frozen_sha256": sha256(ROOT / "CONFIG_FROZEN.json")}
    dump_json_atomic(done_path, result)
    access.update({"status": "DONE", "completed_at_unix": time.time(), "terminal_result_sha256": sha256(done_path)})
    dump_json_atomic(MANIFESTS / "final_evaluation_access.json", access)
    release_final_lease(); return result


def confirm() -> dict:
    if exploratory_mode():
        raise RuntimeError("confirmation is locked for this exploratory pilot protocol")
    try:
        return _confirm_impl()
    finally:
        # OS locks are normally released when the CLI process exits, but the
        # explicit finally also makes library/tests and caught failures safe.
        release_final_lease()


def _accounting() -> dict:
    receipts = []
    for path in sorted(ADAPTERS.glob("**/training_receipt.json")):
        receipt = load_json(path); receipts.append({"path": path.relative_to(ROOT).as_posix(), **receipt})
    atomic_training = []
    for path in sorted((ATOMIC_ROOT / "adapters").glob("*/training_receipt.json")):
        outer = load_json(path); phase = outer.get("phase", outer)
        atomic_training.append({"path": repo_relative(path), "status": outer.get("status"), **phase})
    optimizer_steps = sum(int(row.get("optimizer_steps", 0)) for row in receipts)
    target_tokens = sum(int(row.get("loss_bearing_target_tokens_seen", 0)) for row in receipts)
    training_wall = sum(float(row.get("wall_seconds", 0.0)) for row in receipts)
    peak_gpu = max((int(row.get("peak_gpu_bytes", 0)) for row in receipts), default=0)
    input_tokens = sum(int(row.get("input_tokens_seen", 0)) for row in receipts)
    prompt_tokens = sum(int(row.get("prompt_tokens_seen", 0)) for row in receipts)
    training_forwards = sum(int(row.get("forward_microbatches", 0)) for row in receipts)
    optimizer_steps += sum(int(row.get("optimizer_steps", 0)) for row in atomic_training)
    target_tokens += sum(int(row.get("loss_bearing_target_tokens", 0)) for row in atomic_training)
    training_wall += sum(float(row.get("wall_seconds", 0.0)) for row in atomic_training)
    training_forwards += sum(math.ceil(int(row.get("examples", 0)) / 8) * int(row.get("epochs", 1)) for row in atomic_training)
    metric_rows = 0; candidate_scores = 0
    for path in RUNS.glob("**/metrics/**/part-*.jsonl"):
        rows = read_jsonl(path); metric_rows += len(rows); candidate_scores += sum(5 ** int(row["depth"]) for row in rows)
    atomic_generations = sum(len(read_jsonl(path)) for path in RUNS.glob("**/atomic/*.generations.jsonl"))
    ranking_forwards = ranking_input_tokens = 0; ranking_wall = 0.0
    for path in RUNS.glob("**/summaries/*.json"):
        summary = load_json(path); acc = summary.get("accounting", {})
        ranking_forwards += int(acc.get("model_forward_passes", 0)); ranking_input_tokens += int(acc.get("input_tokens", 0))
        ranking_wall += float(summary.get("wall_seconds", 0.0))
    atomic_forwards = atomic_prompt_tokens = atomic_completion_tokens = 0; atomic_wall = 0.0
    for path in RUNS.glob("**/atomic/*.json"):
        result = load_json(path)
        if result.get("schema") != "stage4.composition.v5.atomic-eval.v1": continue
        plan_acc = result.get("plan", {}).get("accounting", {}); apply_acc = result.get("apply", {}).get("accounting", {})
        atomic_forwards += int(plan_acc.get("model_forward_passes", 0)) + int(apply_acc.get("generation_batches", 0))
        atomic_prompt_tokens += int(plan_acc.get("input_tokens", 0)) + int(apply_acc.get("prompt_tokens", 0))
        atomic_completion_tokens += int(apply_acc.get("decoded_completion_tokens", 0)); atomic_wall += float(result.get("wall_seconds", 0.0))
    atomic_gate_rows = 0; atomic_gate_wall = 0.0
    for path in sorted((ATOMIC_ROOT / "runs").glob("*_gate.json")):
        gate = load_json(path); rows = int(gate.get("generation_rows", 0)); atomic_gate_rows += rows
        atomic_gate_wall += float(gate.get("wall_seconds", 0.0))
    plan_audit_forwards = plan_audit_tokens = plan_audit_generated_slots = 0; plan_audit_wall = 0.0
    plan_audit_path = ATOMIC_ROOT / "manifests/plan_raw_audit.json"
    if plan_audit_path.is_file():
        for row in load_json(plan_audit_path).get("runs", []):
            plan_audit_forwards += int(row.get("model_forward_passes", 0)); plan_audit_tokens += int(row.get("input_tokens", 0))
            plan_audit_generated_slots += int(row.get("generated_token_slots", 0))
            plan_audit_wall += float(row.get("wall_seconds", 0.0))
    return {"schema": "stage4.composition.v5.accounting.v1", "training_runs": len(receipts) + len(atomic_training), "optimizer_steps": optimizer_steps,
            "loss_bearing_target_tokens_seen": target_tokens, "training_wall_seconds": training_wall,
            "peak_gpu_bytes": peak_gpu, "ranking_task_evaluations": metric_rows,
            "candidate_program_scores": candidate_scores, "atomic_generation_evaluations": atomic_generations,
            "input_tokens_seen_composition_training": input_tokens, "prompt_tokens_seen_composition_training": prompt_tokens,
            "model_forward_batches": training_forwards + ranking_forwards + atomic_forwards + 2 * plan_audit_forwards,
            "model_forward_batches_by_phase": {"training": training_forwards, "exact_ranking": ranking_forwards,
                                                "composition_atomic_eval": atomic_forwards, "atomic_v5_gates": plan_audit_forwards,
                                                "atomic_raw_audit": plan_audit_forwards},
            "evaluation_input_tokens": ranking_input_tokens + atomic_prompt_tokens + 2 * plan_audit_tokens,
            "decoded_completion_tokens": atomic_completion_tokens,
            "evaluation_wall_seconds": ranking_wall + atomic_wall + atomic_gate_wall + plan_audit_wall,
            "atomic_v5_gate_rows": atomic_gate_rows, "atomic_evaluation_forward_passes_exact": 2 * plan_audit_forwards,
            "atomic_evaluation_input_tokens_exact": 2 * plan_audit_tokens,
            "atomic_generated_token_slots_exact": 2 * plan_audit_generated_slots,
            "model_forward_batches_are_derived": True,
            "accounting_limitation": "training forward microbatches are derived from the frozen batch schedule; atomic evaluation passes/tokens are exact gate-plus-audit replay counts",
            "receipts": receipts, "atomic_training_receipts": atomic_training}


def _write_confirm_plot(stats: dict) -> None:
    import matplotlib.pyplot as plt
    names = ("final_a_depth3", "final_b_depth3")
    seeds = [str(seed) for seed in PROTOCOL["future_confirm_template_non_executable"]["seeds"]]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, name in zip(axes, names):
        differences = stats["analyses"][name]["seed_differences"]
        values = [100.0 * float(differences[seed]) for seed in seeds]
        axis.bar(seeds, values, color=["#2a9d8f" if value > 0 else "#e76f51" for value in values])
        axis.axhline(0, color="black", linewidth=0.8); axis.axhline(5, color="#264653", linewidth=0.8, linestyle="--")
        axis.set_title(name); axis.set_xlabel("seed"); axis.set_ylabel("Δ Hit@32, п.п.")
    figure.tight_layout(); REPORTS.mkdir(parents=True, exist_ok=True)
    figure.savefig(REPORTS / "confirmatory_seed_deltas.png", dpi=160); plt.close(figure)


def paired_binary_statistics(before: list[bool], after: list[bool], *, repetitions: int = 20000,
                             seed: int = 76444) -> dict:
    """Paired accuracy delta, exact McNemar p, and deterministic task bootstrap CI."""
    if len(before) != len(after) or not before:
        raise RuntimeError("paired binary inputs must be non-empty and aligned")
    if repetitions <= 0:
        raise RuntimeError("bootstrap repetitions must be positive")
    differences = [int(right) - int(left) for left, right in zip(before, after)]
    after_only = sum(not left and right for left, right in zip(before, after))
    before_only = sum(left and not right for left, right in zip(before, after))
    discordant = after_only + before_only
    if discordant:
        tail = sum(math.comb(discordant, k) for k in range(min(after_only, before_only) + 1)) / (2 ** discordant)
        mcnemar_p = min(1.0, 2.0 * tail)
    else:
        mcnemar_p = 1.0
    rng = random.Random(seed); count = len(differences)
    bootstrap = sorted(sum(differences[rng.randrange(count)] for _ in range(count)) / count
                       for _ in range(repetitions))

    def quantile(probability: float) -> float:
        position = (len(bootstrap) - 1) * probability; lower = int(position); upper = min(lower + 1, len(bootstrap) - 1)
        fraction = position - lower
        return bootstrap[lower] * (1.0 - fraction) + bootstrap[upper] * fraction

    return {"schema": "stage4.atomic.v5.paired-statistics.v1", "tasks": count,
            "before_accuracy": sum(before) / count, "after_accuracy": sum(after) / count,
            "delta": sum(differences) / count, "after_only": after_only, "before_only": before_only,
            "exact_mcnemar_p_two_sided": mcnemar_p,
            "paired_bootstrap": {"repetitions": repetitions, "seed": seed,
                                 "ci95": [quantile(0.025), quantile(0.975)]}}


def _atomic_paired_statistics() -> dict:
    before_path = ATOMIC_ROOT / "runs/generations/curriculum_calibration.jsonl"
    after_path = ATOMIC_ROOT / "runs/generations/corrective_calibration.jsonl"
    before_rows = read_jsonl(before_path); after_rows = read_jsonl(after_path)
    before_all = {row["task_id"]: row for row in before_rows}; after_all = {row["task_id"]: row for row in after_rows}
    if len(before_all) != len(before_rows) or len(after_all) != len(after_rows) or set(before_all) != set(after_all):
        raise RuntimeError("atomic APPLY generations are not a strict paired task set")
    initial_gate = load_json(ATOMIC_ROOT / "runs/curriculum_calibration_gate.json")
    corrective_gate = load_json(ATOMIC_ROOT / "runs/corrective_calibration_gate.json")
    by_operation = {}
    for operation in OPS:
        task_ids = sorted(task_id for task_id, row in before_all.items() if row.get("operation") == operation)
        if not task_ids or any(after_all[task_id].get("operation") != operation for task_id in task_ids):
            raise RuntimeError(f"atomic APPLY operation pairing failed: {operation}")
        before_accuracy = sum(bool(before_all[task_id]["correct"]) for task_id in task_ids) / len(task_ids)
        after_accuracy = sum(bool(after_all[task_id]["correct"]) for task_id in task_ids) / len(task_ids)
        if (not math.isclose(before_accuracy, initial_gate["apply"]["by_operation"][operation], abs_tol=1e-15) or
                not math.isclose(after_accuracy, corrective_gate["apply"]["by_operation"][operation], abs_tol=1e-15)):
            raise RuntimeError(f"atomic APPLY raw/gate metric mismatch: {operation}")
        by_operation[operation] = {"tasks": len(task_ids), "before_accuracy": before_accuracy,
                                   "after_accuracy": after_accuracy, "delta": after_accuracy - before_accuracy}
    sh1_ids = sorted(task_id for task_id, row in before_all.items() if row.get("operation") == "SH1")
    task_index = {row["task_id"]: row for row in read_jsonl(ATOMIC_ROOT / "data/calibration.jsonl")}
    slices = {}
    for dimension in ("degree", "p", "state_mode"):
        grouped = {}
        for value in sorted({task_index[task_id][dimension] for task_id in sh1_ids}, key=str):
            ids = [task_id for task_id in sh1_ids if task_index[task_id][dimension] == value]
            before_values = [bool(before_all[task_id]["correct"]) for task_id in ids]
            after_values = [bool(after_all[task_id]["correct"]) for task_id in ids]
            grouped[str(value)] = {"tasks": len(ids), "before_accuracy": sum(before_values) / len(ids),
                                   "after_accuracy": sum(after_values) / len(ids),
                                   "delta": (sum(after_values) - sum(before_values)) / len(ids),
                                   "after_only": sum(not left and right for left, right in zip(before_values, after_values)),
                                   "before_only": sum(left and not right for left, right in zip(before_values, after_values))}
        slices[dimension] = grouped
    result = paired_binary_statistics([bool(before_all[task_id]["correct"]) for task_id in sh1_ids],
                                      [bool(after_all[task_id]["correct"]) for task_id in sh1_ids])
    result.update({"metric": "SH1 APPLY exact accuracy", "before_run": "curriculum_calibration",
                   "after_run": "corrective_calibration", "before_generation_sha256": sha256(before_path),
                   "after_generation_sha256": sha256(after_path), "apply_by_operation": by_operation, "sh1_slices": slices,
                   "all_apply_metrics_rederived_from_raw": True})
    csv_path = REPORTS / "atomic_metrics.csv"; csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(("operation", "tasks", "curriculum_accuracy", "corrective_accuracy", "delta"))
        for operation in OPS:
            values = by_operation[operation]
            writer.writerow((operation, values["tasks"], values["before_accuracy"], values["after_accuracy"], values["delta"]))
    slice_path = REPORTS / "atomic_sh1_slices.csv"
    with slice_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(("dimension", "value", "tasks", "curriculum_accuracy", "corrective_accuracy", "delta", "corrective_only", "curriculum_only"))
        for dimension, groups in slices.items():
            for value, metrics in groups.items():
                writer.writerow((dimension, value, metrics["tasks"], metrics["before_accuracy"], metrics["after_accuracy"],
                                 metrics["delta"], metrics["after_only"], metrics["before_only"]))
    dump_json_atomic(REPORTS / "atomic_statistics.json", result); return result


def _write_atomic_plot(initial: dict, corrective: dict, v4: dict) -> None:
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sh1_values = [100.0 * v4["apply"]["by_operation"]["SH1"],
                  100.0 * initial["apply"]["by_operation"]["SH1"],
                  100.0 * corrective["apply"]["by_operation"]["SH1"]]
    axes[0].bar(("v4 historical", "v5 curriculum", "v5 corrective"), sh1_values,
                color=("#8d99ae", "#457b9d", "#e76f51"))
    axes[0].axhline(90.0, color="#264653", linestyle="--", linewidth=1.0, label="atomic gate 90%")
    axes[0].set_ylim(0, 105); axes[0].set_ylabel("Exact accuracy, %"); axes[0].set_title("SH1 APPLY")
    axes[0].tick_params(axis="x", rotation=15); axes[0].legend(loc="lower right")
    labels = ("PLAN", "SC2", "REV", "AC1", "AX1")
    initial_values = [100.0 * initial["plan"]["overall"],
                      *(100.0 * initial["apply"]["by_operation"][op] for op in labels[1:])]
    corrective_values = [100.0 * corrective["plan"]["overall"],
                         *(100.0 * corrective["apply"]["by_operation"][op] for op in labels[1:])]
    positions = list(range(len(labels))); width = 0.36
    axes[1].bar([position - width / 2 for position in positions], initial_values, width,
                label="curriculum", color="#457b9d")
    axes[1].bar([position + width / 2 for position in positions], corrective_values, width,
                label="corrective", color="#2a9d8f")
    axes[1].set_xticks(positions, labels); axes[1].set_ylim(85, 101); axes[1].set_title("Preserved atomic skills")
    axes[1].legend(loc="lower right"); figure.tight_layout(); REPORTS.mkdir(parents=True, exist_ok=True)
    figure.savefig(REPORTS / "atomic_gate_metrics.png", dpi=160); plt.close(figure)


def report() -> dict:
    if exploratory_mode():
        raise RuntimeError("registered report builder is forbidden for exploratory pilot artifacts")
    ensure_roots(); accounting = _accounting(); dump_json_atomic(REPORTS / "accounting.json", accounting)
    stats_path = REPORTS / "confirmatory_statistics.json"; pilot_path = RUNS / "PILOT_DECISION.json"
    atomic_decision_path = ATOMIC_ROOT / "runs/ATOMIC_DECISION.json"
    lines = ["# Stage 4 Verifier Bottleneck — итоговый отчёт", "", f"Модель: **{PROTOCOL['model']}** (размер зафиксирован).", ""]
    blockers = []
    method_text = "Полный перебор 25/125/625 программ, full-vocabulary log-prob однотокенных действий, exact verifier, paired seed/task statistics и строгий leakage audit."
    if stats_path.exists():
        stats = load_json(stats_path); a = stats["analyses"]["final_a_depth3"]; b = stats["analyses"]["final_b_depth3"]
        outcome = "POSITIVE_CONFIRMED_A" if stats["positive_A"] else "NO_CONFIRMATORY_EVIDENCE_A"
        lines += ["## Научный итог", "", f"Primary final-A: **{outcome}**.",
                  f"Final-B transfer positive: **{stats['positive_B']}**.", "",
                  "| Набор | mean Δ Hit@32 | 95% t-CI | p (t-test) | Holm p | positive seeds | критерий |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for label, analysis in (("A", a), ("B", b)):
            seed_stats = analysis["seed_statistics"]; ci = seed_stats["t_ci95"]
            holm_p = "—" if label == "A" else f"{analysis['holm']:.6g}"
            criterion = analysis["criteria"]["pass"] if label == "A" else stats["positive_B"]
            lines.append(f"| {label} | {100*seed_stats['mean_delta']:.3f} п.п. | [{100*ci[0]:.3f}, {100*ci[1]:.3f}] | {seed_stats['p_two_sided']:.6g} | {holm_p} | {seed_stats['positive_seeds']}/6 | {criterion} |")
        lines += ["", f"Worst aligned atomic forgetting: {100*stats['worst_aligned_forgetting']:.3f} п.п.",
                  "Все значения пересчитываются и сверяются с полными gzip rankings exact verifier.", ""]
        _write_confirm_plot(stats)
        csv_path = REPORTS / "seed_deltas.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle); writer.writerow(["seed", "delta_A_hit32", "delta_B_hit32"])
            for seed in map(str, PROTOCOL["future_confirm_template_non_executable"]["seeds"]):
                writer.writerow([seed, a["seed_differences"][seed], b["seed_differences"][seed]])
        if not stats["positive_A"]: blockers.append("Primary A не прошёл все зарегистрированные статистические критерии.")
    elif pilot_path.exists():
        pilot_decision = load_json(pilot_path); outcome = pilot_decision["status"]
        if outcome == "PASS": raise RuntimeError("pilot PASS is not terminal; confirmatory series/report is required")
        if not (RUNS / "STAGE4_FAILED.json").is_file(): raise RuntimeError("failed pilot lacks terminal failure receipt")
        selected = pilot_decision["selected"]
        lines += ["## Научный итог", "", f"Composition pilot: **{outcome}**.",
                  f"Dev-A Δ Hit@32: {100*selected['delta_hit32']:.3f} п.п.",
                  f"Correct-mass Δ: {selected['correct_mass_delta']:.6g}.",
                  f"Worst atomic forgetting: {100*selected['max_forgetting']:.3f} п.п.",
                  f"Leakage count: {selected['leakage_count']}.", ""]
        if outcome != "PASS": blockers.append("Pilot не прошёл зарегистрированные ворота; final test не создавался.")
    elif atomic_decision_path.exists():
        atomic = load_json(atomic_decision_path); outcome = atomic.get("status", "FAILED_ATOMIC_GATE")
        if not (RUNS / "STAGE4_DONE.json").is_file(): raise RuntimeError("atomic decision is not a released terminal result")
        initial = load_json(ATOMIC_ROOT / "runs/curriculum_calibration_gate.json")
        corrective = atomic["calibration_gate"]
        v4_path = REPO / "artifacts/stage4_sh1_v4/runs/atomic_coordinate_fresh_final_gate.json"
        v4 = load_json(v4_path); paired = _atomic_paired_statistics()
        allocation = load_json(ATOMIC_ROOT / "manifests/corrective_allocation.json")
        _write_atomic_plot(initial, corrective, v4)
        rows = (("PLAN overall", v4["plan"]["overall"], initial["plan"]["overall"], corrective["plan"]["overall"], 0.95),
                *((f"{op} APPLY", v4["apply"]["by_operation"][op], initial["apply"]["by_operation"][op],
                   corrective["apply"]["by_operation"][op], 0.90) for op in OPS))
        lines += ["## Научный итог", "", f"Atomic v5: **{outcome}**.",
                  "Composition pilot не разрешён atomic gate; композиционный перенос **не проверялся** и статистики композиционного эффекта не существует.", "",
                  "## Статус этапов", "", "| Этап | Статус |",
                  "|---|---|", "| Atomic curriculum + corrective | Выполнен; gate не пройден |",
                  "| Untouched atomic final | Не открыт |", "| Discover A/B/C/D | Не запускался |",
                  "| Composition pilot | Не запускался |", "| Seeds 0–5 / exact rankings / leakage / composition statistics | Не запускались; N/A |", "",
                  "## Atomic gate", "", "| Метрика | v4 historical* | v5 curriculum | v5 corrective | Порог | Итог |",
                  "|---|---:|---:|---:|---:|---:|"]
        for label, old, before, after, threshold in rows:
            lines.append(f"| {label} | {100*old:.3f}% | {100*before:.3f}% | {100*after:.3f}% | ≥{100*threshold:.1f}% | {'PASS' if after >= threshold else 'FAIL'} |")
        spread = max(corrective["apply"]["by_operation"].values()) - min(corrective["apply"]["by_operation"].values())
        ci = paired["paired_bootstrap"]["ci95"]
        unique_states = sum(allocation["unique_state_count"].values())
        repeated_states = sum(allocation["reused_after_unique_search_exhaustion_count"].values())
        lines += ["", "*v4 использует исторический fresh-final split и приведён только как непарный ориентир; строгая парная оценка ниже сравнивает два v5 checkpoint на одних и тех же задачах.*", "",
                  f"SH1 APPLY: +{100*(corrective['apply']['by_operation']['SH1'] - v4['apply']['by_operation']['SH1']):.3f} п.п. к историческому v4, но только +{100*paired['delta']:.3f} п.п. к curriculum на тех же 360 задачах.",
                  f"Paired bootstrap 95% CI: [{100*ci[0]:.3f}, {100*ci[1]:.3f}] п.п.; exact McNemar p={paired['exact_mcnemar_p_two_sided']:.6g}; corrective-only={paired['after_only']}, curriculum-only={paired['before_only']}.",
                  f"APPLY spread после corrective: {100*spread:.3f} п.п. при пороге ≤10 п.п.; control forgetting gate: {'PASS' if corrective['checks']['control_forgetting_le_002'] else 'FAIL'}.", "",
                  "### SH1 по степени", "", "| Degree | Задач | Curriculum | Corrective | Δ |",
                  "|---:|---:|---:|---:|---:|"]
        for degree, metrics in paired["sh1_slices"]["degree"].items():
            lines.append(f"| {degree} | {metrics['tasks']} | {100*metrics['before_accuracy']:.3f}% | {100*metrics['after_accuracy']:.3f}% | {100*metrics['delta']:+.3f} п.п. |")
        lines += ["",
                  "## Corrective и целостность", "",
                  f"- 40 000 SH1 full-vector, 10 000 control APPLY и 5 000 PLAN; {unique_states} уникальных SH1 states и {repeated_states} повторов после исчерпания bounded unique search.",
                  f"- Evaluation-state overlap: {allocation['evaluation_state_overlap']}; task IDs unique: {allocation['task_ids_unique']}.",
                  "- Corrective: 860 optimizer steps, один зарегистрированный цикл; других циклов не выполнялось.",
                  "- Untouched final не открывался; final gate отсутствует по протоколу.", ""]
        method_text = "Atomic APPLY полностью пересчитан из raw completions; PLAN повторно записан построчно отдельным hash-bound audit replay и сверен с frozen gate. Изменение SH1 между curriculum и corrective оценено парно по raw generations (20 000 bootstrap повторов и exact McNemar). Перебор композиционных программ, Discover, pilot и seeds 0–5 не запускались, потому что atomic gate не пройден."
        blockers.append("Atomic gate исчерпал зарегистрированный 0.6B protocol без PASS.")
    else:
        raise RuntimeError("no terminal scientific decision is available")
    lines += ["## Фактический бюджет", "",
              f"- Training runs: {accounting['training_runs']}",
              f"- Optimizer steps: {accounting['optimizer_steps']}",
              f"- Loss-bearing target tokens: {accounting['loss_bearing_target_tokens_seen']}",
              f"- Training/evaluation model batches or passes: {accounting['model_forward_batches']}",
              f"- Atomic evaluation forward passes (gate + audit replay, exact): {accounting['atomic_evaluation_forward_passes_exact']}",
              f"- Evaluation input tokens: {accounting['evaluation_input_tokens']}",
              f"- Atomic generated token slots (gate + audit replay, exact): {accounting['atomic_generated_token_slots_exact']}",
              f"- Composition decoded completion tokens: {accounting['decoded_completion_tokens']}",
              f"- Training wall time: {accounting['training_wall_seconds']:.1f} s",
              f"- Evaluation wall time: {accounting['evaluation_wall_seconds']:.1f} s",
              f"- Ranking task evaluations: {accounting['ranking_task_evaluations']}",
              f"- Candidate program scores: {accounting['candidate_program_scores']}", "",
              f"Accounting limitation: {accounting['accounting_limitation']}.", "",
              "## Метод", "", method_text, ""]
    report_path = REPORTS / "FINAL_REPORT.md"; report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    status_text = f"# STATUS\n\nScientific outcome: **{outcome}**\n\nModel: Qwen3-0.6B only.\n"
    (ROOT / "STATUS.md").write_text(status_text, encoding="utf-8")
    (ROOT / "BLOCKERS.md").write_text("# BLOCKERS\n\n" + ("\n".join(f"- {item}" for item in blockers) if blockers else "Нет незакрытых экспериментальных блокеров.") + "\n", encoding="utf-8")
    delivery = "# DELIVERY INDEX\n\n- `reports/FINAL_REPORT.md` — научный отчёт\n- `reports/accounting.json` — фактический бюджет\n- `reports/atomic_statistics.json` — парная статистика atomic corrective\n- `reports/atomic_metrics.csv` — APPLY по операциям из raw\n- `reports/atomic_sh1_slices.csv` — SH1 по degree/field/mode\n- `reports/atomic_gate_metrics.png` — график atomic gate\n- `manifests/VALIDATION.json` — итоговая валидация\n- `manifests/RUN_INDEX.json` — единый индекс DONE/FAILED/superseded\n- `manifests/archive_receipt.json` и `archives/RELEASE.json` — SHA/CRC receipts\n- `archives/stage4_distill_v5_0p6b_audit_report.zip` — компактный самодостаточный аудит\n- `archives/stage4_distill_v5_0p6b_complete.zip` — полный выпуск\n"
    (ROOT / "DELIVERY_INDEX.md").write_text(delivery, encoding="utf-8")
    terminal_dir = RUNS / "terminal"; terminal_dir.mkdir(parents=True, exist_ok=True)
    terminal = {"schema": "stage4.composition.v5.run.v1", "run_id": "stage4-terminal-release", "status": "DONE",
                "scientific_outcome": outcome, "report": "reports/FINAL_REPORT.md", "accounting": accounting}
    dump_json_atomic(terminal_dir / "run.json", terminal)
    return terminal


def _validate_atomic_gate(path: Path, adapter: Path, split: str, run_id: str) -> dict:
    gate = load_json(path); data_manifest = load_json(ATOMIC_ROOT / "manifests/data_manifest.json")
    binding = gate.get("binding", {})
    scalar_ok = (binding.get("schema") == "stage4.sh1.v5.gate-binding.v1" and binding.get("run_id") == run_id and
                 binding.get("split") == split and binding.get("model") == PROTOCOL["model"])
    hashes_ok = (str(binding.get("protocol_sha256", "")).upper() == sha256(ATOMIC_ROOT / "configs/protocol.json") and
                 str(binding.get("dataset_sha256", "")).upper() == str(data_manifest[f"{split}.jsonl"]["sha256"]).upper() and
                 str(binding.get("adapter_tree_sha256", "")).upper() == tree_sha256(adapter) and
                 str(binding.get("evaluator_sha256", "")).upper() == sha256(ATOMIC_ROOT / "code/v5_train.py"))
    generation = ATOMIC_ROOT / "runs/generations" / f"{run_id}.jsonl"
    if (not scalar_ok or not hashes_ok or not generation.is_file() or str(gate.get("generation_sha256", "")).upper() != sha256(generation) or
            gate.get("generation_rows") != len(read_jsonl(generation))):
        raise RuntimeError(f"atomic gate binding/generations invalid: {path}")
    tasks_list = read_jsonl(ATOMIC_ROOT / "data" / f"{split}.jsonl"); generations = read_jsonl(generation)
    tasks = {row["task_id"]: row for row in tasks_list}; generation_ids = [row.get("task_id") for row in generations]
    if len(tasks) != len(tasks_list) or len(generation_ids) != len(set(generation_ids)) or set(generation_ids) != set(tasks):
        raise RuntimeError(f"atomic APPLY raw task inventory invalid: {path}")
    correct = 0; by_operation = {operation: [0, 0] for operation in OPS}
    for raw in generations:
        task = tasks[raw["task_id"]]; operation = task["witness"][0]; expected_target = str(list(task["target"]))
        exact = str(raw.get("raw_completion", "")).strip() == expected_target
        if (raw.get("schema") != "stage4.atomic-generation.v1" or raw.get("operation") != operation or
                raw.get("target") != expected_target or raw.get("correct") is not exact):
            raise RuntimeError(f"atomic APPLY raw row inconsistent: {path} {raw.get('task_id')}")
        correct += int(exact); by_operation[operation][0] += int(exact); by_operation[operation][1] += 1
    apply = {"overall": correct / len(generations), "by_operation": {
             operation: by_operation[operation][0] / by_operation[operation][1] for operation in OPS}}
    if (not math.isclose(apply["overall"], gate["apply"]["overall"], abs_tol=1e-15) or
            any(not math.isclose(apply["by_operation"][operation], gate["apply"]["by_operation"][operation], abs_tol=1e-15)
                for operation in OPS)):
        raise RuntimeError(f"atomic APPLY raw/gate metric mismatch: {path}")
    atomic_protocol = load_json(ATOMIC_ROOT / "configs/protocol.json"); values = list(apply["by_operation"].values())
    expected_checks = {"apply_each_ge_090": min(values) >= atomic_protocol["gates"]["apply_each"],
                       "spread_le_010": max(values) - min(values) <= atomic_protocol["gates"]["spread"] + 1e-12}
    if any(gate.get("checks", {}).get(name) is not value for name, value in expected_checks.items()):
        raise RuntimeError(f"atomic APPLY gate checks are not raw-derived: {path}")
    if "control_checks" in gate:
        expected_controls = {operation: apply["by_operation"][operation] >= atomic_protocol["v4_control_baseline"][operation] - atomic_protocol["gates"]["control_drop"]
                             for operation in OPS[1:]}
        if gate["control_checks"] != expected_controls or gate.get("checks", {}).get("control_forgetting_le_002") is not all(expected_controls.values()):
            raise RuntimeError(f"atomic control forgetting checks are not raw-derived: {path}")
    return gate


def _atomic_corrective_allowed(gate: dict) -> bool:
    atomic_protocol = load_json(ATOMIC_ROOT / "configs/protocol.json")
    return (gate["plan"]["overall"] >= atomic_protocol["gates"]["plan"] and
            all(gate["control_checks"].values()) and
            gate["apply"]["by_operation"]["SH1"] < atomic_protocol["gates"]["apply_each"])


def _validate_atomic_phase_seal(parent: Path, output: Path, phase: str, spec_name: str, examples: int) -> list[Path]:
    atomic_protocol = load_json(ATOMIC_ROOT / "configs/protocol.json"); spec = atomic_protocol[spec_name]
    receipt_path = output / "training_receipt.json"; seal_path = ATOMIC_ROOT / "manifests/phase_seals" / f"{phase}.json"
    if not receipt_path.is_file() or not seal_path.is_file(): raise RuntimeError(f"atomic phase is not sealed: {phase}")
    receipt = load_json(receipt_path); inner = receipt.get("phase", {}); seal = load_json(seal_path)
    input_names = {"coordinate": ("coordinate_source.jsonl",), "prefix": ("coordinate_source.jsonl",),
                   "full": ("full_sh1.jsonl", "control_apply.jsonl"),
                   "corrective": ("corrective_sh1.jsonl", "corrective_control.jsonl")}[phase]
    expected_inputs = {name: sha256(ATOMIC_ROOT / "data" / name) for name in input_names}
    structural = (receipt.get("status") == "DONE" and receipt.get("parent") == str(parent) and
                  inner.get("status") == "DONE" and inner.get("phase") == phase and inner.get("examples") == examples and
                  inner.get("epochs") == spec["epochs"] and inner.get("lr") == spec["lr"] and
                  inner.get("effective_batch") == spec["effective_batch"] and seal.get("phase") == phase and
                  seal.get("model") == PROTOCOL["model"] and seal.get("examples") == examples and seal.get("spec") == spec and
                  bool(seal.get("training_launch_git_commit")))
    hashes = (str(seal.get("protocol_sha256", "")).upper() == sha256(ATOMIC_ROOT / "configs/protocol.json") and
              {name: str(value).upper() for name, value in seal.get("input_sha256", {}).items()} == expected_inputs and
              str(seal.get("parent_tree_sha256", "")).upper() == tree_sha256(parent) and
              str(seal.get("output_tree_sha256", "")).upper() == tree_sha256(output) and
              str(seal.get("receipt_sha256", "")).upper() == sha256(receipt_path))
    if not structural or not hashes: raise RuntimeError(f"atomic phase seal binding mismatch: {phase}")
    return [receipt_path, seal_path, *(ATOMIC_ROOT / "data" / name for name in input_names)]


def _validate_atomic_plan_raw_audit(expected: list[tuple[str, Path, dict]]) -> list[Path]:
    receipt_path = ATOMIC_ROOT / "manifests/plan_raw_audit.json"
    code_path = ATOMIC_ROOT / "code/plan_raw_audit.py"
    if not receipt_path.is_file() or not code_path.is_file():
        raise RuntimeError("atomic PLAN raw audit is missing")
    receipt = load_json(receipt_path)
    if (receipt.get("schema") != "stage4.sh1.v5.plan-raw-audit-receipt.v1" or receipt.get("status") != "DONE" or
            receipt.get("model") != PROTOCOL["model"] or receipt.get("final_dataset_accessed") is not False or
            str(receipt.get("audit_code_sha256", "")).upper() != sha256(code_path)):
        raise RuntimeError("atomic PLAN raw audit receipt is invalid")
    rows_by_id = {row.get("run_id"): row for row in receipt.get("runs", [])}
    if len(rows_by_id) != len(receipt.get("runs", [])) or set(rows_by_id) != {item[0] for item in expected}:
        raise RuntimeError("atomic PLAN raw audit run set is invalid")
    calibration_tasks = {row["task_id"]: row for row in read_jsonl(ATOMIC_ROOT / "data/calibration.jsonl")}
    evidence = [receipt_path, code_path]
    for run_id, adapter, gate in expected:
        audit = rows_by_id[run_id]; raw_path = ATOMIC_ROOT / "runs/audit/plan" / f"{run_id}.jsonl"
        gate_path = ATOMIC_ROOT / "runs" / f"{run_id}_gate.json"
        path_bindings = (audit.get("adapter") == repo_relative(adapter) and
                         audit.get("gate_path") == repo_relative(gate_path) and
                         audit.get("raw_path") == repo_relative(raw_path))
        hash_bindings = (str(audit.get("adapter_tree_sha256", "")).upper() == tree_sha256(adapter) and
                         str(audit.get("gate_sha256", "")).upper() == sha256(gate_path) and
                         str(audit.get("dataset_sha256", "")).upper() == sha256(ATOMIC_ROOT / "data/calibration.jsonl") and
                         raw_path.is_file() and str(audit.get("raw_sha256", "")).upper() == sha256(raw_path))
        records = read_jsonl(raw_path) if raw_path.is_file() else []
        task_ids = [record.get("task_id") for record in records]
        if (not path_bindings or not hash_bindings or audit.get("rows") != len(records) or len(task_ids) != len(set(task_ids)) or
                set(task_ids) != set(calibration_tasks)):
            raise RuntimeError(f"atomic PLAN raw audit binding mismatch: {run_id}")
        correct = 0; by_operation = {operation: [0, 0] for operation in OPS}
        for record in records:
            task = calibration_tasks[record["task_id"]]; target = task["witness"][0]
            logits = record.get("candidate_logits", {}); token_ids = record.get("candidate_token_ids", {})
            if (record.get("schema") != "stage4.sh1.v5.plan-raw-audit.v1" or set(logits) != set(OPS) or set(token_ids) != set(OPS) or
                    any(not math.isfinite(float(logits[operation])) for operation in OPS)):
                raise RuntimeError(f"atomic PLAN raw row malformed: {run_id}")
            predicted = max(OPS, key=lambda operation: float(logits[operation])); ok = predicted == target
            if (record.get("target_operation") != target or record.get("predicted_operation") != predicted or record.get("correct") is not ok):
                raise RuntimeError(f"atomic PLAN raw row inconsistent: {run_id}")
            correct += int(ok); by_operation[target][0] += int(ok); by_operation[target][1] += 1
        metrics = {"overall": correct / len(records),
                   "by_operation": {operation: by_operation[operation][0] / by_operation[operation][1] for operation in OPS}}
        if (not math.isclose(metrics["overall"], gate["plan"]["overall"], abs_tol=1e-15) or
                any(not math.isclose(metrics["by_operation"][operation], gate["plan"]["by_operation"][operation], abs_tol=1e-15)
                    for operation in OPS) or audit.get("metrics") != metrics or audit.get("matches_registered_gate") is not True):
            raise RuntimeError(f"atomic PLAN raw/gate metric mismatch: {run_id}")
        apply_audit = audit.get("apply_token_audit", {}); apply_path = ATOMIC_ROOT / "runs/audit/apply" / f"{run_id}.jsonl"
        gate_raw_path = ATOMIC_ROOT / "runs/generations" / f"{run_id}.jsonl"
        apply_records = read_jsonl(apply_path) if apply_path.is_file() else []
        gate_raw = {row["task_id"]: row for row in read_jsonl(gate_raw_path)}
        apply_ids = [row.get("task_id") for row in apply_records]
        if (apply_audit.get("raw_path") != repo_relative(apply_path) or not apply_path.is_file() or
                str(apply_audit.get("raw_sha256", "")).upper() != sha256(apply_path) or
                str(apply_audit.get("gate_raw_sha256", "")).upper() != sha256(gate_raw_path) or
                apply_audit.get("rows") != len(apply_records) or len(apply_ids) != len(set(apply_ids)) or
                set(apply_ids) != set(calibration_tasks) or apply_audit.get("matches_gate_raw") is not True):
            raise RuntimeError(f"atomic APPLY token audit binding mismatch: {run_id}")
        token_slots = 0
        for record in apply_records:
            prior = gate_raw[record["task_id"]]; task = calibration_tasks[record["task_id"]]
            token_ids = record.get("generated_token_ids")
            if (record.get("schema") != "stage4.sh1.v5.apply-token-audit.v1" or record.get("operation") != task["witness"][0] or
                    record.get("raw_completion") != prior.get("raw_completion") or record.get("matches_gate_raw") is not True or
                    not isinstance(token_ids, list) or any(type(value) is not int or value < 0 for value in token_ids) or
                    record.get("generated_token_slots") != len(token_ids)):
                raise RuntimeError(f"atomic APPLY token audit row inconsistent: {run_id}")
            token_slots += len(token_ids)
        expected_plan_forwards = math.ceil(len(records) / 16)
        expected_generation_batches = math.ceil(len(apply_records) / 8)
        if (apply_audit.get("generated_token_slots") != token_slots or
                apply_audit.get("generation_batches") != expected_generation_batches or
                audit.get("plan_forward_passes") != expected_plan_forwards or
                audit.get("generated_token_slots") != token_slots or
                audit.get("model_forward_passes") != expected_plan_forwards + apply_audit.get("generation_forward_passes", -1) or
                not isinstance(audit.get("input_tokens"), int) or audit["input_tokens"] <= 0):
            raise RuntimeError(f"atomic token accounting audit mismatch: {run_id}")
        evidence.extend((raw_path, apply_path))
    return evidence


def validate_atomic_terminal_negative(decision: dict) -> list[Path]:
    if decision.get("status") not in {"FAILED_CALIBRATION_GATE", "FAILED_FINAL_GATE"}:
        raise RuntimeError(f"unregistered atomic terminal status: {decision.get('status')}")
    evidence = [ATOMIC_ROOT / "configs/protocol.json", ATOMIC_ROOT / "manifests/data_manifest.json",
                ATOMIC_ROOT / "runs/ATOMIC_DECISION.json"]
    atomic_protocol = load_json(ATOMIC_ROOT / "configs/protocol.json")
    parent = REPO / atomic_protocol["parent_adapter"]
    phase_specs = ((parent, ATOMIC_ROOT / "adapters/coordinate", "coordinate", "phase_coordinate", 104000),
                   (ATOMIC_ROOT / "adapters/coordinate", ATOMIC_ROOT / "adapters/prefix", "prefix", "phase_prefix", 80000),
                   (ATOMIC_ROOT / "adapters/prefix", ATOMIC_ROOT / "adapters/curriculum", "full", "phase_full", 90000))
    for phase_args in phase_specs: evidence.extend(_validate_atomic_phase_seal(*phase_args))
    initial_path = ATOMIC_ROOT / "runs/curriculum_calibration_gate.json"
    initial = _validate_atomic_gate(initial_path, ATOMIC_ROOT / "adapters/curriculum", "calibration", "curriculum_calibration")
    evidence.extend((initial_path, ATOMIC_ROOT / "runs/generations/curriculum_calibration.jsonl"))
    selected_gate = initial; plan_audit_expected = [("curriculum_calibration", ATOMIC_ROOT / "adapters/curriculum", initial)]
    if not initial.get("pass") and _atomic_corrective_allowed(initial):
        corrective_receipt = ATOMIC_ROOT / "adapters/corrective/training_receipt.json"
        corrective_path = ATOMIC_ROOT / "runs/corrective_calibration_gate.json"
        if not corrective_receipt.is_file(): raise RuntimeError("authorized atomic corrective was not completed")
        evidence.extend(_validate_atomic_phase_seal(ATOMIC_ROOT / "adapters/curriculum", ATOMIC_ROOT / "adapters/corrective",
                                                    "corrective", "phase_corrective", 55000))
        selected_gate = _validate_atomic_gate(corrective_path, ATOMIC_ROOT / "adapters/corrective", "calibration", "corrective_calibration")
        evidence.extend((corrective_receipt, corrective_path, ATOMIC_ROOT / "runs/generations/corrective_calibration.jsonl"))
        plan_audit_expected.append(("corrective_calibration", ATOMIC_ROOT / "adapters/corrective", selected_gate))
    evidence.extend(_validate_atomic_plan_raw_audit(plan_audit_expected))
    if decision.get("calibration_gate") != selected_gate:
        raise RuntimeError("atomic decision does not contain the exhausted calibration gate")
    if decision["status"] == "FAILED_CALIBRATION_GATE":
        if selected_gate.get("pass") or decision.get("final_gate") is not None:
            raise RuntimeError("invalid FAILED_CALIBRATION_GATE decision")
    else:
        if not selected_gate.get("pass"): raise RuntimeError("final gate was opened without calibration PASS")
        adapter_name = "corrective" if (ATOMIC_ROOT / "adapters/corrective").is_dir() and decision.get("selected_adapter", "").endswith("corrective") else "curriculum"
        final_path = ATOMIC_ROOT / "runs/atomic_v5_final_gate.json"
        final_gate = _validate_atomic_gate(final_path, ATOMIC_ROOT / "adapters" / adapter_name, "final", "atomic_v5_final")
        access = ATOMIC_ROOT / "manifests/final_access_receipt.json"
        access_record = load_json(access) if access.is_file() else {}
        final_generation = ATOMIC_ROOT / "runs/generations/atomic_v5_final.jsonl"
        access_ok = (access_record.get("status") == "DONE" and
                     str(access_record.get("gate_sha256", "")).upper() == sha256(final_path) and
                     str(access_record.get("generation_sha256", "")).upper() == sha256(final_generation))
        if final_gate.get("pass") or decision.get("final_gate") != final_gate or not access_ok:
            raise RuntimeError("invalid FAILED_FINAL_GATE evidence")
        evidence.extend((final_path, final_generation, access))
    return evidence


def finalize_atomic_negative() -> dict:
    ensure_roots(); decision_path = ATOMIC_ROOT / "runs/ATOMIC_DECISION.json"
    if not decision_path.exists(): raise RuntimeError("atomic decision is not terminal yet")
    decision = load_json(decision_path)
    if decision.get("status") == "PASS": raise RuntimeError("atomic PASS cannot be released as atomic-negative")
    candidates = validate_atomic_terminal_negative(decision)
    atomic_reference_manifest(candidates)
    failure = {"schema": "stage4.composition.v5.terminal.v1", "run_id": "stage4-atomic-negative",
               "status": "DONE", "scientific_result": "FAILED_ATOMIC_GATE", "composition_started": False,
               "atomic_decision_sha256": sha256(decision_path), "final_composition_created": False}
    dump_json_atomic(RUNS / "STAGE4_DONE.json", failure); report(); return failure


def archive() -> dict:
    if exploratory_mode():
        raise RuntimeError("registered release builder is forbidden for exploratory pilot artifacts")
    report()
    from composition_release import build_release
    return build_release(ROOT, REPO)


def run() -> dict:
    if exploratory_mode():
        raise RuntimeError("full registered run is forbidden; use import-atomic, prepare, then pilot")
    protocol_guard(); ensure_roots(); atomic_decision = ATOMIC_ROOT / "runs/ATOMIC_DECISION.json"
    if not atomic_decision.exists(): raise RuntimeError("atomic v5 is still running; no terminal decision yet")
    if load_json(atomic_decision).get("status") != "PASS":
        result = finalize_atomic_negative(); archive(); return result
    import_atomic()
    if not (RUNS / "PREPARE_DONE.json").exists(): prepare()
    pilot_decision = load_json(RUNS / "PILOT_DECISION.json") if (RUNS / "PILOT_DECISION.json").exists() else pilot()
    if pilot_decision.get("status") == "PASS" and not (ROOT / "CONFIG_FROZEN.json").exists():
        selected = _validate_cached_pilot_attempt(pilot_decision["selected"])
        freeze_and_generate_final(selected)
        pilot_decision["final_created"] = True
        pilot_decision["config_frozen_sha256"] = sha256(ROOT / "CONFIG_FROZEN.json")
        dump_json_atomic(RUNS / "PILOT_DECISION.json", pilot_decision)
    if pilot_decision.get("status") == "PASS": result = confirm()
    else:
        failed_path = RUNS / "STAGE4_FAILED.json"
        if not failed_path.exists():
            dump_json_atomic(failed_path, {"schema": "stage4.composition.v5.terminal.v1", "status": "FAILED",
                                    "phase": "pilot_gate", "pilot_decision_sha256": sha256(RUNS / "PILOT_DECISION.json"),
                                    "final_created": False, "scientific_result": "no_confirmatory_evidence"})
        result = load_json(failed_path)
    report(); archive(); return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("import-atomic", "prepare", "pilot"))
    command = parser.parse_args().command
    functions = {"import-atomic": import_atomic, "prepare": prepare, "pilot": pilot}
    print(json.dumps(functions[command](), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__": main()
