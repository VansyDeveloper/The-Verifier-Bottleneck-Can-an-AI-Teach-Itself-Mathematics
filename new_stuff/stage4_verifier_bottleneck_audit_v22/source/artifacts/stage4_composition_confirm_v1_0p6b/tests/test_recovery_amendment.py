import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import confirm
import confirm_atomic_v11


def _isolated_recovery_runtime(tmp_path, monkeypatch):
    root = tmp_path
    paths = {
        "ROOT": root,
        "REPO": root,
        "DATA": root / "data",
        "RUNS": root / "runs",
        "RANKINGS": root / "rankings",
        "MANIFESTS": root / "manifests",
        "STATISTICS": root / "statistics",
        "REPORTS": root / "reports",
        "ARCHIVES": root / "archives",
        "AMENDMENT_SPEC_PATH": root / "RECOVERY_AMENDMENT_001.json",
        "AMENDMENT_RECEIPT_PATH": root / "manifests/recovery_amendment_001_receipt.json",
        "BLINDNESS_SNAPSHOT_PATH": root / "manifests/recovery_blindness_snapshot_001.json",
        "RECOVERY_FREEZE_STARTED_PATH": root / "runs/recovery/freeze_attempt1_STARTED.json",
        "RECOVERY_FREEZE_DONE_PATH": root / "runs/recovery/freeze_attempt1_DONE.json",
    }
    for name, path in paths.items():
        monkeypatch.setattr(confirm, name, path)
    confirm.ensure_roots()
    for path in (
        paths["AMENDMENT_SPEC_PATH"],
        paths["AMENDMENT_RECEIPT_PATH"],
        paths["BLINDNESS_SNAPSHOT_PATH"],
        root / "CONFIG_SELECTION_FROZEN.json",
        paths["RUNS"] / "TRAINING_DONE.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}\n', encoding="utf-8")
    amendment = {
        "git_commit": "a" * 40,
        "scope": {"maximum_freeze_retries_under_amendment": 1},
        "original_bindings": {
            "preregistration_sha256": "B" * 64,
            "failed_sha256": "C" * 64,
        },
    }
    monkeypatch.setattr(confirm, "require_recovery_amendment", lambda: amendment)
    monkeypatch.setattr(confirm, "code_manifest", lambda: {"code/confirm.py": {"bytes": 1, "sha256": "A" * 64}})
    monkeypatch.setattr(confirm, "_git_head", lambda: amendment["git_commit"])
    return paths, amendment


def _patch_tiny_freeze_pipeline(monkeypatch):
    monkeypatch.setattr(confirm, "require_selection", lambda: {"status": "FROZEN"})
    monkeypatch.setattr(confirm, "require_training", lambda: {"status": "DONE", "all_six_complete": True})
    monkeypatch.setattr(confirm, "atomic_reference", lambda: object())
    monkeypatch.setattr(confirm, "pilot_splits", lambda: {})
    monkeypatch.setattr(
        confirm.core,
        "generate_final_data",
        lambda *args, **kwargs: ({"final_a": [{"task_id": "composition"}]}, {"ok": True, "errors": []}),
    )
    monkeypatch.setattr(
        confirm.confirm_atomic_v11,
        "generate_confirm_atomic_data_v11",
        lambda *args, **kwargs: ([{"task_id": "atomic"}], {"status": "PASS"}),
    )
    monkeypatch.setattr(confirm.core, "audit_pilot_data", lambda *args, **kwargs: {"ok": True, "errors": []})
    monkeypatch.setattr(confirm, "validate_schema", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        confirm.model_lib,
        "_export_payload_sha",
        lambda *args, **kwargs: confirm.CONFIG["source"]["atomic_export_tree_sha256"],
    )


def _reference_with_forbidden_states(states):
    core = confirm.core
    forbidden = frozenset(states)
    return core.AtomicReference(
        {"synthetic": forbidden},
        {"synthetic": frozenset()},
        forbidden,
        frozenset(),
        {"ok": True},
    )


def test_recovery_spec_keeps_primary_and_training_frozen():
    spec = json.loads((ROOT / "RECOVERY_AMENDMENT_001.json").read_text(encoding="utf-8"))
    protocol = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
    assert spec["status"] == "PREREGISTERED"
    assert spec["algorithm"]["name"] == confirm_atomic_v11.ALGORITHM
    assert spec["algorithm"]["seed"] == protocol["final_data"]["confirm_atomic_seed"]
    assert spec["scope"]["only_non_gating_confirm_atomic_generator_changes"] is True
    assert spec["scope"]["composition_final_generator_unchanged"] is True
    assert spec["scope"]["primary_endpoint_and_gates_unchanged"] is True
    assert spec["scope"]["training_and_all_12_adapters_unchanged"] is True


def test_deterministic_exhaustive_fallback_from_saturated_cell(monkeypatch):
    core = confirm.core
    monkeypatch.setattr(core, "KNOWN_FIELDS", (5, 7))
    monkeypatch.setattr(core, "DEGREES", (2,))
    saturated = {
        core.canonical_state_fingerprint(5, (a, b, c))
        for a in range(5)
        for b in range(5)
        for c in range(5)
    }
    reference = _reference_with_forbidden_states(saturated)
    rows1, audit1 = confirm_atomic_v11.generate_confirm_atomic_data_v11(
        reference, {}, {}, per_operation=2, seed=73200
    )
    rows2, audit2 = confirm_atomic_v11.generate_confirm_atomic_data_v11(
        reference, {}, {}, per_operation=2, seed=73200
    )
    assert rows1 == rows2
    assert audit1 == audit2
    assert len(rows1) == 10
    assert audit1["fallback_count"] > 0
    assert all(row["p"] == 7 for row in rows1)
    assert all(audit1["operation_counts"][operation] == 2 for operation in core.OPS)
    assert core.audit_splits({"confirm_atomic": rows1}, atomic_reference=reference)["ok"] is True


def test_real_reference_has_registered_zero_capacity_cells():
    core = confirm.core
    reference = confirm.atomic_reference()
    for p, degree in ((5, 2), (5, 3), (7, 3)):
        total = p ** (degree + 1)
        occupied = sum(
            core.canonical_state_fingerprint(p, confirm_atomic_v11._state_from_lexicographic_index(index, p, degree + 1))
            in reference.forbidden_state_fingerprints
            for index in range(total)
        )
        assert occupied == total
    free_p7_degree2 = [
        confirm_atomic_v11._state_from_lexicographic_index(index, 7, 3)
        for index in range(7 ** 3)
        if core.canonical_state_fingerprint(
            7, confirm_atomic_v11._state_from_lexicographic_index(index, 7, 3)
        ) not in reference.forbidden_state_fingerprints
    ]
    assert free_p7_degree2 == [(6, 0, 0)]


def test_failure_writer_never_overwrites_primary(tmp_path, monkeypatch):
    monkeypatch.setattr(confirm, "RUNS", tmp_path)
    primary = tmp_path / "FAILED.json"
    primary.write_text('{"original":true}\n', encoding="utf-8")
    before = primary.read_bytes()
    path = confirm._write_failure_receipt({"command": "recover", "status": "FAILED"})
    assert path != primary
    assert primary.read_bytes() == before
    assert json.loads(path.read_text(encoding="utf-8"))["command"] == "recover"


def test_recovery_receipt_requires_exact_spec_fields():
    spec = json.loads((ROOT / "RECOVERY_AMENDMENT_001.json").read_text(encoding="utf-8"))
    receipt = {key: deepcopy(spec[key]) for key in ("original_bindings", "scope", "algorithm")}
    confirm._assert_recovery_receipt_spec_fields(receipt, spec)
    for key in ("original_bindings", "scope", "algorithm"):
        changed = deepcopy(receipt)
        changed[key]["tampered"] = True
        with pytest.raises(RuntimeError, match=key):
            confirm._assert_recovery_receipt_spec_fields(changed, spec)


def test_amendment_commit_rejects_unbound_value():
    with pytest.raises(RuntimeError, match="commit is invalid"):
        confirm._assert_amendment_commit({"git_commit": "not-a-commit", "code": {}})


def test_started_marker_is_exclusive_and_byte_immutable(tmp_path, monkeypatch):
    paths, _ = _isolated_recovery_runtime(tmp_path, monkeypatch)
    _, started, resumed = confirm.begin_recovery_freeze_attempt()
    before = paths["RECOVERY_FREEZE_STARTED_PATH"].read_bytes()
    before_sha = confirm.sha256(paths["RECOVERY_FREEZE_STARTED_PATH"])
    _, same, resumed_again = confirm.begin_recovery_freeze_attempt()
    assert resumed is False
    assert resumed_again is True
    assert same == started
    assert paths["RECOVERY_FREEZE_STARTED_PATH"].read_bytes() == before
    assert confirm.sha256(paths["RECOVERY_FREEZE_STARTED_PATH"]) == before_sha


def test_started_cannot_be_backfilled_after_partial_output(tmp_path, monkeypatch):
    paths, _ = _isolated_recovery_runtime(tmp_path, monkeypatch)
    partial = paths["DATA"] / "final_a.jsonl"
    partial.write_text('{"task_id":"premature"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="appeared before recovery freeze"):
        confirm.begin_recovery_freeze_attempt()
    assert not paths["RECOVERY_FREEZE_STARTED_PATH"].exists()


def test_resume_after_audit_and_some_final_jsonl(tmp_path, monkeypatch):
    paths, _ = _isolated_recovery_runtime(tmp_path, monkeypatch)
    _, started, _ = confirm.begin_recovery_freeze_attempt()
    audit = paths["MANIFESTS"] / "confirm_atomic_generation_audit.json"
    first = paths["DATA"] / "final_a.jsonl"
    manifest = paths["MANIFESTS"] / "final_data_manifest.json"
    audit.write_text('{"ok":true}\n', encoding="utf-8")
    first.write_text('{"task_id":"kept"}\n', encoding="utf-8")
    manifest.write_text('{"status":"partial"}\n', encoding="utf-8")
    bytes_before = {path: path.read_bytes() for path in (audit, first, manifest)}
    _, resumed_started, resumed = confirm.begin_recovery_freeze_attempt()
    assert resumed is True
    assert resumed_started == started
    assert all(path.read_bytes() == payload for path, payload in bytes_before.items())


@pytest.mark.parametrize("evidence", ["access", "ranking", "statistics", "evaluation_done", "eval_receipt"])
def test_resume_rejects_evaluator_evidence(tmp_path, monkeypatch, evidence):
    paths, _ = _isolated_recovery_runtime(tmp_path, monkeypatch)
    confirm.begin_recovery_freeze_attempt()
    targets = {
        "access": paths["MANIFESTS"] / "final_evaluation_access.json",
        "ranking": paths["RANKINGS"] / "replicate0/rankings/job/part-00000.jsonl.gz",
        "statistics": paths["STATISTICS"] / "unexpected.json",
        "evaluation_done": paths["RUNS"] / "PRIMARY_EVALUATION_DONE.json",
        "eval_receipt": paths["RUNS"] / "confirm/replicate0/eval_atomic_control_abc.json",
    }
    target = targets[evidence]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('{}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="resume guard failed"):
        confirm.begin_recovery_freeze_attempt()


def test_done_binds_stable_generation_across_final_access(tmp_path, monkeypatch):
    paths, _ = _isolated_recovery_runtime(tmp_path, monkeypatch)
    confirm.begin_recovery_freeze_attempt()
    frozen = paths["ROOT"] / "CONFIG_FROZEN.json"
    generation = paths["MANIFESTS"] / "final_generation_receipt.json"
    frozen.write_text('{"status":"FROZEN"}\n', encoding="utf-8")
    generation.write_text(json.dumps({"status": "DONE", "evaluation_count": 0}) + "\n", encoding="utf-8")
    done = confirm.complete_recovery_freeze_attempt(frozen, generation)
    done_bytes = paths["RECOVERY_FREEZE_DONE_PATH"].read_bytes()
    generation.write_text(json.dumps({"status": "DONE", "evaluation_count": 1}) + "\n", encoding="utf-8")
    assert confirm.require_recovery_freeze_done(frozen, generation) == done
    assert paths["RECOVERY_FREEZE_DONE_PATH"].read_bytes() == done_bytes


def test_freeze_resumes_after_interruption_immediately_after_audit(tmp_path, monkeypatch):
    paths, _ = _isolated_recovery_runtime(tmp_path, monkeypatch)
    _patch_tiny_freeze_pipeline(monkeypatch)
    original_write = confirm.write_jsonl

    def interrupt_before_first_jsonl(path, rows):
        raise RuntimeError("simulated interruption after audit")

    monkeypatch.setattr(confirm, "write_jsonl", interrupt_before_first_jsonl)
    with pytest.raises(RuntimeError, match="after audit"):
        confirm.freeze_final()
    assert (paths["MANIFESTS"] / "confirm_atomic_generation_audit.json").is_file()
    assert not (paths["ROOT"] / "CONFIG_FROZEN.json").exists()
    started_before = paths["RECOVERY_FREEZE_STARTED_PATH"].read_bytes()
    monkeypatch.setattr(confirm, "write_jsonl", original_write)
    frozen = confirm.freeze_final()
    assert frozen["status"] == "FROZEN"
    assert paths["RECOVERY_FREEZE_STARTED_PATH"].read_bytes() == started_before
    assert paths["RECOVERY_FREEZE_DONE_PATH"].is_file()


def test_freeze_resumes_after_interruption_with_partial_jsonl(tmp_path, monkeypatch):
    paths, _ = _isolated_recovery_runtime(tmp_path, monkeypatch)
    _patch_tiny_freeze_pipeline(monkeypatch)
    original_write = confirm.write_jsonl
    calls = 0

    def interrupt_after_first_jsonl(path, rows):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption after one JSONL")
        return original_write(path, rows)

    monkeypatch.setattr(confirm, "write_jsonl", interrupt_after_first_jsonl)
    with pytest.raises(RuntimeError, match="after one JSONL"):
        confirm.freeze_final()
    written = sorted(paths["DATA"].glob("*.jsonl"))
    assert len(written) == 1
    written_before = written[0].read_bytes()
    started_before = paths["RECOVERY_FREEZE_STARTED_PATH"].read_bytes()
    monkeypatch.setattr(confirm, "write_jsonl", original_write)
    frozen = confirm.freeze_final()
    assert frozen["status"] == "FROZEN"
    assert written[0].read_bytes() == written_before
    assert paths["RECOVERY_FREEZE_STARTED_PATH"].read_bytes() == started_before
    assert paths["RECOVERY_FREEZE_DONE_PATH"].is_file()


def test_final_access_resumes_after_receipt_before_generation_count(tmp_path, monkeypatch):
    paths, _ = _isolated_recovery_runtime(tmp_path, monkeypatch)
    _patch_tiny_freeze_pipeline(monkeypatch)
    frozen = confirm.freeze_final()
    done_before = paths["RECOVERY_FREEZE_DONE_PATH"].read_bytes()
    generation_path = paths["MANIFESTS"] / "final_generation_receipt.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    assert generation["evaluation_count"] == 0
    access_path = paths["MANIFESTS"] / "final_evaluation_access.json"
    access = {
        "schema": "stage4.composition.confirmation.final-access.v1",
        "status": "STARTED",
        "config_frozen_sha256": confirm.sha256(paths["ROOT"] / "CONFIG_FROZEN.json"),
        "test_files": frozen["test_files"],
        "opened_at_unix": 1.0,
        "owner_pid": 12345,
        "resume_count": 0,
    }
    confirm.dump_json(access_path, access, exclusive=True)
    resumed = confirm.begin_final_access()
    repaired_generation = json.loads(generation_path.read_text(encoding="utf-8"))
    assert repaired_generation["evaluation_count"] == 1
    assert resumed["resume_count"] == 1
    assert resumed["owner_pid"] == confirm.os.getpid()
    assert paths["RECOVERY_FREEZE_DONE_PATH"].read_bytes() == done_before
