import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent.parent
sys.path.insert(0, str(ROOT / "code"))
SPEC = importlib.util.spec_from_file_location("stage4_exploratory_guard", ROOT / "code/stage4.py")
stage4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage4)


def test_authorization_is_hash_bound_and_never_references_old_final():
    authorization = json.loads((ROOT / "configs/exploratory_atomic_authorization.json").read_text(encoding="utf-8"))
    assert authorization["status"] == "AUTHORIZED_FOR_EXPLORATION"
    assert authorization["registered_atomic_pass"] is False
    assert authorization["old_atomic_final_access_forbidden"] is True
    assert "final" not in " ".join(authorization["source_files"]).lower()
    assert "final.jsonl" not in " ".join(entry["path"] for entry in authorization["source_files"].values()).lower()
    for entry in authorization["source_files"].values():
        path = REPO / entry["path"]
        assert path.is_file()
        assert stage4.sha256(path).upper() == entry["sha256"].upper()


def test_real_bound_source_validation_succeeds_without_old_final(tmp_path, monkeypatch):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    monkeypatch.setattr(stage4, "MANIFESTS", manifests)
    stage4.dump_json(manifests / "preregistration.json", {
        "protocol_sha256": stage4.sha256(stage4.PROTOCOL_PATH),
        "exploratory_atomic_authorization_sha256": stage4.sha256(stage4.ATOMIC_AUTHORIZATION_PATH),
    })
    evidence = stage4._require_exploratory_atomic_source()
    audit = evidence["source_audit"]
    assert audit["status"] == "PASS"
    assert audit["registered_atomic_pass"] is False
    assert audit["old_atomic_final_accessed"] is False
    assert all(audit["structural_checks"][key] for key in audit["structural_checks"]
               if key != "registered_atomic_pass")
    assert audit["structural_checks"]["registered_atomic_pass"] is False


def test_atomic_reference_does_not_read_old_final(monkeypatch):
    reads = []
    selected = REPO / "artifacts/stage4_sh1_v5_0p6b/adapters/corrective"
    authorization = stage4.load_json(ROOT / "configs/exploratory_atomic_authorization.json")
    monkeypatch.setattr(stage4, "require_atomic_pass", lambda: {"selected_adapter": selected,
                                                                 "decision_path": ROOT / "configs/exploratory_atomic_authorization.json",
                                                                 "decision": authorization})
    monkeypatch.setattr(stage4, "read_jsonl", lambda path: reads.append(Path(path).name) or [])
    monkeypatch.setattr(stage4, "audit_atomic_reference",
                        lambda *args, **kwargs: SimpleNamespace(report={"ok": True}))
    monkeypatch.setattr(stage4, "dump_json_atomic", lambda *args, **kwargs: None)
    stage4.build_atomic_reference()
    assert "final.jsonl" not in reads
    assert "calibration.jsonl" in reads


def test_exploratory_checks_never_report_atomic_pass():
    names = stage4._pilot_required_check_names()
    assert "exploratory_source_authorized" in names
    assert "registered_atomic_pass" not in names
    checks = {name: True for name in names}
    checks.update({"atomic_pass": False, "registered_atomic_pass": False})
    assert stage4._pilot_pass(checks)
    assert checks["atomic_pass"] is False


def test_frozen_source_identity_requires_exact_metrics_and_per_task_rows(tmp_path, monkeypatch):
    gate_path = tmp_path / "gate.json"
    source_apply_path = tmp_path / "apply.jsonl"
    source_plan_path = tmp_path / "plan.jsonl"
    authorization_path = tmp_path / "authorization.json"
    runs = tmp_path / "runs"; manifests = tmp_path / "manifests"
    base_path = runs / "pilot/shared_atomic_base/eval/atomic/atomic_base.generations.jsonl"
    by_operation = {operation: 1.0 for operation in stage4.OPS}
    gate = {"plan": {"overall": 1.0, "by_operation": by_operation},
            "apply": {"overall": 1.0, "by_operation": by_operation}}
    stage4.dump_json(gate_path, gate)
    stage4.write_jsonl(source_apply_path, [{"task_id": "t1", "operation": "SH1",
                                            "raw_completion": "[1]", "correct": True}])
    stage4.write_jsonl(source_plan_path, [{"task_id": "t1", "predicted_operation": "SH1", "correct": True}])
    stage4.write_jsonl(base_path, [{"task_id": "t1", "operation": "SH1", "raw_completion": "[1]",
                                    "correct": True, "plan_prediction": "SH1", "plan_correct": True}])
    stage4.dump_json(authorization_path, {"source_files": {
        "calibration_gate": {"path": str(gate_path)},
        "calibration_generations": {"path": str(source_apply_path)},
        "corrective_plan_raw": {"path": str(source_plan_path)},
    }})
    monkeypatch.setattr(stage4, "ATOMIC_AUTHORIZATION_PATH", authorization_path)
    monkeypatch.setattr(stage4, "RUNS", runs)
    monkeypatch.setattr(stage4, "MANIFESTS", manifests)
    monkeypatch.setattr(stage4, "_repo_path", lambda value: Path(value))
    atomic = {"plan": {"overall": 1.0, "by_operation": by_operation},
              "apply": {"overall": 1.0, "by_operation": by_operation}}
    receipt = stage4.validate_frozen_source_identity(atomic)
    assert receipt["status"] == "PASS"
    assert receipt["per_task_exact_match"] is True
    stage4.write_jsonl(base_path, [{"task_id": "t1", "operation": "SH1", "raw_completion": "[2]",
                                    "correct": False, "plan_prediction": "SH1", "plan_correct": True}])
    with pytest.raises(RuntimeError, match="per-task behavior changed"):
        stage4.validate_frozen_source_identity(atomic)


@pytest.mark.parametrize("function_name", ["freeze_and_generate_final", "confirm", "report", "archive", "run"])
def test_exploratory_protocol_blocks_registered_only_paths(function_name):
    function = getattr(stage4, function_name)
    with pytest.raises(RuntimeError, match="exploratory|confirmation|registered|full registered"):
        function({"pass": True} if function_name == "freeze_and_generate_final" else None) if function_name == "freeze_and_generate_final" else function()
