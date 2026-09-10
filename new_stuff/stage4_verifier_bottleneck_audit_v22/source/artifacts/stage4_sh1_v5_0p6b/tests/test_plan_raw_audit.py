import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "code" / "plan_raw_audit.py"
SPEC = importlib.util.spec_from_file_location("stage4_plan_raw_audit", PATH)
AUDIT = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(AUDIT)


def _records():
    rows = []
    for operation in AUDIT.v5.OPS:
        rows.append({"task_id": f"{operation}-ok", "target_operation": operation,
                     "predicted_operation": operation, "correct": True})
        rows.append({"task_id": f"{operation}-bad", "target_operation": operation,
                     "predicted_operation": "SC2" if operation != "SC2" else "REV", "correct": False})
    return rows


def test_plan_metrics_are_rederived_from_unique_raw_rows():
    metrics = AUDIT.derive_plan_metrics(_records())
    assert metrics["overall"] == 0.5
    assert metrics["by_operation"] == {operation: 0.5 for operation in AUDIT.v5.OPS}


def test_plan_metrics_reject_duplicate_or_inconsistent_rows():
    rows = _records(); rows.append(dict(rows[0]))
    with pytest.raises(RuntimeError, match="duplicate"):
        AUDIT.derive_plan_metrics(rows)
    rows = _records(); rows[0]["correct"] = False
    with pytest.raises(RuntimeError, match="inconsistent"):
        AUDIT.derive_plan_metrics(rows)


def test_audit_jsonl_resume_accepts_only_identical_output(tmp_path):
    path = tmp_path / "audit.jsonl"; rows = [{"task_id": "one", "value": 1}]
    AUDIT.write_or_verify_jsonl(path, rows)
    AUDIT.write_or_verify_jsonl(path, rows)
    with pytest.raises(RuntimeError, match="differs"):
        AUDIT.write_or_verify_jsonl(path, [{"task_id": "one", "value": 2}])
