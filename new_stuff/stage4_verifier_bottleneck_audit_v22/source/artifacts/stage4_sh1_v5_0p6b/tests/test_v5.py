import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "code/v5_train.py"
SPEC = importlib.util.spec_from_file_location("v5_train", PATH)
V5 = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(V5)


def test_mode_quota_is_twenty_and_diverse():
    assert len(V5.MODE_QUOTA) == 20
    assert set(V5.MODE_QUOTA) == set(V5.CFG["modes"])


def test_curriculum_sizes():
    assert sum(V5.CFG["coordinate_states_by_degree"].values()) == 24000
    assert sum(V5.CFG["full_sh1_by_degree"].values()) == 64000


def test_prefix_count_exact():
    counts = V5.CFG["coordinate_states_by_degree"]
    assert sum(int(counts[str(d)]) * d for d in V5.CFG["degrees"]) == 80000


def test_model_is_frozen_to_0p6b():
    assert V5.CFG["model"] == "Qwen/Qwen3-0.6B"
    assert "1.7" not in V5.CFG["parent_adapter"]


def test_state_digest_is_canonical_and_field_bound():
    assert V5.state_key_digest(7, [1, 2, 3]) == V5.state_key_digest(7, (1, 2, 3))
    assert V5.state_key_digest(7, [1, 2, 3]) != V5.state_key_digest(11, [1, 2, 3])


def test_apply_metric_is_strict_not_prefix_acceptance():
    rows = [
        {"operation": "SH1", "target": "[1, 2]", "raw_completion": " [1, 2] ", "correct": True},
        {"operation": "SH1", "target": "[1, 2]", "raw_completion": "[1, 2] trailing", "correct": True},
    ]
    metrics = V5.exact_apply_metrics(rows)
    assert metrics["overall"] == 0.5
    assert rows[0]["correct"] is True
    assert rows[1]["legacy_prefix_correct"] is True
    assert rows[1]["correct"] is False


def test_corrective_requires_plan_controls_and_low_sh1():
    gate = {
        "plan": {"overall": 0.99},
        "control_checks": {op: True for op in ("SC2", "REV", "AC1", "AX1")},
        "apply": {"by_operation": {"SH1": 0.70}},
    }
    assert V5.corrective_allowed(gate)
    gate["plan"]["overall"] = 0.90
    assert not V5.corrective_allowed(gate)


def test_final_access_is_gate_bound_and_exactly_once(tmp_path, monkeypatch):
    runs = tmp_path / "runs"; manifests = tmp_path / "manifests"; data = tmp_path / "data"
    runs.mkdir(); manifests.mkdir(); data.mkdir(); (runs / "generations").mkdir()
    adapter = tmp_path / "curriculum"; adapter.mkdir(); (adapter / "adapter_model.bin").write_bytes(b"adapter")
    calibration = data / "calibration.jsonl"; calibration.write_text('{"task_id":"one"}\n', encoding="utf-8")
    (manifests / "data_manifest.json").write_text(json.dumps({
        "calibration.jsonl": {"sha256": V5.sha256(calibration), "rows": 1}
    }), encoding="utf-8")
    exclusion = manifests / "eval_state_exclusion.json"
    exclusion.write_text(json.dumps({"state_digests": []}), encoding="utf-8")
    monkeypatch.setattr(V5, "RUNS", runs)
    monkeypatch.setattr(V5, "MANIFESTS", manifests)
    monkeypatch.setattr(V5, "DATA", data)
    generations = runs / "generations" / "curriculum_calibration.jsonl"
    generations.write_text('{"task_id":"one"}\n', encoding="utf-8")
    gate = runs / "curriculum_calibration_gate.json"
    gate.write_text(json.dumps({
        "pass": True,
        "binding": V5.gate_binding(adapter, "calibration", "curriculum_calibration"),
        "generation_sha256": V5.sha256(generations),
        "generation_rows": 1,
    }), encoding="utf-8")
    receipt = V5.authorize_final_access(adapter, "atomic_v5_final")
    assert receipt["status"] == "STARTED"
    access_path = manifests / "final_access_receipt.json"
    competing = json.loads(access_path.read_text(encoding="utf-8")); competing["owner_pid"] = 987654321
    access_path.write_text(json.dumps(competing), encoding="utf-8")
    monkeypatch.setattr(V5, "pid_alive", lambda pid: pid == 987654321)
    with pytest.raises(RuntimeError, match="already active"):
        V5.authorize_final_access(adapter, "atomic_v5_final")
    monkeypatch.setattr(V5, "pid_alive", lambda _pid: False)
    resumed = V5.authorize_final_access(adapter, "atomic_v5_final")
    assert resumed["status"] == "STARTED" and resumed["resume_count"] == 1
    assert resumed["owner_pid"] == V5.os.getpid()
    generation_path = runs / "generations/atomic_v5_final.jsonl"
    V5.write_jsonl(generation_path, [{"task_id": "final-1"}])
    final_gate_path = runs / "atomic_v5_final_gate.json"
    final_gate_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(V5, "load_bound_gate", lambda *_args, **_kwargs: {"pass": True})
    completed = V5.finalize_final_access(adapter, "atomic_v5_final", final_gate_path)
    assert completed["status"] == "DONE" and completed["generation_rows"] == 1
    assert V5.finalize_final_access(adapter, "atomic_v5_final", final_gate_path) == completed
    with pytest.raises(RuntimeError, match="already evaluated"):
        V5.authorize_final_access(adapter, "atomic_v5_final")


def test_corrective_state_tracks_reuse_after_unique_search_exhaustion(monkeypatch):
    calls = []
    unique_forbidden = {"eval", "used"}; eval_forbidden = {"eval"}
    def fake_make(_rng, p, degree, mode, forbidden, attempts=5000):
        calls.append(set(forbidden))
        if forbidden is unique_forbidden:
            raise RuntimeError("unique search exhausted")
        return [1, 2, 3], mode
    monkeypatch.setattr(V5, "make_unique_state_hashed", fake_make)
    monkeypatch.setattr(V5, "state_key_digest", lambda _p, _state: "new")
    state, mode, reused = V5.make_corrective_state(object(), 5, 2, "progression",
                                                    unique_forbidden, eval_forbidden)
    assert state == [1, 2, 3] and mode == "progression" and reused is False
    assert "new" in unique_forbidden
    _, _, reused = V5.make_corrective_state(object(), 5, 2, "progression",
                                            unique_forbidden, eval_forbidden)
    assert reused is True
    assert calls == [{"eval", "used"}, {"eval"}, {"eval", "new", "used"}, {"eval"}]


def test_phase_receipt_and_seal_fail_closed_on_adapter_change(tmp_path, monkeypatch):
    data = tmp_path / "data"; manifests = tmp_path / "manifests"
    data.mkdir(); manifests.mkdir()
    source = data / "coordinate_source.jsonl"; source.write_text('{"row":1}\n', encoding="utf-8")
    (manifests / "environment.json").write_text(json.dumps({"git_commit": "launch-commit"}), encoding="utf-8")
    parent = tmp_path / "parent"; parent.mkdir(); (parent / "adapter_model.bin").write_bytes(b"parent")
    output = tmp_path / "coordinate"; output.mkdir()
    (output / "adapter_model.safetensors").write_bytes(b"weights")
    (output / "adapter_config.json").write_text("{}", encoding="utf-8")
    (output / "tokenizer.json").write_text("{}", encoding="utf-8")
    spec = {"epochs": 1, "lr": 1e-5, "effective_batch": 64}
    receipt = {"status": "DONE", "parent": str(parent), "phase": {
        "phase": "coordinate", "status": "DONE", "examples": 10,
        "epochs": 1, "lr": 1e-5, "effective_batch": 64,
    }}
    (output / "training_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(V5, "DATA", data)
    monkeypatch.setattr(V5, "MANIFESTS", manifests)
    assert V5.validate_phase_receipt(parent, output, spec, "coordinate", 10)
    V5.seal_phase(parent, output, spec, "coordinate", 10)
    assert V5.phase_complete(parent, output, spec, "coordinate", 10)
    (output / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="phase seal"):
        V5.phase_complete(parent, output, spec, "coordinate", 10)
