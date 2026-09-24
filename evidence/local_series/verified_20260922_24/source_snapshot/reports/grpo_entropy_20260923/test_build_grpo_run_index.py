import json

import pytest

from build_grpo_run_index import build_index


def test_index_has_six_arms_and_does_not_mistake_stale_running_marker_for_failure(tmp_path):
    root = tmp_path / "artifacts/grpo_entropy_20260923"
    control = root / "runs/seed0_control_fp32"
    evaluation = control / "evaluation"
    evaluation.mkdir(parents=True)
    (control / "RUNNING.json").write_text('{"status":"STARTED"}')
    (control / "DONE.json").write_text(json.dumps({"status": "DONE", "steps": 150,
                                                    "groups": 150, "answers": 1200}))
    (evaluation / "DONE.json").write_text(json.dumps({"status": "DONE", "tasks": 1000,
                                                       "candidate_program_scores": 125000}))
    entropy = root / "runs/seed0_entropy_fp32"
    entropy.mkdir(parents=True)
    (entropy / "RUNNING.json").write_text('{"status":"STARTED"}')

    rows = build_index(root)

    assert len(rows) == 6
    assert {(row["seed"], row["arm"]) for row in rows} == {
        (seed, arm) for seed in (0, 1, 2) for arm in ("control", "entropy")}
    assert rows[0]["training_status"] == "DONE"
    assert rows[0]["evaluation_status"] == "DONE"
    assert rows[0]["gpu_index"] == 1
    assert rows[1]["training_status"] == "STARTED"
    assert rows[1]["evaluation_status"] == "NOT_STARTED"
    assert next(row for row in rows if row["seed"] == 1 and row["arm"] == "control")["gpu_index"] == 2


def test_failed_and_done_markers_for_same_run_are_rejected(tmp_path):
    run = tmp_path / "artifacts/grpo_entropy_20260923/runs/seed0_control_fp32"
    run.mkdir(parents=True)
    (run / "DONE.json").write_text('{"status":"DONE"}')
    (run / "FAILED.json").write_text('{"status":"FAILED"}')

    with pytest.raises(ValueError, match="contradictory"):
        build_index(tmp_path / "artifacts/grpo_entropy_20260923")
