import json
from pathlib import Path


def test_duration_choice_has_25_percent_margin_before_deadline():
    data = json.loads(Path(__file__).with_name("DURATION_FREEZE.json").read_text(encoding="utf-8"))
    per_arm = (data["steps_per_arm"] * data["conservative_step_seconds"] +
               1000 * data["conservative_evaluation_task_seconds"] +
               data["per_arm_loading_overhead_seconds"])
    predicted = per_arm * data["arms_per_wave"] * data["paired_seed_waves"]
    assert predicted == data["predicted_makespan_seconds"]
    assert predicted * 1.25 == data["with_25_percent_margin_seconds"]
    assert data["with_25_percent_margin_seconds"] < data["available_to_0800_msk_seconds"]
    assert data["heldout_outcomes_viewed_before_decision"] is False


def test_repaired_fp32_duration_is_blind_and_rejects_400_steps():
    data = json.loads(Path(__file__).with_name("DURATION_FREEZE_FP32.json").read_text(encoding="utf-8"))
    assert data["schema"] == "grpo-entropy.duration-freeze-fp32.v1"
    assert data["status"] == "FROZEN_BEFORE_FULL_PAIR"
    assert data["steps_per_arm"] == 150
    assert data["benchmark_evaluation_input"] == "10 depth-three training tasks, not held-out tasks"
    per_arm = (data["steps_per_arm"] * data["conservative_step_seconds"] +
               1000 * data["conservative_evaluation_task_seconds"] +
               data["per_arm_loading_overhead_seconds"])
    assert per_arm * data["arms_per_wave"] * data["paired_seed_waves"] == data["predicted_makespan_seconds"]
    assert data["predicted_makespan_seconds"] * 1.25 == data["with_25_percent_margin_seconds"]
    assert data["with_25_percent_margin_seconds"] < data["available_to_0800_msk_seconds"]
    assert data["400_step_with_25_percent_margin_seconds"] > data["available_to_0800_msk_seconds"]
    assert data["heldout_outcomes_viewed_before_decision"] is False
