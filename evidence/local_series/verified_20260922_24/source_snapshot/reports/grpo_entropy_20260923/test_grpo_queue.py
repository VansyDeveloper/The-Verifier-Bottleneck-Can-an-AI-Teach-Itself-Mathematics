import pytest


def test_queue_orders_complete_seed_pairs_on_one_gpu():
    from grpo_queue import ordered_jobs

    assert ordered_jobs((0, 2)) == [
        (0, "train", "control"), (0, "train", "entropy"),
        (0, "evaluate", "control"), (0, "evaluate", "entropy"),
        (2, "train", "control"), (2, "train", "entropy"),
        (2, "evaluate", "control"), (2, "evaluate", "entropy"),
    ]


def test_queue_resume_never_duplicates_live_or_accepted_job():
    from grpo_queue import resume_decision

    assert resume_decision(False, None) == "START"
    assert resume_decision(False, {"Status": "running", "ExitCode": 0}) == "WAIT"
    assert resume_decision(True, {"Status": "exited", "ExitCode": 0}) == "SKIP"
    with pytest.raises(ValueError, match="failed"):
        resume_decision(False, {"Status": "exited", "ExitCode": 1})
    with pytest.raises(ValueError, match="acceptance"):
        resume_decision(False, {"Status": "exited", "ExitCode": 0})


def test_pair_config_refuses_different_gpu_or_step_count():
    from grpo_queue import validate_pair

    common = {"seed": 0, "steps": 400, "gpu_uuid": "GPU-1", "input_sha256": "A"}
    validate_pair({**common, "entropy_coef": 0.0}, {**common, "entropy_coef": 0.01})
    with pytest.raises(ValueError, match="unequal"):
        validate_pair({**common, "entropy_coef": 0.0},
                      {**common, "gpu_uuid": "GPU-2", "entropy_coef": 0.01})
    with pytest.raises(ValueError, match="unequal"):
        validate_pair({**common, "entropy_coef": 0.0},
                      {**common, "steps": 150, "entropy_coef": 0.01})


def test_training_acceptance_uses_frozen_150_step_budget(tmp_path):
    import json
    import grpo_queue

    run = tmp_path / "runs/seed0_control_fp32"
    adapter = run / "final_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter.bin").write_bytes(b"weights")
    metrics = run / "metrics.jsonl"
    generations = run / "generations.jsonl"
    metrics.write_bytes(b"{}\n" * 150)
    generations.write_bytes(b"{}\n" * 1200)
    receipt = {
        "status": "DONE", "steps": 150, "groups": 150, "answers": 1200,
        "metrics_sha256": grpo_queue.sha256(metrics),
        "generations_sha256": grpo_queue.sha256(generations),
        "adapter_tree_sha256": grpo_queue.tree_sha256(adapter),
        "config": {"seed": 0, "gpu_uuid": "GPU-1", "entropy_coef": 0.0,
                   "gradient_checkpointing": "non_reentrant_trainable_model",
                   "input_sha256": grpo_queue.INPUT_SHA,
                   "base_sha256": grpo_queue.BASE_SHA,
                   "adapter_sha256": grpo_queue.ADAPTER_SHA},
    }
    (run / "DONE.json").write_text(json.dumps(receipt), encoding="utf-8")
    assert grpo_queue.accepted(tmp_path, 0, "train", "control", "GPU-1", 150)
    with pytest.raises(ValueError, match="frozen protocol"):
        grpo_queue.accepted(tmp_path, 0, "train", "control", "GPU-1", 400)


def test_training_command_uses_only_assigned_gpu_and_frozen_inputs(tmp_path):
    from grpo_queue import docker_command

    args = docker_command(tmp_path, 1, "GPU-1", 0, "train", "entropy", "2031:2032", 150)
    assert args[args.index("--gpus") + 1] == "device=1"
    assert args[args.index("--steps") + 1] == "150"
    assert args[args.index("--entropy-coef") + 1] == "0.01"
    assert "--gpu-uuid" in args and "GPU-1" in args
    assert args[args.index("--run-name") + 1] == "seed0_entropy_fp32"


def test_fp32_queue_evidence_is_separate_from_failed_fp16_evidence(tmp_path):
    from grpo_queue import queue_paths

    allocation, events, lock = queue_paths(tmp_path, 1)
    assert allocation.name == "queue_gpu1_fp32_allocation.json"
    assert events.name == "queue_gpu1_fp32.jsonl"
    assert lock.name == "queue_gpu1_fp32.lock"
    assert not any("queue_gpu1_allocation.json" == p.name for p in (allocation, events, lock))


def test_queue_rejects_steps_different_from_blind_duration_freeze():
    from grpo_queue import require_frozen_steps

    freeze = {"status": "FROZEN_BEFORE_FULL_PAIR", "steps_per_arm": 150,
              "heldout_outcomes_viewed_before_decision": False}
    require_frozen_steps(freeze, 150)
    with pytest.raises(ValueError, match="duration freeze"):
        require_frozen_steps(freeze, 400)


def test_evaluation_launch_requires_training_but_not_empty_run_directory(tmp_path):
    from grpo_queue import launch_path_is_clear

    output = tmp_path / "study"
    run = output / "runs/seed0_control_fp32"
    run.mkdir(parents=True)
    (run / "DONE.json").write_text("{}", encoding="utf-8")
    launch_path_is_clear(output, 0, "evaluate", "control")
    with pytest.raises(ValueError, match="partial"):
        launch_path_is_clear(output, 0, "train", "control")
    (run / "evaluation").mkdir()
    with pytest.raises(ValueError, match="partial"):
        launch_path_is_clear(output, 0, "evaluate", "control")


def test_queue_pins_current_fp32_train_and_evaluation_code():
    import grpo_queue
    from pathlib import Path

    directory = Path(__file__).parent
    assert grpo_queue.sha256(directory / "run_ablation.py") == grpo_queue.TRAIN_CODE_SHA
    assert grpo_queue.sha256(directory / "run_evaluation.py") == grpo_queue.EVAL_CODE_SHA
