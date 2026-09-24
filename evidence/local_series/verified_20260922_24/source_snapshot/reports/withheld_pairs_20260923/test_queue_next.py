import pytest
from pathlib import Path
from subprocess import CompletedProcess


def test_jobs_keep_complete_seed_pairs_and_all_evaluations_last():
    from queue_next import ordered_jobs

    jobs = ordered_jobs()
    assert len(jobs) == 12
    assert jobs[:4] == [
        ("train", 85000, "random"),
        ("train", 85000, "withheld"),
        ("train", 85001, "random"),
        ("train", 85001, "withheld"),
    ]
    assert [mode for mode, _, _ in jobs] == ["train"] * 6 + ["evaluate"] * 6


@pytest.mark.parametrize(
    ("accepted", "state", "device", "expected"),
    [
        (True, None, None, "SKIP"),
        (False, None, None, "START"),
        (False, {"Status": "running", "ExitCode": 0}, "GPU-123", "WAIT"),
        (False, {"Status": "exited", "ExitCode": 0}, "GPU-123", "VERIFY"),
    ],
)
def test_resume_decision_never_restarts_accepted_or_live_job(accepted, state, device, expected):
    from queue_next import resume_decision

    assert resume_decision(accepted, state, device, "GPU-123") == expected


def test_resume_refuses_wrong_gpu_or_failed_container():
    from queue_next import resume_decision

    with pytest.raises(ValueError, match="GPU UUID"):
        resume_decision(False, {"Status": "running", "ExitCode": 0}, "GPU-OTHER", "GPU-123")
    with pytest.raises(ValueError, match="failed"):
        resume_decision(False, {"Status": "exited", "ExitCode": 1}, "GPU-123", "GPU-123")


def test_gpu_snapshot_rejects_busy_or_wrong_device():
    from queue_next import gpu_is_available

    snapshot = "0, GPU-123, NVIDIA GeForce RTX 3060, 8 MiB, 0 %\n1, GPU-456, Tesla V100, 5000 MiB, 40 %\n"
    assert gpu_is_available(snapshot, 0, "GPU-123")
    assert not gpu_is_available(snapshot, 1, "GPU-456")
    with pytest.raises(ValueError, match="GPU UUID"):
        gpu_is_available(snapshot, 0, "GPU-OTHER")


def test_training_receipt_requires_finite_complete_fp16_fp32_on_assigned_gpu():
    from queue_next import validate_training_receipt

    receipt = {
        "status": "DONE", "optimizer_steps": 30, "optimizer_steps_expected": 30,
        "mean_loss": 0.8, "runtime_compute_dtype": "torch.float16",
        "runtime_master_dtype": "torch.float32", "cuda_device_name": "NVIDIA GeForce RTX 3060",
    }
    validate_training_receipt(receipt, "RTX 3060")
    for changed in (
        {"status": "FAILED"}, {"optimizer_steps": 29}, {"mean_loss": float("nan")},
        {"runtime_compute_dtype": "torch.bfloat16"},
        {"runtime_master_dtype": "torch.float16"},
        {"cuda_device_name": "Tesla V100"},
    ):
        with pytest.raises(ValueError):
            validate_training_receipt({**receipt, **changed}, "RTX 3060")


def test_training_receipt_accepts_shell_safe_gpu_name_spelling():
    from queue_next import validate_training_receipt

    receipt = {
        "status": "DONE", "optimizer_steps": 30, "optimizer_steps_expected": 30,
        "mean_loss": 0.8, "runtime_compute_dtype": "torch.float16",
        "runtime_master_dtype": "torch.float32", "cuda_device_name": "Tesla V100-SXM2-32GB",
    }
    validate_training_receipt(receipt, "Tesla_V100-SXM2-32GB")


def test_evaluation_receipt_requires_all_tasks_and_programs():
    from queue_next import validate_evaluation_receipt

    status = {"status": "DONE", "tasks": 1000}
    summary = {"n_tasks": 1000, "candidate_program_scores": 125000, "hit@32": 0.3}
    validate_evaluation_receipt(status, summary)
    with pytest.raises(ValueError):
        validate_evaluation_receipt(status, {**summary, "candidate_program_scores": 124999})
    with pytest.raises(ValueError):
        validate_evaluation_receipt(status, {**summary, "hit@32": float("nan")})


def test_container_launch_binds_the_assigned_physical_gpu_uuid(monkeypatch):
    import queue_next

    captured = []

    def external_docker(args):
        captured.extend(args)
        return CompletedProcess(args, 0, stdout="container-123\n", stderr="")

    monkeypatch.setattr(queue_next, "command", external_docker)
    assert queue_next.start_job(Path("/scratch"), 0, 1, 2, "train", 85000,
                                "random", "1045:1046", "/work/python_pkgs_eval",
                                "run_first.py", "GPU-123") == "container-123"
    assert captured[captured.index("--gpus") + 1] == "device=0"
    assert "ASSIGNED_GPU_UUID=GPU-123" in captured


def test_accepted_exited_container_logs_are_captured_on_resume(monkeypatch, tmp_path):
    import queue_next

    captured = []
    monkeypatch.setattr(queue_next, "capture_container", lambda name, output: captured.append((name, output)))
    queue_next.capture_accepted_if_exited(
        "training-123", {"State": {"Status": "exited", "ExitCode": 0}}, tmp_path
    )
    assert captured == [("training-123", tmp_path)]
    queue_next.capture_accepted_if_exited(
        "training-123", {"State": {"Status": "running", "ExitCode": 0}}, tmp_path
    )
    assert len(captured) == 1


def test_amended_runner_accepts_shell_safe_rtx_name_only():
    from queue_next import validate_runner_assignment

    validate_runner_assignment("run_amended.py", "NVIDIA_GeForce_RTX_3060")
    with pytest.raises(ValueError, match="RTX 3060"):
        validate_runner_assignment("run_amended.py", "Tesla_V100-SXM2-32GB")
