import pytest


def test_rtx_runner_accepts_only_assigned_3060_uuid():
    from run_amended import validate_device_binding

    validate_device_binding("NVIDIA GeForce RTX 3060", "GPU-123", "GPU-123")
    for name, actual, assigned in (
        ("Tesla V100", "GPU-123", "GPU-123"),
        ("NVIDIA GeForce RTX 2080 Ti", "GPU-123", "GPU-123"),
        ("NVIDIA GeForce RTX 3060", "GPU-OTHER", "GPU-123"),
        ("NVIDIA GeForce RTX 3060", "GPU-123", ""),
    ):
        with pytest.raises(ValueError):
            validate_device_binding(name, actual, assigned)


def test_rtx_runner_requires_exactly_one_visible_gpu_uuid():
    from run_amended import parse_visible_uuid

    assert parse_visible_uuid("GPU-123\n") == "GPU-123"
    with pytest.raises(ValueError):
        parse_visible_uuid("GPU-123\nGPU-456\n")
    with pytest.raises(ValueError):
        parse_visible_uuid("\n")
