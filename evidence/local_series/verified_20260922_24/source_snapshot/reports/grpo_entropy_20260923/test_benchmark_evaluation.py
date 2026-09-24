from types import SimpleNamespace

import pytest


def test_speed_probe_uses_only_ten_depth_three_training_tasks():
    from benchmark_evaluation import select_training_tasks

    tasks = [SimpleNamespace(task_id=f"other-{i}", max_steps=2) for i in range(20)]
    tasks += [SimpleNamespace(task_id=f"train-{i}", max_steps=3) for i in range(11)]
    assert [task.task_id for task in select_training_tasks(tasks)] == [f"train-{i}" for i in range(10)]
    with pytest.raises(ValueError, match="depth-three"):
        select_training_tasks(tasks[:20])
