import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from confirm_stats import (analyze_primary, crossed_seed_task_bootstrap, exact_sign_flip,
                           holm, seed_t_statistics)


def test_sign_flip_minimum_and_five_of_six_limit():
    assert exact_sign_flip([7, 7, 7, 7, 7, 7], 100)["p_two_sided"] == 0.03125
    assert exact_sign_flip([7, 7, 7, 7, 7, -1], 100)["p_two_sided"] == 0.0625
    assert exact_sign_flip([7, 7, 7, 7, 7, 0], 100)["p_two_sided"] >= 0.0625


def test_t_fixture_and_strict_gate():
    stats = seed_t_statistics([0.07, 0.07, 0.07, 0.07, 0.07, -0.01])
    assert abs(stats["mean_delta"] - 0.05666666666666667) < 1e-12
    assert abs(stats["p_two_sided"] - 0.00809) < 2e-5
    assert stats["t_ci95"][0] > 0


def test_crossed_bootstrap_is_deterministic():
    matrix = np.arange(60, dtype=np.float64).reshape(6, 10) / 100
    left, left_values = crossed_seed_task_bootstrap(matrix, repetitions=200, seed=17)
    right, right_values = crossed_seed_task_bootstrap(matrix, repetitions=200, seed=17)
    assert left == right
    assert np.array_equal(left_values, right_values)


def test_holm_fixed_family():
    rows = holm({"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.2, "e": 0.5})
    assert [row["endpoint"] for row in rows] == ["a", "b", "c", "d", "e"]
    assert rows[0]["holm_adjusted_p"] == 0.05


def test_primary_exact_pairing_and_counts():
    tasks = [f"t{i}" for i in range(100)]
    control = {seed: {task: float(index < 40) for index, task in enumerate(tasks)} for seed in range(6)}
    distill = {seed: {task: float(index < 50) for index, task in enumerate(tasks)} for seed in range(6)}
    result, replicates = analyze_primary(control, distill, repetitions=100, bootstrap_seed=9)
    assert result["criteria"]["composition_effect_pass"] is True
    assert result["per_seed"][0]["delta_numerator"] == 10
    assert replicates.shape == (100,)

