from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_capacity import paired_contrast
from scripts.analyze_ranking import _bootstrap_p, _crossed_draws, factorial_effects, holm


def test_crossed_bootstrap_and_factorial_effects_resample_seeds_and_tasks(monkeypatch):
    values = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    draws = _crossed_draws(values, np.random.default_rng(7), 200)
    assert len(draws) == 200
    assert 0.0 <= draws.min() <= draws.max() <= 1.0
    assert 0.0 < _bootstrap_p(draws, values.mean()) <= 1.0

    cell_values = {
        "motif_a_d3": 0.6,
        "motif_b_d3": -0.2,
        "motif_c_d3": 0.5,
        "motif_d_d3": -0.1,
    }

    def fake_aligned(_found, split, _baseline, _roles, _k):
        value = cell_values[split]
        base = np.zeros(4)
        branch = np.full((3, 4), value)
        return base, branch, ["t0", "t1", "t2", "t3"]

    monkeypatch.setattr("scripts.analyze_ranking._aligned", fake_aligned)
    result = factorial_effects(
        {}, ["nomotif0", "nomotif1", "nomotif2"], 32, np.random.default_rng(8), 200
    )
    assert result["motif_effect_pp"] == pytest.approx(70.0)
    assert result["field_effect_pp"] == pytest.approx(0.0)
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": 0.03, "c": 0.06, "b": 0.06}


def test_sampled_paired_contrast_rejects_dropped_tasks(monkeypatch, tmp_path):
    full = {f"t{i}": (False, False) for i in range(300)}
    partial = dict(list(full.items())[:-1])

    def fake_outcomes(run, _method):
        return full if run.name == "full" else partial

    monkeypatch.setattr("scripts.analyze_capacity.outcomes", fake_outcomes)
    with pytest.raises(ValueError, match="paired task mismatch"):
        paired_contrast(
            [(0, tmp_path / "full", tmp_path / "partial", "iid_action@0.7")],
            np.random.default_rng(1),
            10,
        )
