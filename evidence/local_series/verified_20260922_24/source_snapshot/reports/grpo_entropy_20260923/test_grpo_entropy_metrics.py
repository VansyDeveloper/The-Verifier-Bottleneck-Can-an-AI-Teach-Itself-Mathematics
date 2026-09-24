import math

import pytest
import torch


def test_entropy_bonus_gradient_flattens_peaked_operation_policy():
    from metrics import normalized_entropy

    logits = torch.tensor([5.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
    loss = -0.01 * normalized_entropy(logits)
    loss.backward()
    assert logits.grad[0] > 0
    assert math.isclose(normalized_entropy(torch.zeros(5)).item(), 1.0, abs_tol=1e-6)


def test_group_summary_separates_three_reward_states_and_diversity():
    from metrics import group_summary

    programs = [("SH1", "SH1", "SH1")] * 4 + [("REV", "SC2", "AX1")] * 4
    mixed = group_summary(programs, [False] * 4 + [True] * 4)
    assert mixed["group_class"] == "mixed"
    assert mixed["correct_count"] == 4
    assert mixed["unique_programs"] == 2
    assert mixed["unique_correct_programs"] == 1
    assert group_summary(programs, [False] * 8)["group_class"] == "all_wrong"
    assert group_summary(programs, [True] * 8)["group_class"] == "all_correct"


def test_exact_ranking_rejects_missing_duplicate_or_nonfinite_program():
    from metrics import exact_ranking_metrics, all_programs

    programs = all_programs()
    rows = [(program, float(-index), index == 0) for index, program in enumerate(programs)]
    summary, ranked = exact_ranking_metrics(rows)
    assert len(ranked) == 125
    assert summary["hit@32"] == 1.0
    assert summary["best_rank"] == 1
    assert 0 < summary["correct_mass"] < 1
    with pytest.raises(ValueError, match="125"):
        exact_ranking_metrics(rows[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        exact_ranking_metrics(rows[:-1] + [rows[0]])
    with pytest.raises(ValueError, match="finite"):
        exact_ranking_metrics(rows[:-1] + [(programs[-1], float("nan"), False)])


def test_pair_config_must_differ_only_by_entropy_coefficient():
    from metrics import validate_pair_config

    common = {"seed": 0, "steps": 150, "input_sha256": "a", "base_sha256": "b",
              "adapter_sha256": "c", "gpu_uuid": "GPU-1", "group_size": 8}
    validate_pair_config({**common, "entropy_coef": 0.0}, {**common, "entropy_coef": 0.01})
    with pytest.raises(ValueError, match="unequal"):
        validate_pair_config({**common, "entropy_coef": 0.0},
                             {**common, "steps": 149, "entropy_coef": 0.01})
