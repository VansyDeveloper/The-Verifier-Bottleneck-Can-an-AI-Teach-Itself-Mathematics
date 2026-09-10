from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "code/composition_stats.py"
SPEC = importlib.util.spec_from_file_location("composition_stats", PATH)
STATS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(STATS)


def metric_row(
    task_id: str,
    *,
    seed: int = 0,
    split: str = "final_a",
    branch: str = "baseline",
    hit32: float = 0.0,
    mass: float = 0.2,
    log_gap: float = 1.0,
) -> dict:
    return {
        "task_id": task_id,
        "seed": seed,
        "split": split,
        "branch": branch,
        "hit@1": 0.0,
        "hit@8": 0.0,
        "hit@16": 0.0,
        "hit@32": hit32,
        "hit@64": hit32,
        "correct_mass": mass,
        "best_rank": 4,
        "mrr": 0.25,
        "log_gap": log_gap,
    }


def test_aggregate_metrics_analytical_fixture_and_nonfinite_log_gap() -> None:
    rows = [
        metric_row("a", hit32=0.0, mass=0.2, log_gap=1.0),
        {**metric_row("b", hit32=1.0, mass=0.4, log_gap=math.inf), "best_rank": 2, "mrr": 0.5},
    ]
    summary = STATS.aggregate_metrics(rows)
    assert summary["n_tasks"] == 2
    assert summary["hit@32"] == pytest.approx(0.5)
    assert summary["correct_mass"] == pytest.approx(0.3)
    assert summary["best_rank"] == pytest.approx(3.0)
    assert summary["mrr"] == pytest.approx(0.375)
    assert summary["log_gap"] == pytest.approx(1.0)
    assert summary["finite_counts"]["log_gap"] == 1


def test_strict_join_uses_task_id_not_row_position() -> None:
    baseline = [metric_row("a", hit32=0.0), metric_row("b", hit32=1.0)]
    treatment = [
        metric_row("b", branch="distill", hit32=1.0),
        metric_row("a", branch="distill", hit32=1.0),
    ]
    assert STATS.paired_task_differences(baseline, treatment) == pytest.approx(
        {"a": 1.0, "b": 0.0}
    )


@pytest.mark.parametrize("side", ["baseline", "treatment"])
def test_strict_join_rejects_duplicates(side: str) -> None:
    baseline = [metric_row("a")]
    treatment = [metric_row("a", branch="distill")]
    if side == "baseline":
        baseline.append(metric_row("a"))
    else:
        treatment.append(metric_row("a", branch="distill"))
    with pytest.raises(ValueError, match="duplicate task_id"):
        STATS.strict_join_by_task_id(baseline, treatment)


def test_strict_join_rejects_missing_and_context_mismatch() -> None:
    with pytest.raises(ValueError, match="task_id sets differ"):
        STATS.strict_join_by_task_id(
            [metric_row("a"), metric_row("b")],
            [metric_row("a", branch="distill")],
        )
    with pytest.raises(ValueError, match="seed mismatch"):
        STATS.strict_join_by_task_id(
            [metric_row("a", seed=0)],
            [metric_row("a", seed=1, branch="distill")],
        )


def test_flat_seed_pairing_rejects_multiple_unselected_splits() -> None:
    baseline = [metric_row("a", split="a"), metric_row("b", split="b")]
    treatment = [
        metric_row("a", split="a", branch="distill"),
        metric_row("b", split="b", branch="distill"),
    ]
    with pytest.raises(ValueError, match="multiple splits"):
        STATS.paired_seed_task_differences(baseline, treatment)


def test_student_t_statistics_match_known_six_seed_fixture() -> None:
    differences = [0.05, 0.07, 0.09, 0.11, 0.13, 0.15]
    result = STATS.six_seed_statistics(differences)
    expected_se = math.sqrt(0.0014) / math.sqrt(6)
    expected_half_width = 2.570581835636314 * expected_se
    assert result["mean_delta"] == pytest.approx(0.1)
    assert result["standard_error"] == pytest.approx(expected_se)
    assert result["t_ci95"] == pytest.approx(
        [0.1 - expected_half_width, 0.1 + expected_half_width]
    )
    assert result["t"] == pytest.approx(0.1 / expected_se)
    assert result["p_two_sided"] < 0.002
    assert result["positive_seeds"] == 6


def test_zero_variance_student_cases_are_explicit() -> None:
    positive = STATS.six_seed_statistics([0.06] * 6)
    assert positive["t"] is None
    assert positive["t_nonfinite"] == "positive_infinity"
    assert positive["zero_variance"] is True
    assert positive["p_two_sided"] == 0.0
    assert positive["t_ci95"] == pytest.approx([0.06, 0.06])
    zero = STATS.six_seed_statistics([0.0] * 6)
    assert zero["t"] == 0.0
    assert zero["p_two_sided"] == 1.0


def test_exact_sign_flip_has_analytical_probability_and_is_deterministic() -> None:
    differences = [1.0] * 6
    assert STATS.exact_sign_flip(differences) == pytest.approx(2 / 64)
    assert STATS.exact_sign_flip(differences) == STATS.exact_sign_flip(differences)


def test_two_level_bootstrap_20000_is_seed_deterministic() -> None:
    seed_tasks = {
        seed: {"a": 0.01 * (seed + 1), "b": 0.01 * (seed + 1)}
        for seed in range(6)
    }
    first = STATS.hierarchical_bootstrap(seed_tasks, seed=17)
    second = STATS.hierarchical_bootstrap(seed_tasks, seed=17)
    assert first == second
    assert first["repetitions"] == 20_000
    assert first["observed_mean"] == pytest.approx(0.035)
    assert first["ci95"][0] < first["observed_mean"] < first["ci95"][1]


def test_holm_step_down_analytical_fixture() -> None:
    adjusted = STATS.holm({"a": 0.01, "b": 0.03, "c": 0.04})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_confirmation_criteria_pass_boundaries_and_failures() -> None:
    passed = STATS.confirmation_criteria(
        [0.06] * 6, forgetting=0.02, mass_growth=0.001
    )
    assert passed["pass"] is True

    forgetting_failure = STATS.confirmation_criteria(
        [0.06] * 6, forgetting=0.020001, mass_growth=0.001
    )
    assert forgetting_failure["pass"] is False

    mass_failure = STATS.confirmation_criteria(
        [0.06] * 6, forgetting=0.0, mass_growth=0.0
    )
    assert mass_failure["checks"]["mass_growth_gt_0"] is False

    separate = STATS.criteria_a_b(
        [0.06] * 6,
        [0.06] * 6,
        forgetting_a=0.01,
        mass_growth_a=0.001,
        forgetting_b=0.01,
        mass_growth_b=0.0,
    )
    assert separate["A"]["pass"] is True
    assert separate["B"]["pass"] is False
    assert separate["pass_both"] is False


def test_confirmation_analysis_pairs_shuffled_flat_rows_end_to_end() -> None:
    baseline = []
    treatment = []
    for seed in range(6):
        for task_number in range(100):
            task_id = f"task-{task_number:03d}"
            baseline_hit = float(task_number < 20)
            treatment_hit = float(task_number < 26)
            baseline.append(metric_row(task_id, seed=seed, hit32=baseline_hit, mass=0.20))
            treatment.insert(
                0,
                metric_row(
                    task_id,
                    seed=seed,
                    branch="composition_distill",
                    hit32=treatment_hit,
                    mass=0.21,
                ),
            )
    result = STATS.analyze_confirmation(
        baseline,
        treatment,
        forgetting=0.01,
        split="final_a",
        secondary_pvalues={"transfer_b": 0.01, "depth4": 0.04},
        bootstrap_repetitions=500,
        bootstrap_seed=9,
    )
    assert result["seed_statistics"]["mean_delta"] == pytest.approx(0.06)
    assert result["mass_growth"] == pytest.approx(0.01)
    assert result["exact_sign_flip_p_two_sided"] == pytest.approx(2 / 64)
    assert result["criteria"]["pass"] is True
    assert result["secondary_holm"] == pytest.approx(
        {"transfer_b": 0.02, "depth4": 0.04}
    )


def test_seed_can_be_read_from_evaluation_binding() -> None:
    baseline = [{**metric_row("a"), "binding": {"seed": 0}}]
    treatment = [{**metric_row("a", branch="distill"), "binding": {"seed": 0}}]
    baseline[0].pop("seed")
    treatment[0].pop("seed")
    assert STATS.paired_seed_task_differences(baseline, treatment) == {0: {"a": 0.0}}


def test_contradictory_top_level_and_bound_seed_is_rejected() -> None:
    baseline = [{**metric_row("a", seed=0), "binding": {"seed": 1}}]
    treatment = [{**metric_row("a", seed=0, branch="distill"), "binding": {"seed": 0}}]
    with pytest.raises(ValueError, match="contradictory"):
        STATS.paired_seed_task_differences(baseline, treatment)


def test_failure_cases_for_seed_count_nonfinite_and_invalid_pvalue() -> None:
    with pytest.raises(ValueError, match="exactly six"):
        STATS.six_seed_statistics([0.1] * 5)
    with pytest.raises(ValueError, match="finite"):
        STATS.seed_statistics([0.1, math.nan])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        STATS.holm({"bad": 1.1})
