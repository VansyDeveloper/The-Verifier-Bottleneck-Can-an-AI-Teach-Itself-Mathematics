import itertools
import math

import pytest


def test_group_audit_separates_wrong_mixed_and_correct():
    from audit_ablation import group_metrics

    programs = [("SH1", "SC2", "REV")] * 4 + [("REV", "SC2", "SH1")] * 4
    wrong = group_metrics(programs, [False] * 8)
    assert wrong["group_class"] == "all_wrong"
    mixed = group_metrics(programs, [False] * 4 + [True] * 4)
    assert mixed["group_class"] == "mixed"
    assert mixed["unique_correct_programs"] == 1
    assert mixed["collision_fraction"] == pytest.approx(12 / 28)
    assert group_metrics(programs, [True] * 8)["group_class"] == "all_correct"


def test_rank_audit_rejects_missing_program_false_correctness_and_nonfinite_score():
    from audit_ablation import audit_ranked_task
    from vbexp.polynomial import apply_program
    from vbexp.task import Task
    from vbexp.verifier import verify

    ops = ("SH1", "SC2", "REV", "AC1", "AX1")
    start = (1, 2, 3)
    target = apply_program(start, ("SH1", "SC2", "REV"), 11)
    task = Task("unit", "plan", "test", 11, 2, start, ops, target=target, max_steps=3)
    ranking = []
    for rank, program in enumerate(itertools.product(ops, repeat=3), 1):
        ranking.append({"program": list(program), "score": -float(rank), "rank": rank,
                        "correct": verify(task, "PROGRAM: " + " ".join(program)).is_correct})
    derived = audit_ranked_task(task, ranking)
    assert derived["best_rank"] >= 1
    assert math.isfinite(derived["entropy_nats"])
    with pytest.raises(ValueError, match="program"):
        audit_ranked_task(task, ranking[:-1] + [{**ranking[-1], "program": ranking[0]["program"]}])
    with pytest.raises(ValueError, match="correct"):
        audit_ranked_task(task, [{**ranking[0], "correct": not ranking[0]["correct"]}] + ranking[1:])
    with pytest.raises(ValueError, match="score"):
        audit_ranked_task(task, [{**ranking[0], "score": float("nan")}] + ranking[1:])


def test_three_paired_seeds_have_exact_signflip_floor():
    from audit_ablation import paired_stats

    summary = paired_stats([0.1, 0.2, 0.3])
    assert summary["n"] == 3
    assert summary["mean_difference"] == pytest.approx(0.2)
    assert summary["exact_signflip_two_sided_p"] == 0.25
