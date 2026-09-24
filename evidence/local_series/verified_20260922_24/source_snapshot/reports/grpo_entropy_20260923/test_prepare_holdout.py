import pytest


def _task(task_id, split="heldout"):
    return {
        "task_id": task_id, "mode": "plan", "split": split, "p": 11,
        "degree_cap": 2, "start": [1, 2, 3], "target": [3, 2, 1],
        "max_steps": 3, "operations": ["SH1", "SC2", "REV", "AC1", "AX1"],
    }


def test_semantic_key_ignores_split_and_task_id():
    from prepare_holdout import semantic_key

    assert semantic_key(_task("a", "train")) == semantic_key(_task("b", "heldout"))


def test_holdout_validation_rejects_duplicate_and_leaked_tasks():
    from prepare_holdout import semantic_key, validate_holdout

    first = _task("a")
    duplicate = _task("b")
    with pytest.raises(ValueError, match="duplicate"):
        validate_holdout([first, duplicate], set(), expected_count=2)
    with pytest.raises(ValueError, match="leakage"):
        validate_holdout([first], {semantic_key(first)}, expected_count=1)


def test_generated_holdout_is_depth_three_and_has_no_forbidden_semantics():
    from prepare_holdout import build_holdout, semantic_key, validate_holdout

    forbidden = {semantic_key(_task("old"))}
    rows = build_holdout(forbidden, count=8, seed=20260923)
    validate_holdout(rows, forbidden, expected_count=8)
    assert {row["max_steps"] for row in rows} == {3}
    assert {row["p"] for row in rows} <= {11, 17}
