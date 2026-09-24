import json

import pytest
import torch


def test_exact_ranker_scores_all_125_legacy_action_programs(monkeypatch):
    import run_evaluation
    from vbexp.polynomial import apply_program
    from vbexp.task import Task

    monkeypatch.setattr(run_evaluation, "operation_scores",
                        lambda model, tokenizer, task, prefix: torch.tensor([-2., -2., -2., -2., 5.]))
    start = (1, 2, 3)
    target = apply_program(start, ("AX1", "AX1", "AX1"), 11)
    task = Task(task_id="t", mode="plan", split="holdout", p=11, degree_cap=2,
                start=start, target=target, operations=("SH1", "SC2", "REV", "AC1", "AX1"),
                max_steps=3)
    summary, ranked = run_evaluation.score_task(None, None, task, temperature=0.7)
    assert len(ranked) == 125
    assert summary["best_rank"] == 1
    assert ranked[0]["program"] == ["AX1", "AX1", "AX1"]


def test_cached_shard_receipt_rejects_wrong_hash(tmp_path):
    from run_evaluation import verify_cached_shard

    data = tmp_path / "part-000.jsonl"
    receipt = tmp_path / "part-000.receipt.json"
    data.write_text('{"task_id":"t"}\n', encoding="utf-8")
    receipt.write_text(json.dumps({"status": "DONE", "sha256": "bad", "tasks": 1,
                                   "task_ids": ["t"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_cached_shard(data, receipt, ["t"])
