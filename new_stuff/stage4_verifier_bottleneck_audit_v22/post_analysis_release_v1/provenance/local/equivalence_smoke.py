from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

import torch


REPO = Path("/workspace/repo")
CONFIRM_CODE = REPO / "artifacts/stage4_composition_confirm_v1_0p6b/code"
sys.path.insert(0, str(CONFIRM_CODE))

import confirm  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    expected_path = Path("/workspace/smoke/expected.jsonl.gz")
    expected = read_jsonl(expected_path)[0]
    rows = confirm.read_jsonl(confirm.DATA / "final_b_depth4.jsonl")
    task = next(row for row in rows if row["task_id"] == expected["task_id"])
    adapter = confirm._adapter(0, "atomic_control")
    model, tokenizer, token_ids = confirm.model_lib.load_branch(confirm.EXPORT, adapter)
    try:
        metric, actual = confirm.eval_lib.score_task(
            model, tokenizer, token_ids, task, batch_size=confirm.CONFIG["ranking"]["batch_size"]
        )
    finally:
        confirm.eval_lib.unload(model)
    expected_candidates = expected["ranking"]
    actual_candidates = actual["ranking"]
    expected_programs = [item["program"] for item in expected_candidates]
    actual_programs = [item["program"] for item in actual_candidates]
    expected_correct = [item["correct"] for item in expected_candidates]
    actual_correct = [item["correct"] for item in actual_candidates]
    expected_by_program = {tuple(item["program"]): item for item in expected_candidates}
    actual_by_program = {tuple(item["program"]): item for item in actual_candidates}
    max_score_abs_diff = max(
        abs(float(expected_by_program[program]["score"]) - float(actual_by_program[program]["score"]))
        for program in expected_by_program
    )
    correct_rank_map_exact = all(
        expected_by_program[program]["rank"] == actual_by_program[program]["rank"]
        for program in expected_by_program
        if expected_by_program[program]["correct"]
    )
    mismatch_positions = sum(left != right for left, right in zip(expected_programs, actual_programs))
    expected_metrics = confirm.eval_lib.ranking_metrics(expected_candidates)
    rank_metrics = ("hit@1", "hit@8", "hit@16", "hit@32", "hit@64", "best_rank", "mrr")
    rank_metrics_equal = all(metric[key] == expected_metrics[key] for key in rank_metrics)
    result = {
        "schema": "stage4.ai01.cross-environment-equivalence.v1",
        "status": "PASS" if (
            expected_programs == actual_programs
            and expected_correct == actual_correct
            and rank_metrics_equal
            and correct_rank_map_exact
        ) else "FAIL",
        "task_id": task["task_id"],
        "depth": task["depth"],
        "candidate_count": len(actual_candidates),
        "program_order_exact": expected_programs == actual_programs,
        "mismatch_positions": mismatch_positions,
        "correct_flags_exact": expected_correct == actual_correct,
        "correct_program_rank_map_exact": correct_rank_map_exact,
        "rank_metrics_exact": rank_metrics_equal,
        "expected_rank_metrics": {key: expected_metrics[key] for key in rank_metrics},
        "actual_rank_metrics": {key: metric[key] for key in rank_metrics},
        "max_score_abs_diff": max_score_abs_diff,
        "expected_correct_mass": expected_metrics["correct_mass"],
        "actual_correct_mass": metric["correct_mass"],
        "expected_log_gap": expected_metrics["log_gap"],
        "actual_log_gap": metric["log_gap"],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "pid": os.getpid(),
    }
    out = Path("/workspace/smoke/remote_equivalence.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
