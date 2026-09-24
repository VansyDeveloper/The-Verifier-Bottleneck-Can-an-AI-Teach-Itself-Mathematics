"""Exact 125-program held-out ranking for the legacy GRPO action policy."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from train_action_grpo import OPERATIONS, operation_scores  # noqa: E402
from vbexp.io import read_tasks  # noqa: E402
from vbexp.modeling import load_tokenizer  # noqa: E402
from vbexp.verifier import verify  # noqa: E402

from metrics import exact_ranking_metrics  # noqa: E402
from run_ablation import assigned_gpu, load_fp32_model, sha256, tree_sha256, write_json  # noqa: E402


@torch.inference_mode()
def score_task(model, tokenizer, task, temperature: float) -> tuple[dict, list[dict]]:
    prefix_scores: dict[tuple[str, ...], float] = {(): 0.0}
    for _ in range(task.max_steps):
        next_scores = {}
        for prefix, prior in sorted(prefix_scores.items()):
            option_scores = operation_scores(model, tokenizer, task, list(prefix)) / temperature
            if not bool(torch.isfinite(option_scores).all().item()):
                raise RuntimeError("non-finite operation score")
            log_prob = torch.log_softmax(option_scores, dim=0).detach().cpu().tolist()
            for operation, value in zip(OPERATIONS, log_prob):
                next_scores[prefix + (operation,)] = prior + float(value)
        prefix_scores = next_scores
    candidates = []
    for program, score in prefix_scores.items():
        outcome = verify(task, "PROGRAM: " + " ".join(program))
        if not outcome.parse_ok:
            raise RuntimeError("exact verifier could not parse an enumerated program")
        candidates.append((program, score, bool(outcome.is_correct)))
    return exact_ranking_metrics(candidates)


def _read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def verify_cached_shard(data_path: Path, receipt_path: Path, task_ids: list[str]) -> list[dict]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "DONE" or receipt.get("tasks") != len(task_ids) or receipt.get("task_ids") != task_ids:
        raise ValueError("cached shard receipt differs from task assignment")
    if not data_path.exists() or sha256(data_path) != receipt.get("sha256"):
        raise ValueError("cached shard SHA-256 mismatch")
    rows = _read_jsonl(data_path)
    if [row.get("task_id") for row in rows] != task_ids:
        raise ValueError("cached shard task order mismatch")
    for row in rows:
        derived, _ = exact_ranking_metrics([
            (tuple(candidate["program"]), candidate["score"], candidate["correct"])
            for candidate in row["ranking"]
        ])
        if derived != row["metrics"]:
            raise ValueError("cached shard metric differs from raw ranking")
    return rows


def _write_shard(data_path: Path, receipt_path: Path, rows: list[dict]) -> None:
    data_path.parent.mkdir(parents=True, exist_ok=True)
    partial = data_path.with_name(data_path.name + ".partial")
    with gzip.open(partial, "wt", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"),
                                    allow_nan=False) + "\n")
    os.replace(partial, data_path)
    write_json(receipt_path, {"status": "DONE", "tasks": len(rows),
                              "task_ids": [row["task_id"] for row in rows],
                              "sha256": sha256(data_path)})


def evaluate(args: argparse.Namespace) -> dict:
    run_dir = args.output_root / args.run_name
    training_path = run_dir / "DONE.json"
    if not training_path.exists():
        raise RuntimeError("training run is not accepted DONE")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    config = training["config"]
    if config["gpu_uuid"] != args.gpu_uuid:
        raise RuntimeError("evaluation GPU differs from paired training")
    if sha256(args.holdout) != args.expected_holdout_sha256.upper():
        raise RuntimeError("frozen 1000-task holdout SHA-256 mismatch")
    if sha256(args.base_model / "model.safetensors") != config["base_sha256"]:
        raise RuntimeError("base model SHA-256 differs from training")
    adapter = run_dir / "final_adapter"
    if tree_sha256(adapter) != training["adapter_tree_sha256"]:
        raise RuntimeError("final adapter SHA-256 differs from training receipt")
    tasks = read_tasks(args.holdout)
    if len(tasks) != 1000 or len({task.task_id for task in tasks}) != 1000 or any(task.max_steps != 3 for task in tasks):
        raise RuntimeError("holdout is not 1000 unique depth-3 tasks")
    out = run_dir / "evaluation"
    if (out / "FAILED.json").exists():
        raise RuntimeError("prior failed evaluation requires separate audit")
    if (out / "DONE.json").exists():
        raise FileExistsError("accepted evaluation already exists")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    write_json(out / "RUNNING.json", {"status": "RUNNING", "started_at_unix": started,
                                      "run_name": args.run_name})
    try:
        gpu = assigned_gpu(args.gpu_uuid)
        write_json(out / "environment.json", gpu)
        tokenizer = load_tokenizer(str(args.base_model))
        model = load_fp32_model(args.base_model, adapter, trainable=False)
        model.eval()
        if next(model.parameters()).dtype != torch.float32:
            raise RuntimeError("evaluation model must be FP32")
        all_rows = []
        for index, offset in enumerate(range(0, len(tasks), args.shard_size)):
            shard_tasks = tasks[offset:offset + args.shard_size]
            task_ids = [task.task_id for task in shard_tasks]
            data_path = out / "shards" / f"part-{index:05d}.jsonl.gz"
            receipt_path = out / "shards" / f"part-{index:05d}.receipt.json"
            if receipt_path.exists():
                rows = verify_cached_shard(data_path, receipt_path, task_ids)
            else:
                if data_path.exists() or data_path.with_name(data_path.name + ".partial").exists():
                    raise RuntimeError(f"partial shard needs audit: {data_path}")
                rows = []
                for task in shard_tasks:
                    metrics, ranked = score_task(model, tokenizer, task, config["temperature"])
                    rows.append({"task_id": task.task_id, "metrics": metrics, "ranking": ranked})
                _write_shard(data_path, receipt_path, rows)
            all_rows.extend(rows)
            print(json.dumps({"status": "EVALUATING", "run_name": args.run_name,
                              "shards_done": index + 1, "tasks_done": len(all_rows)}), flush=True)
        if len(all_rows) != 1000:
            raise RuntimeError("evaluation task count incomplete")
        best_ranks = [row["metrics"]["best_rank"] for row in all_rows]
        summary = {
            "status": "DONE", "run_name": args.run_name, "tasks": len(all_rows),
            "candidate_program_scores": 125 * len(all_rows),
            "hit@32": sum(rank <= 32 for rank in best_ranks) / len(best_ranks),
            "hit@k": {str(k): sum(rank <= k for rank in best_ranks) / len(best_ranks)
                      for k in range(1, 126)},
            "mean_correct_mass": sum(row["metrics"]["correct_mass"] for row in all_rows) / len(all_rows),
            "mean_entropy_nats": sum(row["metrics"]["entropy_nats"] for row in all_rows) / len(all_rows),
            "mean_correct_entropy_nats": sum(row["metrics"]["correct_entropy_nats"] for row in all_rows) / len(all_rows),
            "holdout_sha256": args.expected_holdout_sha256.upper(),
            "adapter_tree_sha256": training["adapter_tree_sha256"],
            "gpu_uuid": args.gpu_uuid, "elapsed_seconds": time.time() - started,
        }
        write_json(out / "DONE.json", summary)
        print(json.dumps({"status": "DONE", "run_name": args.run_name,
                          "hit@32": summary["hit@32"]}), flush=True)
        return summary
    except BaseException as exc:
        write_json(out / "FAILED.json", {"status": "FAILED", "run_name": args.run_name,
                                           "error_type": type(exc).__name__, "error": str(exc),
                                           "traceback": traceback.format_exc(),
                                           "failed_at_unix": time.time()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--expected-holdout-sha256", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--shard-size", type=int, default=25)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
