"""Blind FP32 exact-ranking speed probe on ten training tasks only."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def select_training_tasks(tasks: list, count: int = 10) -> list:
    selected = [task for task in tasks if task.max_steps == 3][:count]
    if len(selected) != count:
        raise ValueError("not enough depth-three training tasks")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()

    from run_ablation import assigned_gpu, load_fp32_model, sha256, tree_sha256
    from run_evaluation import score_task
    from vbexp.io import read_tasks
    from vbexp.modeling import load_tokenizer

    gpu = assigned_gpu(args.gpu_uuid)
    if (sha256(args.base_model / "model.safetensors") != args.expected_base_sha256.upper() or
            tree_sha256(args.adapter) != args.expected_adapter_sha256.upper() or
            sha256(args.input) != args.expected_input_sha256.upper()):
        raise ValueError("frozen speed-probe input SHA-256 mismatch")
    tasks = select_training_tasks(read_tasks(args.input))
    tokenizer = load_tokenizer(str(args.base_model))
    model = load_fp32_model(args.base_model, args.adapter, trainable=False)
    model.eval()
    durations = []
    for task in tasks:
        started = time.perf_counter()
        score_task(model, tokenizer, task, 0.7)
        durations.append(time.perf_counter() - started)
    print(json.dumps({"status": "DONE", "task_count": len(tasks),
                      "input_kind": "depth-three training tasks, not held-out tasks",
                      "task_seconds": durations, "total_seconds": sum(durations),
                      "gpu_uuid": gpu["uuid"], "precision": gpu["precision"]}), flush=True)


if __name__ == "__main__":
    main()
