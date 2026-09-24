"""Matched IID group-normalized policy-gradient training with optional entropy.

This is the original categorical action-policy objective plus a normalized
operation-choice entropy bonus. It is not clipped GRPO.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from train_action_grpo import OPERATIONS, operation_scores, sample_group  # noqa: E402
from vbexp.experiment import seed_everything  # noqa: E402
from vbexp.io import read_tasks  # noqa: E402
from vbexp.modeling import load_tokenizer  # noqa: E402
from vbexp.verifier import verify  # noqa: E402

from metrics import group_summary, normalized_entropy  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((path for path in root.rglob("*") if path.is_file()),
                   key=lambda item: item.relative_to(root).as_posix())
    if not paths:
        raise ValueError(f"empty adapter tree: {root}")
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest().upper()


def reserve_run_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False,
                                    separators=(",", ":")) + "\n")
        stream.flush()


def candidate_objective(log_probability: torch.Tensor, kl: torch.Tensor,
                        entropies: list[torch.Tensor], advantage: float,
                        kl_beta: float, entropy_coef: float) -> torch.Tensor:
    base = -float(advantage) * log_probability + kl_beta * kl
    if not entropies:
        return base
    return base - entropy_coef * torch.stack(entropies).mean()


def load_fp32_model(base_model: str | Path, adapter: str | Path, trainable: bool):
    base = AutoModelForCausalLM.from_pretrained(
        str(base_model), dtype=torch.float32, trust_remote_code=True,
    ).cuda()
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=trainable)
    if trainable:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model


def assigned_gpu(assigned_uuid: str) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("one assigned CUDA device is required")
    name = torch.cuda.get_device_name(0)
    if "RTX 2080 Ti" not in name:
        raise RuntimeError(f"GRPO entropy protocol requires RTX 2080 Ti: {name}")
    result = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                            capture_output=True, text=True, check=True)
    uuids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if uuids != [assigned_uuid]:
        raise RuntimeError(f"GPU UUID differs from assignment: {uuids}")
    return {"name": name, "uuid": assigned_uuid, "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "precision": "float32"}


def train(args: argparse.Namespace) -> dict:
    run_dir = args.output_root / args.run_name
    reserve_run_directory(run_dir)
    started = time.time()
    write_json(run_dir / "RUNNING.json", {"status": "RUNNING", "started_at_unix": started,
                                           "run_name": args.run_name})
    try:
        gpu = assigned_gpu(args.gpu_uuid)
        actual_input = sha256(args.input)
        actual_base = sha256(args.base_model / "model.safetensors")
        actual_adapter = tree_sha256(args.adapter)
        expected = (args.expected_input_sha256.upper(), args.expected_base_sha256.upper(),
                    args.expected_adapter_sha256.upper())
        if (actual_input, actual_base, actual_adapter) != expected:
            raise RuntimeError("frozen input, base model, or adapter SHA-256 mismatch")
        if args.group_size != 8 or args.method != "iid_action" or args.temperature != 0.7:
            raise RuntimeError("training configuration differs from frozen comparison")
        if args.entropy_coef not in (0.0, 0.01):
            raise RuntimeError("entropy coefficient must be 0 or 0.01")
        config = {
            "algorithm": "group_normalized_categorical_policy_gradient_without_clipping",
            "seed": args.seed, "steps": args.steps, "group_size": args.group_size,
            "method": args.method, "temperature": args.temperature,
            "learning_rate": args.learning_rate, "kl_beta": args.kl_beta,
            "compute_dtype": "torch.float32", "reference_dtype": "torch.float32",
            "gradient_checkpointing": "non_reentrant_trainable_model",
            "entropy_coef": args.entropy_coef, "entropy_normalization": "ln(5)",
            "input_sha256": actual_input, "base_sha256": actual_base,
            "adapter_sha256": actual_adapter, "gpu_uuid": args.gpu_uuid,
            "code_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in (
                Path(__file__), Path(__file__).with_name("metrics.py"),
                ROOT / "scripts/train_action_grpo.py", ROOT / "src/vbexp/modeling.py")},
        }
        write_json(run_dir / "config.json", config)
        write_json(run_dir / "environment.json", gpu)
        seed_everything(args.seed)
        rng = random.Random(args.seed)
        tasks = read_tasks(args.input)
        if not tasks or len({task.task_id for task in tasks}) != len(tasks):
            raise RuntimeError("training task list is empty or has duplicates")
        tokenizer = load_tokenizer(str(args.base_model))
        model = load_fp32_model(args.base_model, args.adapter, trainable=True)
        reference = load_fp32_model(args.base_model, args.adapter, trainable=False)
        if next(model.parameters()).dtype != torch.float32:
            raise RuntimeError("training model must be FP32")
        reference.eval()
        optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad),
                                      lr=args.learning_rate)
        reached_tasks: set[str] = set()
        correct_programs: set[tuple[str, ...]] = set()
        for step in range(args.steps):
            step_started = time.time()
            task = tasks[step % len(tasks)]
            programs = sample_group(model, tokenizer, task, args.method,
                                    args.group_size, args.temperature, rng)
            rewards = []
            generation_rows = []
            for candidate_id, (program, forced) in enumerate(programs):
                result = verify(task, "PROGRAM: " + " ".join(program))
                if not result.parse_ok or forced is not None:
                    raise RuntimeError("IID sampling or exact verification violated")
                rewards.append(bool(result.is_correct))
                generation_rows.append({"step": step + 1, "task_id": task.task_id,
                                        "candidate_id": candidate_id, "program": list(program),
                                        "correct": bool(result.is_correct)})
                if result.is_correct:
                    reached_tasks.add(task.task_id)
                    correct_programs.add(tuple(program))
            append_jsonl(run_dir / "generations.jsonl", generation_rows)
            mean_reward = sum(rewards) / len(rewards)
            variance = sum((float(reward) - mean_reward) ** 2 for reward in rewards) / len(rewards)
            scale = max(math.sqrt(variance), 1e-6)
            advantages = [(float(reward) - mean_reward) / scale for reward in rewards]
            optimizer.zero_grad(set_to_none=True)
            model.train()
            losses = []
            kls = []
            policy_entropies = []
            for (program, forced), advantage in zip(programs, advantages):
                log_probability = torch.zeros((), device=model.device)
                kl = torch.zeros((), device=model.device)
                entropies = []
                prefix = []
                for index, action in enumerate(program):
                    if index == 0 and forced is not None:
                        prefix.append(action)
                        continue
                    current_scores = operation_scores(model, tokenizer, task, prefix) / args.temperature
                    with torch.no_grad():
                        reference_scores = operation_scores(reference, tokenizer, task, prefix) / args.temperature
                    current_log = torch.log_softmax(current_scores, dim=0)
                    reference_log = torch.log_softmax(reference_scores, dim=0)
                    current_prob = current_log.exp()
                    action_index = OPERATIONS.index(action)
                    log_probability = log_probability + current_log[action_index]
                    kl = kl + (current_prob * (current_log - reference_log)).sum()
                    entropies.append(normalized_entropy(current_scores))
                    prefix.append(action)
                loss = candidate_objective(log_probability, kl, entropies, advantage,
                                           args.kl_beta, args.entropy_coef)
                if not bool(torch.isfinite(loss).item()):
                    raise RuntimeError("non-finite policy loss")
                (loss / len(programs)).backward()
                losses.append(float(loss.detach().cpu()))
                kls.append(float(kl.detach().cpu()))
                policy_entropies.extend(float(value.detach().cpu()) for value in entropies)
            norm = torch.nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad], 1.0)
            if not bool(torch.isfinite(norm).item()):
                raise RuntimeError("non-finite gradient norm")
            optimizer.step()
            group = group_summary([program for program, _ in programs], rewards)
            metric = {"step": step + 1, "task_id": task.task_id, **group,
                      "mean_reward": mean_reward, "loss": sum(losses) / len(losses),
                      "mean_kl": sum(kls) / len(kls),
                      "mean_policy_entropy_normalized": sum(policy_entropies) / len(policy_entropies),
                      "cumulative_tasks_reached": len(reached_tasks),
                      "cumulative_unique_correct_programs": len(correct_programs),
                      "wall_seconds": time.time() - step_started}
            append_jsonl(run_dir / "metrics.jsonl", [metric])
            if (step + 1) % 25 == 0 or step + 1 == args.steps:
                print(json.dumps({"status": "TRAINING", "run_name": args.run_name,
                                  "step": step + 1, "elapsed_seconds": time.time() - started}), flush=True)
        adapter_out = run_dir / "final_adapter"
        model.save_pretrained(adapter_out)
        final = {"status": "DONE", "run_name": args.run_name, "steps": args.steps,
                 "groups": args.steps, "answers": args.steps * args.group_size,
                 "adapter_tree_sha256": tree_sha256(adapter_out),
                 "metrics_sha256": sha256(run_dir / "metrics.jsonl"),
                 "generations_sha256": sha256(run_dir / "generations.jsonl"),
                 "elapsed_seconds": time.time() - started,
                 "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
                 "finished_at_unix": time.time(), "config": config}
        write_json(run_dir / "DONE.json", final)
        print(json.dumps({"status": "DONE", "run_name": args.run_name,
                          "elapsed_seconds": final["elapsed_seconds"]}), flush=True)
        return final
    except BaseException as exc:
        write_json(run_dir / "FAILED.json", {"status": "FAILED", "run_name": args.run_name,
                                             "error_type": type(exc).__name__, "error": str(exc),
                                             "traceback": traceback.format_exc(),
                                             "failed_at_unix": time.time()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--entropy-coef", type=float, required=True)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--method", default="iid_action")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--kl-beta", type=float, default=0.01)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
