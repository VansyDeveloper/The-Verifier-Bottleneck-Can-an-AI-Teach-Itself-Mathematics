from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from vbexp.experiment import RunDirectory, seed_everything, sha256_adapter, sha256_file
from vbexp.io import read_tasks
from vbexp.modeling import load_model, load_tokenizer
from vbexp.prompts import build_prompt
from vbexp.verifier import verify


def answer_line(text: str, mode: str) -> str:
    prefix = "RESULT:" if mode == "apply" else "PROGRAM:"
    for line in text.splitlines():
        if line.strip().upper().startswith(prefix):
            return line.strip()
    return text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--label", default="baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name = "Qwen/Qwen3-0.6B-Base"
    resolved = {
        "source_config": str(args.config),
        "model_name_or_path": model_name,
        "adapter": str(args.adapter) if args.adapter else None,
        "adapter_sha256": sha256_adapter(args.adapter) if args.adapter else None,
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "k": args.k,
        "seed": args.seed,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "label": args.label,
    }
    run = RunDirectory.create(
        kind="evaluation",
        model_tag="qwen3-0.6b",
        task=args.input.stem,
        method=args.label,
        verifier="exact",
        seed=args.seed,
        config=resolved,
    )
    started = time.perf_counter()
    try:
        seed_everything(args.seed)
        tokenizer = load_tokenizer(model_name)
        model = load_model(model_name, adapter=args.adapter)
        model.eval()
        tasks = read_tasks(args.input)
        task_success = []
        total_correct = total_parse = total_candidates = total_tokens = 0
        for task in tasks:
            prompt = build_prompt(task) + "\n"
            encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
            before = time.perf_counter()
            with torch.inference_mode():
                generation_args = dict(
                    **encoded,
                    num_return_sequences=args.k,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
                if args.temperature > 0:
                    generation_args.update(do_sample=True, temperature=args.temperature, top_p=1.0)
                else:
                    generation_args.update(do_sample=False)
                output = model.generate(**generation_args)
            elapsed = time.perf_counter() - before
            records = []
            correct_flags = []
            for candidate_id, sequence in enumerate(output):
                completion_ids = sequence[encoded.input_ids.shape[1] :]
                raw_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                text = answer_line(raw_text, task.mode)
                result = verify(task, text)
                tokens = int(completion_ids.numel())
                total_tokens += tokens
                total_parse += int(result.parse_ok)
                total_correct += int(result.is_correct)
                total_candidates += 1
                correct_flags.append(result.is_correct)
                records.append(
                    {
                        "run_id": run.run_id,
                        "task_id": task.task_id,
                        "candidate_id": candidate_id,
                        "text": text,
                        "raw_text": raw_text,
                        "parsed_program": list(result.parsed_program) if result.parsed_program is not None else None,
                        "parsed_result": list(result.parsed_result) if result.parsed_result is not None else None,
                        "parse_ok": result.parse_ok,
                        "is_correct": result.is_correct,
                        "accepted": result.is_correct,
                        "alpha": 1.0,
                        "beta": 0.0,
                        "prompt_tokens": int(encoded.input_ids.numel()),
                        "completion_tokens": tokens,
                        "elapsed_seconds": elapsed / args.k,
                        "sampling": {"temperature": args.temperature, "top_p": 1.0},
                    }
                )
            run.generations(records)
            task_success.append(any(correct_flags))
        metrics = {
            f"pass_at_{args.k}": sum(task_success) / len(task_success),
            "candidate_correct_rate": total_correct / total_candidates,
            "parse_rate": total_parse / total_candidates,
            "tasks": len(tasks),
            "candidates": total_candidates,
            "actual_completion_tokens": total_tokens,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        }
        run.metric(args.input.stem, 0, metrics)
        run.finish(metrics)
        print(json.dumps({"run_id": run.run_id, **metrics}, ensure_ascii=False, indent=2))
    except BaseException as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
