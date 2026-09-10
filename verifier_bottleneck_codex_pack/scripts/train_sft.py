from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import torch

from vbexp.experiment import RunDirectory, load_yaml, seed_everything
from vbexp.io import read_tasks
from vbexp.modeling import (
    attach_lora,
    encode_supervised,
    load_model,
    load_tokenizer,
    pad_supervised,
    target_text,
)
from vbexp.prompts import build_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tier", choices=["smoke", "pilot", "full"], default="smoke")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--lora-config", type=Path)
    parser.add_argument("--apply-weight", type=int, default=1)
    parser.add_argument("--base-adapter", type=Path)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--output-name")
    parser.add_argument("--extra-train-file", type=Path, action="append", default=[])
    parser.add_argument(
        "--allow-composition",
        action="store_true",
        help=(
            "Permit depth>1 tasks in the training stream (D-021). The default SFT "
            "protocol forbids compositions; this flag exists so that a composition "
            "supervision run is an explicit, recorded choice rather than a silent one."
        ),
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        help=(
            "Split each optimisation step into micro-batches of this size (D-017). "
            "Losses are re-weighted by supervised-token count, so the accumulated "
            "gradient equals the full-batch gradient. Memory only; omit to keep the "
            "original single-forward behaviour."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_yaml(args.config)
    model_cfg = load_yaml(raw["model_config"])
    max_steps = args.max_steps or (20 if args.tier == "smoke" else 0)
    if max_steps <= 0:
        raise ValueError("pilot/full step derivation is not implemented; pass --max-steps explicitly")
    resolved = {
        **raw,
        "tier": args.tier,
        "seed": args.seed,
        "max_steps": max_steps,
        "model_name_or_path": model_cfg["model_name_or_path"],
        "train_files": [
            f"artifacts/data/{args.tier}/sft_train_apply.jsonl",
            f"artifacts/data/{args.tier}/sft_train_plan.jsonl",
        ],
        "lora_config": str(args.lora_config or raw["lora_config"]),
        "apply_weight": args.apply_weight,
        "base_adapter": str(args.base_adapter) if args.base_adapter else None,
        "learning_rate_resolved": args.learning_rate or float(raw["learning_rate"]),
        "extra_train_files": [str(path) for path in args.extra_train_file],
        "micro_batch_size": args.micro_batch_size,
        "allow_composition": args.allow_composition,
    }
    run = RunDirectory.create(
        kind="sft",
        model_tag="qwen3-0.6b",
        task="atomic",
        method="lora-r16",
        verifier="exact",
        seed=args.seed,
        config=resolved,
    )
    started = time.perf_counter()
    try:
        seed_everything(args.seed)
        tokenizer = load_tokenizer(model_cfg["model_name_or_path"])
        if args.base_adapter:
            model = load_model(model_cfg["model_name_or_path"], adapter=args.base_adapter, trainable=True)
            lora_resolved = load_yaml(resolved["lora_config"])
        else:
            model = load_model(model_cfg["model_name_or_path"])
            model, lora_resolved = attach_lora(model, resolved["lora_config"])
        resolved["lora_resolved"] = lora_resolved
        apply_tasks = read_tasks(resolved["train_files"][0])
        plan_tasks = read_tasks(resolved["train_files"][1])
        tasks = apply_tasks * args.apply_weight + plan_tasks
        for path in args.extra_train_file:
            tasks.extend(read_tasks(path))
        depths = sorted({task.difficulty.get("depth") for task in tasks})
        if not args.allow_composition:
            if depths != [1]:
                raise ValueError(
                    f"SFT data contains non-atomic tasks (depths {depths}); "
                    "pass --allow-composition only for a declared composition-supervision run"
                )
        elif depths == [1]:
            raise ValueError("--allow-composition passed but the data is atomic only")
        resolved["train_depths"] = depths
        random.Random(args.seed).shuffle(tasks)
        examples = [
            encode_supervised(tokenizer, build_prompt(task), target_text(task), int(raw["max_seq_length"]))
            for task in tasks
        ]
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=float(resolved["learning_rate_resolved"]),
        )
        start_step = 0
        if args.resume:
            state = torch.load(args.resume, map_location="cpu", weights_only=False)
            optimizer.load_state_dict(state["optimizer"])
            start_step = int(state["step"])
        model.train()
        losses: list[float] = []
        for step in range(start_step, max_steps):
            batch_items = [
                examples[(step * int(raw["per_device_train_batch_size"]) + offset) % len(examples)]
                for offset in range(min(int(raw["per_device_train_batch_size"]), len(examples)))
            ]
            micro = args.micro_batch_size or len(batch_items)
            chunks = [batch_items[i : i + micro] for i in range(0, len(batch_items), micro)]
            padded = [pad_supervised(tokenizer, chunk) for chunk in chunks]
            # Supervised-token counts weight each micro-batch so the accumulated
            # gradient is identical to one forward over the whole batch.
            weights = [int((item["labels"] != -100).sum()) for item in padded]
            total_tokens = sum(weights)
            optimizer.zero_grad(set_to_none=True)
            loss = 0.0
            for item, weight in zip(padded, weights):
                batch = {key: value.cuda() for key, value in item.items()}
                output = model(**batch)
                (output.loss * weight / total_tokens).backward()
                loss += float(output.loss.detach().cpu()) * weight / total_tokens
            optimizer.step()
            losses.append(loss)
            run.metric("sft_train", step + 1, {"loss": loss})
            run.log(f"step={step + 1} loss={loss:.6f}")
            if (step + 1) % int(raw["save_steps"]) == 0 or step + 1 == max_steps:
                checkpoint = run.path / "checkpoints" / f"step-{step + 1}.pt"
                torch.save({"step": step + 1, "optimizer": optimizer.state_dict()}, checkpoint)
        adapter_dir = run.path / "checkpoints" / "final_adapter"
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        rank = int(lora_resolved["r"])
        name = args.output_name or f"sft_atomic_r{rank}_{args.tier}_aw{args.apply_weight}"
        canonical = Path("artifacts/adapters") / name
        if canonical.exists():
            shutil.rmtree(canonical)
        shutil.copytree(adapter_dir, canonical)
        peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        metrics = {
            "steps": max_steps,
            "mean_loss": sum(losses) / len(losses),
            "last_loss": losses[-1],
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_bytes": peak,
            "adapter": str(canonical),
            "train_examples": len(tasks),
        }
        run.finish(metrics)
        print(json.dumps({"run_id": run.run_id, **metrics}, ensure_ascii=False, indent=2))
    except BaseException as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
