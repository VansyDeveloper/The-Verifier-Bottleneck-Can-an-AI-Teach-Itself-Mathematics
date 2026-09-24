"""Separate V100 FP16 training implementation for withheld-pair study."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup


def fp16_runtime(resolved):
    if resolved.get("dtype") != "float16":
        raise RuntimeError("withheld-pair V100 training requires float16")
    if not torch.cuda.is_available():
        raise RuntimeError("withheld-pair training requires CUDA")
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    if "V100" not in device_name:
        raise RuntimeError(f"withheld-pair FP16 protocol requires V100, got {device_name}")
    return {"runtime_device": "cuda", "runtime_dtype": "torch.float16",
            "runtime_compute_dtype": "torch.float16", "runtime_master_dtype": "torch.float32",
            "runtime_autocast_enabled": True, "cuda_device_name": device_name,
            "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda}


def load_atomic_for_fp16(model_lib, export_dir: Path, resolved: dict):
    fp16_runtime(resolved)
    receipt = json.loads((export_dir / "export_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("status") != "DONE" or receipt.get("model") != model_lib.PROTOCOL["model"]:
        raise RuntimeError("atomic export receipt invalid")
    if receipt.get("export_tree_sha256") != model_lib._export_payload_sha(export_dir):
        raise RuntimeError("atomic export hash mismatch")
    tokenizer, token_ids = model_lib.tokenizer_and_ids(export_dir)
    model = AutoModelForCausalLM.from_pretrained(
        str(export_dir), dtype=torch.float32, trust_remote_code=True, low_cpu_mem_usage=True)
    cfg = resolved["lora"]
    model = get_peft_model(model, LoraConfig(
        r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"],
        bias="none", task_type="CAUSAL_LM", target_modules=cfg["target_modules"],
        trainable_token_indices=sorted(token_ids.values()), ensure_weight_tying=True))
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.cuda()
    return model, tokenizer


def train_branch_fp16(model_lib, export_dir: Path, output: Path, encoded: list[dict],
                      resolved: dict, branch: str, input_hash: str) -> dict:
    output = Path(output)
    export_dir = Path(export_dir)
    runtime = fp16_runtime(resolved)
    receipt_path = output / "training_receipt.json"
    resolved_sha = hashlib.sha256(json.dumps(resolved, sort_keys=True, separators=(",", ":"),
                                            ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    target_tokens = sum(int(row["target_tokens"]) for row in encoded)
    input_tokens = sum(int(row["total_tokens"]) for row in encoded)
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in encoded)
    expected = {"branch": branch, "seed": resolved["seed"], "lr": resolved["lr"],
                "epochs": resolved["epochs"], "effective_batch": resolved["effective_batch"],
                "records": len(encoded), "input_sha256": input_hash,
                "atomic_export_sha256": model_lib._export_payload_sha(export_dir),
                "resolved_config_sha256": resolved_sha,
                "loss_bearing_target_tokens_per_epoch": target_tokens,
                "loss_bearing_target_tokens_expected": target_tokens * int(resolved["epochs"]),
                "input_tokens_per_epoch": input_tokens, "prompt_tokens_per_epoch": prompt_tokens,
                "grad_scaler_initial_scale": 128.0, **runtime}
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") == "DONE" and all(receipt.get(key) == value for key, value in expected.items()):
            return receipt
        raise RuntimeError(f"mismatched completed adapter: {output}")
    if output.exists() or output.with_name(output.name + ".partial").exists():
        raise RuntimeError(f"partial adapter requires audit: {output}")
    partial = output.with_name(output.name + ".partial")
    model_lib.seed_all(resolved["seed"])
    model, tokenizer = load_atomic_for_fp16(model_lib, export_dir, resolved)
    model.train()
    try:
        if next(model.parameters()).dtype != torch.float32:
            raise RuntimeError("master model weights are not FP32")
        if {parameter.dtype for parameter in model.parameters() if parameter.requires_grad} != {torch.float32}:
            raise RuntimeError("trainable master weights are not FP32")
        packed_attention = model_lib.packed_attention_runtime_guard(model)
        micro = int(resolved["micro_batch"])
        effective = int(resolved["effective_batch"])
        if effective % micro:
            raise RuntimeError("effective batch is not divisible by micro batch")
        accumulation = effective // micro
        loader = DataLoader(model_lib.EncodedRows(encoded), batch_size=micro, shuffle=True,
                            collate_fn=lambda rows: model_lib.collate_packed(tokenizer, rows),
                            generator=torch.Generator().manual_seed(resolved["seed"]))
        steps_per_epoch = math.ceil(len(loader) / accumulation)
        total_steps = steps_per_epoch * int(resolved["epochs"])
        optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=resolved["lr"])
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, max(1, round(total_steps * resolved["warmup_ratio"])), total_steps)
        scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
        status_path = model_lib._training_status_path(output)
        status = {"schema": "withheld-pairs.fp16-training-status.v1", "status": "STARTED",
                  **expected, "optimizer_steps_expected": total_steps,
                  "started_at_unix": time.time()}
        model_lib.dump_json(status_path, status)
        started = time.time()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        optimizer_steps = 0
        seen_target_tokens = 0
        torch.cuda.reset_peak_memory_stats()
        try:
            for _epoch in range(int(resolved["epochs"])):
                for batch_index, batch in enumerate(loader):
                    seen_target_tokens += int((batch["labels"] != -100).sum())
                    batch = {key: value.cuda() for key, value in batch.items()}
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        raw_loss = model(**batch).loss
                    loss = float(raw_loss.detach())
                    if not math.isfinite(loss):
                        raise RuntimeError(f"non-finite FP16 loss at optimizer step {optimizer_steps}")
                    scaler.scale(raw_loss / accumulation).backward()
                    losses.append(loss)
                    if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), resolved["max_grad_norm"])
                        if not torch.isfinite(grad_norm):
                            raise RuntimeError(f"non-finite FP16 gradient at optimizer step {optimizer_steps}")
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        optimizer_steps += 1
                        if optimizer_steps % 10 == 0:
                            print(json.dumps({"branch": branch, "step": optimizer_steps,
                                              "steps": total_steps, "loss": loss}), flush=True)
            partial.mkdir(parents=True)
            model.save_pretrained(partial)
            tokenizer.save_pretrained(partial)
            receipt = {"schema": "withheld-pairs.fp16-training-receipt.v1", "status": "DONE",
                       **expected, "optimizer_steps": optimizer_steps,
                       "optimizer_steps_expected": total_steps,
                       "loss_bearing_target_tokens_seen": seen_target_tokens,
                       "mean_loss": sum(losses) / len(losses),
                       "input_tokens_seen": input_tokens * int(resolved["epochs"]),
                       "prompt_tokens_seen": prompt_tokens * int(resolved["epochs"]),
                       "forward_microbatches": len(loader) * int(resolved["epochs"]),
                       "wall_seconds": time.time() - started,
                       "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                       "grad_scaler_final_scale": scaler.get_scale(),
                       "packed_attention": packed_attention,
                       "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
            model_lib.dump_json(partial / "training_receipt.json", receipt)
            partial.replace(output)
            status.update({"status": "DONE", "finished_at_unix": time.time(),
                           "adapter_tree_sha256": model_lib.tree_sha256(output)})
            model_lib.dump_json(status_path, status)
            return receipt
        except Exception as exc:
            status.update({"status": "FAILED", "finished_at_unix": time.time(),
                           "error_type": type(exc).__name__, "error": str(exc)})
            model_lib.dump_json(status_path, status)
            raise
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
