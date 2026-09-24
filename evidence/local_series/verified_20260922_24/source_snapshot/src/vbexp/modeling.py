from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .experiment import load_yaml


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(
    model_name: str,
    *,
    adapter: str | Path | None = None,
    trainable: bool = False,
):
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable)
    return model


def attach_lora(model, config_path: str | Path):
    raw = load_yaml(config_path)
    available = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    requested = list(raw["target_modules"])
    resolved = [name for name in requested if name in available]
    missing = sorted(set(requested) - set(resolved))
    if missing:
        raise ValueError(f"LoRA target modules missing from model: {missing}")
    config = LoraConfig(
        r=int(raw["r"]),
        lora_alpha=int(raw["lora_alpha"]),
        lora_dropout=float(raw["lora_dropout"]),
        bias=str(raw["bias"]),
        task_type=str(raw["task_type"]),
        target_modules=resolved,
    )
    return get_peft_model(model, config), {**raw, "target_modules_resolved": resolved}


def target_text(task) -> str:
    if task.mode == "apply":
        return "RESULT: [" + ", ".join(str(v) for v in task.target) + "]"
    witness = task.metadata.get("witness_program")
    if not witness:
        raise ValueError(f"PLAN task {task.task_id} has no witness")
    return "PROGRAM: " + " ".join(witness)


def encode_supervised(tokenizer, prompt: str, answer: str, max_length: int) -> dict[str, torch.Tensor]:
    prompt_ids = tokenizer(prompt + "\n", add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer + tokenizer.eos_token, add_special_tokens=False).input_ids
    full = (prompt_ids + answer_ids)[:max_length]
    prompt_length = min(len(prompt_ids), len(full))
    labels = [-100] * prompt_length + full[prompt_length:]
    return {
        "input_ids": torch.tensor(full, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def pad_supervised(tokenizer, examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].numel() for item in examples)
    input_ids = []
    labels = []
    attention = []
    for item in examples:
        length = item["input_ids"].numel()
        pad = max_len - length
        input_ids.append(torch.cat([torch.full((pad,), tokenizer.pad_token_id), item["input_ids"]]))
        labels.append(torch.cat([torch.full((pad,), -100), item["labels"]]))
        attention.append(torch.cat([torch.zeros(pad, dtype=torch.long), torch.ones(length, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention),
    }


def completion_log_probability(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, prompt_length: int):
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = output.logits[:, :-1].float()
    targets = input_ids[:, 1:]
    token_logp = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    positions = torch.arange(token_logp.shape[1], device=token_logp.device)
    mask = positions.unsqueeze(0) >= max(prompt_length - 1, 0)
    return (token_logp * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
