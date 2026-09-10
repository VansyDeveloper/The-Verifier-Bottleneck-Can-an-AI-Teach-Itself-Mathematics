"""Shared action-space scoring for the composition PLAN policy.

The action policy scores the five allowed next operations at a given program
prefix. This module is the single implementation used by both the exploration
screen and the GRPO trainer, so the two cannot drift apart.

Memory note (D-017): the language-model head is applied only to the last few
positions via ``logits_to_keep``, and the float32 upcast happens on the single
scored row instead of the whole ``[5, L, vocab]`` tensor. Both changes leave the
returned scores numerically identical to the original implementation; only peak
memory differs. ``tests/test_action_policy.py`` asserts the equivalence.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from .prompts import build_prompt

OPERATIONS = ("SH1", "SC2", "REV", "AC1", "AX1")


def build_operation_batch(tokenizer, task, prefix: Sequence[str]):
    """Left-padded batch of `prompt + PROGRAM: <prefix> <op>` for all five ops."""
    base = build_prompt(task) + "\nPROGRAM:" + ((" " + " ".join(prefix)) if prefix else "")
    base_ids = tokenizer(base, add_special_tokens=False).input_ids
    rows: list[list[int]] = []
    added_lengths: list[int] = []
    for operation in OPERATIONS:
        added_ids = tokenizer(" " + operation, add_special_tokens=False).input_ids
        rows.append(base_ids + added_ids)
        added_lengths.append(len(added_ids))
    max_len = max(len(row) for row in rows)
    padded: list[list[int]] = []
    masks: list[list[int]] = []
    for row in rows:
        pad = max_len - len(row)
        padded.append([tokenizer.pad_token_id] * pad + row)
        masks.append([0] * pad + [1] * len(row))
    return padded, masks, added_lengths, max_len


def score_operations(model, tokenizer, task, prefix: Sequence[str]) -> torch.Tensor:
    """Summed token log-probabilities of each operation continuation.

    Returns a tensor of shape ``[5]`` aligned with :data:`OPERATIONS`. Gradients
    flow unless the caller disables them.
    """
    padded, masks, added_lengths, max_len = build_operation_batch(tokenizer, task, prefix)
    input_ids = torch.tensor(padded, device=model.device)
    attention = torch.tensor(masks, device=model.device)

    keep = max(added_lengths) + 1
    try:
        logits = model(input_ids=input_ids, attention_mask=attention, logits_to_keep=keep).logits
        offset = max_len - keep
    except TypeError:  # transformers without logits_to_keep
        logits = model(input_ids=input_ids, attention_mask=attention).logits
        offset = 0

    scores = []
    for row_index, added in enumerate(added_lengths):
        # Every row is left-padded to max_len, so the operation tokens always
        # occupy positions [max_len - added, max_len).
        score = torch.zeros((), device=logits.device, dtype=torch.float32)
        for position in range(max_len - added, max_len):
            target = input_ids[row_index, position]
            row_logits = logits[row_index, position - 1 - offset].float()
            score = score + torch.log_softmax(row_logits, dim=-1)[target]
        scores.append(score)
    return torch.stack(scores)


def sample_action(scores: Sequence[float], temperature: float, rng) -> str:
    """Sample one operation from temperature-scaled scores using ``rng``."""
    scaled = [float(score) / temperature for score in scores]
    maximum = max(scaled)
    weights = [math.exp(value - maximum) for value in scaled]
    return rng.choices(OPERATIONS, weights=weights, k=1)[0]
