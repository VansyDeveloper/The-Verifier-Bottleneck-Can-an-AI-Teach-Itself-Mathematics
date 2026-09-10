"""D-017 equivalence guard.

`vbexp.action_policy.score_operations` restricts the language-model head to the
last few positions and upcasts one row at a time. That is a memory change only.
This test pins that claim: the optimized path must reproduce the original
full-logits implementation to floating-point tolerance on a real forward pass.

Skipped when the base model is not present locally.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vbexp.action_policy import OPERATIONS, build_operation_batch, score_operations
from vbexp.prompts import build_prompt
from vbexp.task import Task

MODEL = "Qwen/Qwen3-0.6B-Base"

TASK = Task(
    task_id="equivalence",
    mode="plan",
    split="unit",
    p=11,
    degree_cap=3,
    start=(2, 5, 0, 1),
    operations=tuple(OPERATIONS),
    target=(7, 3, 4, 8),
    max_steps=3,
)


def _score_operations_original(model, tokenizer, task, prefix):
    """Verbatim transcription of the pre-D-017 scoring loop."""
    base = build_prompt(task) + "\nPROGRAM:" + ((" " + " ".join(prefix)) if prefix else "")
    base_ids = tokenizer(base, add_special_tokens=False).input_ids
    rows, lengths = [], []
    for operation in OPERATIONS:
        added_ids = tokenizer(" " + operation, add_special_tokens=False).input_ids
        rows.append(base_ids + added_ids)
        lengths.append(len(added_ids))
    max_len = max(map(len, rows))
    padded, masks = [], []
    for row in rows:
        pad = max_len - len(row)
        padded.append([tokenizer.pad_token_id] * pad + row)
        masks.append([0] * pad + [1] * len(row))
    input_ids = torch.tensor(padded, device=model.device)
    attention = torch.tensor(masks, device=model.device)
    logits = model(input_ids=input_ids, attention_mask=attention).logits.float()
    scores = []
    for row_index, added in enumerate(lengths):
        row_length = len(rows[row_index])
        pad = max_len - row_length
        start = pad + len(base_ids)
        score = torch.zeros((), device=logits.device)
        for position in range(start, pad + row_length):
            target = input_ids[row_index, position]
            score = score + torch.log_softmax(logits[row_index, position - 1], dim=-1)[target]
        scores.append(score)
    return torch.stack(scores)


@pytest.fixture(scope="module")
def loaded():
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, local_files_only=True
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"base model unavailable locally: {type(exc).__name__}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, tokenizer


@pytest.mark.parametrize("prefix", [(), ("SH1",), ("REV", "SC2")])
def test_scores_match_original_implementation(loaded, prefix):
    model, tokenizer = loaded
    with torch.inference_mode():
        optimized = score_operations(model, tokenizer, TASK, prefix)
        original = _score_operations_original(model, tokenizer, TASK, prefix)
    assert torch.allclose(optimized, original, atol=1e-4, rtol=1e-4), (
        f"prefix={prefix} optimized={optimized.tolist()} original={original.tolist()}"
    )


def test_batch_layout_is_left_padded_to_common_length():
    tokenizer = pytest.importorskip("transformers").AutoTokenizer.from_pretrained(
        MODEL, local_files_only=True
    )
    padded, masks, added_lengths, max_len = build_operation_batch(tokenizer, TASK, ("SH1",))
    assert len(padded) == len(masks) == len(added_lengths) == len(OPERATIONS)
    assert all(len(row) == max_len for row in padded)
    assert all(len(row) == max_len for row in masks)
    # The scored operation tokens must sit at the very end of every row.
    assert all(all(mask[-added:]) for mask, added in zip(masks, added_lengths))
