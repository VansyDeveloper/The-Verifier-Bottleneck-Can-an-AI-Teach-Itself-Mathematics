"""Reuse the canonical prefix scorer for full/local policies and state intervention."""

from pathlib import Path
import json
import random
from contextlib import contextmanager

import numpy as np
import torch

import composition_core as core
from .modeling import program_scores
from .research_objectives import StateProjection
from .research_dsl import plan_prompt, prompt_prefix, trajectory, verify


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


@contextmanager
def evaluation_context(model, head=None):
    """Evaluation must not consume the training RNG or change any module's mode."""
    state = rng_state()
    modules = list(model.modules()) + (list(head.modules()) if head is not None else [])
    modes = [m.training for m in modules]
    try:
        model.eval()
        if head is not None:
            head.eval()
        with torch.no_grad():
            yield
    finally:
        for module, mode in zip(modules, modes):
            module.training = mode
        restore_rng(state)


def deterministic_training(model):
    """Keep HF gradient checkpointing active while disabling Qwen/LoRA dropout."""
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
        if isinstance(getattr(module, 'attention_dropout', None), (float, int)):
            module.attention_dropout = 0.


def score_task(model, tokenizer, ids, row, batch_size=32, *, programs=None,
               head=None, ablate=False, accounting=None, prompt_builder=None):
    programs = list(core.enumerate_programs(row['depth'])) if programs is None else list(map(tuple, programs))
    prefixes = []
    full = program_scores(model, tokenizer, ids, row, batch_size, accounting,
                          programs=programs, action_head=head, ablate_state=ablate,
                          prefix_tensors=prefixes, prompt_builder=prompt_builder or prompt_prefix)
    lookup = {r['prefix']: r['full'].log_softmax(-1) for r in prefixes}
    local = torch.stack([sum(lookup[p[:i]][core.OPS.index(op)] for i, op in enumerate(p)) for p in programs])
    return full, local, prefixes


def load_head(adapter, device):
    path = Path(adapter) / 'state_projection.json' if adapter else None
    if path is None or not path.exists():
        return None
    head = StateProjection(**json.loads(path.read_text())).to(device)
    head.load_state_dict(torch.load(path.with_suffix('.pt'), map_location=device, weights_only=True))
    head.eval()
    return head


def state_prompt_ids(row, prefix, tokenizer, mode):
    prompt = plan_prompt(row)
    suffix = ' ' + core.program_answer(prefix) if prefix else ''
    if mode == 'plain':
        return tokenizer.encode(prompt + suffix, add_special_tokens=False), 0
    if mode not in ('state', 'token_control'):
        raise ValueError('Unknown state-access intervention')
    current = trajectory(row, prefix)[-1]
    extra = tokenizer.encode('CURRENT_STATE: ' + core.format_state(current) + '\n', add_special_tokens=False)
    if mode == 'token_control':
        # Same number of tokens as the actual state annotation, with a fixed,
        # task-independent token. The residual length signal is present in both.
        token = tokenizer.encode(' 0', add_special_tokens=False)
        if not token:
            raise ValueError('No neutral token for the length control')
        extra = [token[-1]] * len(extra)
    before, after = prompt.split('Return exactly one line:')
    encoded = (tokenizer.encode(before, add_special_tokens=False) + extra
               + tokenizer.encode('Return exactly one line:' + after + suffix, add_special_tokens=False))
    return encoded, len(extra)


def next_logits(model, tokens, ids, head=None, ablate=False):
    device = next(model.parameters()).device
    inputs = torch.tensor([tokens], device=device)
    output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), logits_to_keep=1,
                   **({'output_hidden_states': True} if head else {}))
    logits = output.logits[0, -1].float()
    if head:
        _, adjustment = head(output.hidden_states[-1][:, -1], ablate=ablate)
        logits = logits.clone()
        logits[[ids[op] for op in core.OPS]] += adjustment[0]
    return logits


def solve(model, tokenizer, ids, row, mode='plain', head=None, ablate=False, free=False):
    prefix, tokens, queries, forward_tokens, annotation = [], [], 0, 0, 0
    base = tokenizer.encode(plan_prompt(row), add_special_tokens=False)
    for _ in range(row['depth'] + 8 if free else row['depth']):
        if free:
            encoded = base + tokens
        else:
            encoded, added = state_prompt_ids(row, prefix, tokenizer, mode)
            annotation += added
            queries += int(mode in ('state', 'token_control')) * len(prefix)
        logits = next_logits(model, encoded, ids, head, ablate)
        forward_tokens += len(encoded)
        if free:
            token = int(logits.argmax())
            if token == tokenizer.eos_token_id:
                break
            tokens.append(token)
        else:
            prefix.append(core.OPS[int(logits[[ids[op] for op in core.OPS]].argmax())])
    if free:
        inverse = {v: k for k, v in ids.items()}
        prefix = [inverse[token] for token in tokens] if all(token in inverse for token in tokens) else []
    parsed = len(prefix) == row['depth']
    return {'program': prefix, 'parse_ok': parsed,
            'correct': parsed and verify(row, prefix),
            'raw_tokens': tokens if free else [ids[op] for op in prefix],
            'forward_tokens': forward_tokens, 'state_interpreter_operations': queries,
            'annotation_tokens': annotation, 'max_new_tokens': row['depth'] + 8 if free else row['depth']}
