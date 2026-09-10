import itertools
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from tokenizers import AddedToken
from transformers import AutoModelForCausalLM, AutoTokenizer

import composition_core as core

SCORER_ID = 'full_vocab_op_tokens_v1'
PROMPT_VERSION = 'stage4_plan_v1'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load(base, *, adapter=None, train=False, initialize=False, device='auto',
         dtype='float32', revision=None, lora_rank=32, lora_alpha=64,
         lora_dropout=.05, gradient_checkpointing=True, **unused):
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    if device not in ('cpu', 'cuda') or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError(f'Unavailable device: {device}')
    if dtype not in ('float32', 'bfloat16'):
        raise ValueError('dtype must be float32 or bfloat16')
    if dtype == 'bfloat16' and (device != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('bfloat16 requires a CUDA device with BF16 support')
    tokenizer = AutoTokenizer.from_pretrained(base, revision=revision)
    if initialize:
        tokenizer.add_special_tokens({'additional_special_tokens': [
            AddedToken(core.TOKENS[op], lstrip=True, rstrip=False, normalized=False, special=True)
            for op in core.OPS]})
    if tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer needs EOS')
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = 'left'
    token_ids = {op: tokenizer.convert_tokens_to_ids(core.TOKENS[op]) for op in core.OPS}
    if len(set(token_ids.values())) != 5 or any(value is None or value == tokenizer.unk_token_id for value in token_ids.values()):
        raise ValueError('Atomic checkpoint lacks five distinct operation tokens')
    for prefix in ('', 'PROGRAM:', 'PROGRAM: ' + core.TOKENS['SH1']):
        for op in core.OPS:
            expected = tokenizer.encode(prefix, add_special_tokens=False) + [token_ids[op]]
            if tokenizer.encode(prefix + ' ' + core.TOKENS[op], add_special_tokens=False) != expected:
                raise ValueError('Operation tokenization differs between training and ranking')
    model = AutoModelForCausalLM.from_pretrained(
        base, revision=revision, torch_dtype=getattr(torch, dtype), attn_implementation='sdpa')
    if initialize:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        with torch.no_grad():
            for op in core.OPS:
                source = tokenizer.encode(op, add_special_tokens=False)
                for layer in (model.get_input_embeddings(), model.get_output_embeddings()):
                    layer.weight[token_ids[op]].copy_(layer.weight[source].mean(0))
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    elif train:
        config = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            modules_to_save=['embed_tokens', 'lm_head'] if initialize else None,
            ensure_weight_tying=bool(initialize and model.config.tie_word_embeddings),
            bias='none', task_type='CAUSAL_LM')
        model = get_peft_model(model, config)
    model.to(device)
    if train and gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    model.train(train)
    return model, tokenizer, token_ids


def encode(tokenizer, example):
    prompt = tokenizer.encode(example['prompt'], add_special_tokens=False)
    answer = tokenizer.encode(' ' + example['answer'], add_special_tokens=False) + [tokenizer.eos_token_id]
    return {**example, 'input_ids': prompt + answer, 'labels': [-100] * len(prompt) + answer,
            'target_tokens': len(answer), 'prompt_tokens': len(prompt)}


def collate(tokenizer, examples, device):
    width = max(len(item['input_ids']) for item in examples)
    ids, masks, labels = [], [], []
    for item in examples:
        padding = width - len(item['input_ids'])
        ids.append([tokenizer.pad_token_id] * padding + item['input_ids'])
        masks.append([0] * padding + [1] * len(item['input_ids']))
        labels.append([-100] * padding + item['labels'])
    attention = torch.tensor(masks, device=device)
    positions = (attention.cumsum(-1) - 1).clamp(min=0)
    return dict(input_ids=torch.tensor(ids, device=device), attention_mask=attention,
                position_ids=positions, labels=torch.tensor(labels, device=device))


def ce_sum(model, batch):
    labels = batch['labels'][:, 1:]
    logits = model(**{k: v for k, v in batch.items() if k != 'labels'}).logits[:, :-1].float()
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                           reduction='sum', ignore_index=-100)


def program_scores(model, tokenizer, token_ids, row, batch_size=8, accounting=None):
    """Differentiable equivalent of the frozen Stage4 full-vocabulary scorer."""
    device = next(model.parameters()).device
    scores = {(): torch.zeros((), device=device)}
    for _ in range(int(row['depth'])):
        prefixes = sorted(scores)
        next_scores = {}
        for start in range(0, len(prefixes), batch_size):
            chunk = prefixes[start:start + batch_size]
            texts = [core.plan_prompt(row) + (' ' + core.program_answer(p) if p else '') for p in chunk]
            encoded = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False).to(device)
            if accounting is not None:
                accounting['total_forward_tokens'] += encoded.input_ids.numel()
                accounting['prefix_sequences'] += len(chunk)
            positions = (encoded.attention_mask.cumsum(-1) - 1).clamp(min=0)
            logits = model(**encoded, position_ids=positions).logits[:, -1].float()
            logprobs = logits.log_softmax(-1)
            for i, prefix in enumerate(chunk):
                for op in core.OPS:
                    next_scores[prefix + (op,)] = scores[prefix] + logprobs[i, token_ids[op]]
        scores = next_scores
    return torch.stack([scores[p] for p in core.enumerate_programs(row['depth'])])


def set_loss(scores, correct_mask, witness_index, method):
    if not bool(correct_mask.any()) or not bool(correct_mask[witness_index]):
        raise ValueError('Correct set must contain the witness')
    if method == 'single_norm':
        numerator = scores[witness_index]
    elif method == 'set_mass':
        numerator = torch.logsumexp(scores[correct_mask], 0)
    else:
        raise ValueError(f'Unknown set objective: {method}')
    return torch.logsumexp(scores, 0) - numerator
