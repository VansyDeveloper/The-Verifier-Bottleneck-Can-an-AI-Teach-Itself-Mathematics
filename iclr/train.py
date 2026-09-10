import argparse
import contextlib
import hashlib
import json
import math
import random
import time
import traceback
from collections import Counter
from pathlib import Path

import torch
from transformers import get_cosine_schedule_with_warmup
import composition_core as core
from .common import code_hash, environment, file_hash, read_jsonl, tree_hash, verify_data, write_json
from .data import mixture
from .evaluate import evaluate
from .modeling import collate, ce_sum, encode, load, program_scores, seed_all, set_loss

DEFAULTS = dict(method='ce', supervision='trace', replay_fraction=.2,
                budget_mode='examples', depth3_only=False, seed=0, epochs=2,
                effective_batch=64, micro_batch=1, learning_rate=1e-4,
                lora_rank=32, lora_alpha=64, lora_dropout=.05,
                device='auto', dtype='float32', gradient_checkpointing=True, prefix_batch=8)


def epoch_groups(examples, batch_size, seed):
    order = list(examples)
    random.Random(seed).shuffle(order)
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


def token_matched_groups(pool, budgets, seed):
    """Separate examples: no attention crosses examples; only excess labels are masked."""
    rng = random.Random(seed)
    order = list(pool)
    rng.shuffle(order)
    cursor = 0
    groups = []
    for budget in budgets:
        group = []
        remaining = budget
        while remaining:
            item = dict(order[cursor % len(order)])
            cursor += 1
            item['labels'] = list(item['labels'])
            keep = min(remaining, item['target_tokens'])
            if keep < item['target_tokens']:
                active = [i for i, value in enumerate(item['labels']) if value != -100]
                for i in active[keep:]:
                    item['labels'][i] = -100
            item['masked_target_tokens'] = item['target_tokens'] - keep
            item['target_tokens'] = keep
            group.append(item)
            remaining -= keep
        groups.append(group)
    return groups


def train_updates(model, tokenizer, ids, groups, cfg, output):
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=cfg['learning_rate'], weight_decay=0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(.03 * len(groups)), len(groups))
    device = next(model.parameters()).device
    receipt = Counter()
    unique = set()
    started = time.monotonic()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    model.train()
    with (output / 'training_metrics.jsonl').open('w') as log, (output / 'training_stream.jsonl').open('w') as stream:
        for step, group in enumerate(groups, 1):
            optimizer.zero_grad(set_to_none=True)
            total_tokens = sum(item['target_tokens'] for item in group)
            total_loss = 0.
            if cfg['method'] == 'ce':
                for offset in range(0, len(group), cfg['micro_batch']):
                    chunk = group[offset:offset + cfg['micro_batch']]
                    batch = collate(tokenizer, chunk, device)
                    loss = ce_sum(model, batch) / total_tokens
                    loss.backward()
                    total_loss += float(loss.detach())
                    receipt['total_forward_tokens'] += batch['input_ids'].numel()
            else:
                for item in group:
                    row = item['row']
                    programs = core.enumerate_programs(row['depth'])
                    mask = torch.tensor([core.verify_program(row['start'], row['target'], p, row['p'])
                                         for p in programs], device=device)
                    if any(core.motif_count(p) for p, correct in zip(programs, mask.tolist()) if correct):
                        raise ValueError('Set supervision leaks a withheld pair')
                    witness = programs.index(tuple(row['witness']))
                    scores = program_scores(model, tokenizer, ids, row, cfg['prefix_batch'], receipt)
                    loss = set_loss(scores, mask, witness, cfg['method']) / len(group)
                    loss.backward()
                    total_loss += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            for item in group:
                stream.write(json.dumps({'step': step, 'task_id': item['task_id'], 'kind': item['kind'],
                                         'input_ids': item['input_ids'], 'labels': item['labels'],
                                         'set_objective': cfg['method'] if cfg['method'] != 'ce' else None}) + '\n')
                receipt['example_exposures'] += 1
                receipt['prompt_tokens'] += item['prompt_tokens']
                receipt['all_target_tokens'] += item['target_tokens']
                receipt[item['kind'] + '_target_tokens'] += item['target_tokens']
                receipt[item['kind'] + '_exposures'] += 1
                if item.get('operation'):
                    receipt[item['operation'] + '_exposures'] += 1
                receipt['masked_target_tokens'] += item.get('masked_target_tokens', 0)
                unique.add(item['task_id'])
            log.write(json.dumps({'step': step, 'loss': total_loss, 'gradient_norm': float(norm),
                                  'target_tokens': total_tokens, 'examples': len(group),
                                  'seconds': time.monotonic() - started}, allow_nan=False) + '\n')
            log.flush()
            print(f'step {step}/{len(groups)} loss={total_loss:.5f}', flush=True)
    return {**receipt, 'optimizer_steps': len(groups), 'unique_tasks': len(unique),
            'wall_seconds': time.monotonic() - started,
            'peak_memory_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
            'loss_normalization': 'supervised_tokens_per_update' if cfg['method'] == 'ce' else 'tasks_per_update',
            'set_token_counters': 'reference program labels, not CE targets' if cfg['method'] != 'ce' else None}


def run(config):
    cfg = {**DEFAULTS, **config}
    for key in ('epochs', 'effective_batch', 'micro_batch', 'prefix_batch', 'lora_rank', 'lora_alpha'):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if cfg['method'] not in ('ce', 'single_norm', 'set_mass') or cfg['supervision'] not in ('program', 'trace'):
        raise ValueError('Unknown objective or supervision')
    if cfg['budget_mode'] not in ('examples', 'target_tokens') or not 0 <= cfg['replay_fraction'] <= 1:
        raise ValueError('Invalid budget or replay')
    if cfg['learning_rate'] <= 0 or not 0 <= cfg['lora_dropout'] < 1:
        raise ValueError('Invalid learning rate or dropout')
    if cfg['method'] != 'ce' and (not cfg['depth3_only'] or cfg['replay_fraction'] != 0 or cfg['budget_mode'] != 'examples'):
        raise ValueError('Set comparison is depth3-only, replay0, task-averaged')
    data, output = Path(cfg['data']), Path(cfg['output'])
    manifest = verify_data(data)
    initialize = cfg.get('initialize', False)
    default_size = 2 * manifest['files']['atomic_train.jsonl']['rows'] if initialize else manifest['config']['size']
    size = default_size if cfg.get('num_examples') is None else cfg['num_examples']
    if type(size) is not int or size < 1:
        raise ValueError('num_examples must be positive')
    cfg['num_examples'] = size
    source = cfg['model'] if initialize else cfg['base']
    source_identity = Path(source) / 'iclr_atomic.json'
    if not initialize and source_identity.is_file():
        atomic_identity = json.loads(source_identity.read_text())
        if atomic_identity['model'] != cfg['model']:
            raise ValueError(f"Atomic model is {atomic_identity['model']}, requested {cfg['model']}")
    binding = {'config': cfg, 'data_hash': file_hash(data / 'manifest.json'), 'code_hash': code_hash(),
               'base_hash': tree_hash(Path(source)) if Path(source).is_dir() else None}
    if (output / 'DONE').exists():
        done = json.loads((output / 'DONE').read_text())
        if done['binding'] != binding or done['payload_hash'] != tree_hash(output / ('checkpoint' if initialize else 'adapter')):
            raise ValueError('Completed run differs from requested config or saved payload')
        for name, digest in done['files'].items():
            if file_hash(output / name) != digest:
                raise ValueError(f'Completed output changed: {name}')
        print(f'Already complete: {output}')
        return
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'config.resolved.json').exists():
        if json.loads((output / 'config.resolved.json').read_text()) != binding:
            raise ValueError('Output directory belongs to another run; choose a new directory')
    write_json(output / 'config.resolved.json', binding)
    if not (output / 'environment.json').exists():
        write_json(output / 'environment.json', environment())
    try:
        with (output / 'stdout.log').open('a') as stdout, (output / 'stderr.log').open('a') as stderr:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                seed_all(cfg['seed'])
                payload = output / ('checkpoint' if initialize else 'adapter')
                if not (output / 'TRAINED').exists():
                    model, tokenizer, ids = load(source, train=True, initialize=initialize,
                                                  **{k: v for k, v in cfg.items() if k not in ('base', 'initialize')})
                    write_json(output / 'resolved_model.json', {
                        'config': model.config.to_dict(),
                        'parameters': sum(p.numel() for p in model.parameters()),
                        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                        'dtype': str(next(model.parameters()).dtype),
                        'source_identity_verified': initialize or source_identity.is_file()})
                    compositions = read_jsonl(data / 'train.jsonl')
                    if cfg['depth3_only']:
                        compositions = [row for row in compositions if row['depth'] == 3]
                    atomic = read_jsonl(data / 'atomic_train.jsonl')
                    examples = mixture(compositions, atomic, size, cfg['replay_fraction'], cfg['seed'],
                                       'program_trace' if cfg['supervision'] == 'trace' else 'program_only')
                    encoded = [encode(tokenizer, item) for item in examples]
                    # Same source order and reference budgets across trace/replay comparisons.
                    reference = [encode(tokenizer, item) for item in mixture(
                        compositions, atomic, size, .2, cfg['seed'], 'program_trace')]
                    groups = []
                    for epoch in range(cfg['epochs']):
                        epoch_seed = cfg['seed'] + epoch
                        if cfg['budget_mode'] == 'target_tokens':
                            budgets = [sum(item['target_tokens'] for item in group) for group in
                                       epoch_groups(reference, cfg['effective_batch'], epoch_seed)]
                            groups.extend(token_matched_groups(encoded, budgets, epoch_seed))
                        else:
                            groups.extend(epoch_groups(encoded, cfg['effective_batch'], epoch_seed))
                    budget = train_updates(model, tokenizer, ids, groups, cfg, output)
                    write_json(output / 'budget.json', budget)
                    model.eval()
                    probe = compositions[0]
                    with torch.inference_mode():
                        before = program_scores(model, tokenizer, ids, probe, cfg['prefix_batch']).cpu()
                    write_json(output / 'reload_probe.json', {'task': probe, 'scores': before.tolist()})
                    if initialize:
                        model = model.merge_and_unload()
                    model.save_pretrained(payload)
                    tokenizer.save_pretrained(payload)
                    if initialize:
                        write_json(payload / 'iclr_atomic.json', {'model': cfg['model'],
                            'model_revision': getattr(model.config, '_commit_hash', None),
                            'data_hash': binding['data_hash'], 'protocol': 'fresh_atomic_plan_apply_v1'})
                    write_json(output / 'TRAINED', {'payload_hash': tree_hash(payload),
                        'model_revision': getattr(model.config, '_commit_hash', None),
                        'files': {name: file_hash(output / name) for name in
                                  ('budget.json', 'training_metrics.jsonl', 'training_stream.jsonl',
                                   'reload_probe.json', 'resolved_model.json', 'environment.json')}})
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                trained = json.loads((output / 'TRAINED').read_text())
                if trained['payload_hash'] != tree_hash(payload):
                    raise ValueError('Saved checkpoint changed after training')
                for name, digest in trained['files'].items():
                    if file_hash(output / name) != digest:
                        raise ValueError(f'Training receipt changed after training: {name}')
                model, tokenizer, ids = load(str(payload) if initialize else source,
                    adapter=None if initialize else str(payload), device=cfg['device'], dtype=cfg['dtype'])
                probe = json.loads((output / 'reload_probe.json').read_text())
                with torch.inference_mode():
                    after = program_scores(model, tokenizer, ids, probe['task'], cfg['prefix_batch']).cpu()
                difference = float((after - torch.tensor(probe['scores'])).abs().max())
                tolerance = .1 if cfg['dtype'] == 'bfloat16' else 1e-4
                if difference > tolerance:
                    raise RuntimeError(f'Save/reload scores differ by {difference} > {tolerance}')
                write_json(output / 'reload_check.json', {'max_absolute_difference': difference, 'tolerance': tolerance})
                evaluate(model, tokenizer, ids, data, output / 'eval',
                         {'training_seed': cfg['seed'], 'method': cfg['method'], 'base_hash': binding['base_hash'],
                          'model_hash': trained['payload_hash'], 'data_hash': binding['data_hash'],
                          'code_hash': binding['code_hash']}, prefix_batch=cfg['prefix_batch'])
        files = {str(p.relative_to(output)): file_hash(p) for p in output.rglob('*.json*')
                 if payload not in p.parents}
        write_json(output / 'DONE', {'binding': binding, 'payload_hash': trained['payload_hash'], 'files': files})
        (output / 'FAILED').unlink(missing_ok=True)
        print(f'Complete: {output}')
    except Exception:
        (output / 'FAILED').write_text(traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description='Atomic initialization or one fixed supervised experiment')
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--config', type=Path)
    choice.add_argument('--init', action='store_true')
    parser.add_argument('--model')
    parser.add_argument('--revision')
    parser.add_argument('--data')
    parser.add_argument('--out')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--effective-batch', type=int, default=64)
    parser.add_argument('--micro-batch', type=int, default=1)
    parser.add_argument('--prefix-batch', type=int, default=8)
    parser.add_argument('--num-examples', type=int)
    args = parser.parse_args()
    if args.config:
        config = json.loads(args.config.read_text())
    else:
        if not all((args.model, args.data, args.out)):
            parser.error('--init requires --model, --data, --out')
        config = dict(initialize=True, model=args.model, revision=args.revision, data=args.data, output=args.out,
                      replay_fraction=1., epochs=args.epochs, effective_batch=args.effective_batch,
                      micro_batch=args.micro_batch, prefix_batch=args.prefix_batch, device=args.device,
                      dtype=args.dtype, num_examples=args.num_examples)
    run(config)


if __name__ == '__main__':
    main()
