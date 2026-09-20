"""Sampled versus exact finite-group reward gradient; identical local action policy."""

import argparse
import itertools
import json
import math
from pathlib import Path
import random
import time
import traceback

import torch

import composition_core as core
from .common import code_hash, environment, file_hash, read_jsonl, tree_hash, verify_data, verify_receipt, write_json
from .modeling import load, program_scores, seed_all


def exact_reward_loss(logq, correct, group_size):
    if group_size < 2:
        raise ValueError('Finite-group normalization requires G >= 2')
    mass = logq[correct].exp().sum()
    m = float(mass.detach())
    if m <= 0 or m >= 1:
        return mass * 0
    a = sum(math.comb(group_size, j) * m ** j * (1 - m) ** (group_size - j)
            * math.sqrt(j * (group_size - j)) / group_size for j in range(1, group_size))
    return -a / (m * (1 - m)) * mass


def sampled_reward_loss(logq, correct, indices):
    rewards = correct[indices].to(logq.dtype)
    advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp(min=1e-6)
    return -(advantages.detach() * logq[indices]).mean()


def gradient_check(out):
    checks = []
    for group in (2, 3, 5):
        for mask in ([True, False, False], [True, True, False], [True, True, True], [False, False, False]):
            logits = torch.tensor([-.3, .2, .8], dtype=torch.float64, requires_grad=True)
            logq = logits.log_softmax(0)
            correct = torch.tensor(mask)
            exact = torch.autograd.grad(exact_reward_loss(logq, correct, group), logits, retain_graph=True)[0]
            expectation = logits.sum() * 0
            for draw in itertools.product(range(3), repeat=group):
                indices = torch.tensor(draw)
                probability = logq[indices].sum().exp().detach()
                expectation = expectation + probability * sampled_reward_loss(logq, correct, indices)
            enumerated = torch.autograd.grad(expectation, logits)[0]
            torch.testing.assert_close(exact, enumerated, rtol=1e-10, atol=1e-12)
            checks.append({'group_size': group, 'correct': mask, 'max_error': float((exact - enumerated).abs().max())})
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'gradient_check.json', {'status': 'PASS', 'checks': checks,
        'scope': 'exact expectation of the binary population-std normalized reward gradient; no KL, clipping or optimizer expectation identity'})
    write_json(out / 'DONE', {'files': {'gradient_check.json': file_hash(out / 'gradient_check.json')}})
    return checks


def run(config):
    defaults = {'group_size': 32, 'steps': 158, 'learning_rate': 1e-4, 'optimizer': 'adamw',
                'seed': 0, 'method': 'sampled', 'device': 'cuda', 'dtype': 'bfloat16', 'prefix_batch': 32}
    allowed = defaults.keys() | {'base', 'adapter', 'data', 'output', 'expected_base_hash', 'expected_adapter_hash'}
    if set(config) - allowed:
        raise ValueError('Unknown reward-control configuration fields')
    cfg = {**defaults, **config}
    if cfg['method'] not in ('sampled', 'exact') or cfg['optimizer'] not in ('sgd', 'adamw'):
        raise ValueError('Unknown reward method or optimizer')
    if any(type(cfg[key]) is not int or cfg[key] < minimum for key, minimum in
           (('group_size', 2), ('steps', 1), ('prefix_batch', 1))):
        raise ValueError('Group size, steps and prefix batch must be positive integers; G >= 2')
    if type(cfg['seed']) is not int or not 0 <= cfg['seed'] < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    if not math.isfinite(cfg['learning_rate']) or not 0 < cfg['learning_rate'] < 1:
        raise ValueError('Invalid group size, steps, batch size or learning rate')
    verify_data(cfg['data'])
    binding = {'config': cfg, 'base_hash': tree_hash(cfg['base']), 'initial_adapter_hash': tree_hash(cfg['adapter']),
               'data_hash': file_hash(Path(cfg['data']) / 'manifest.json'), 'code_hash': code_hash()}
    for field, key in (('expected_base_hash', 'base_hash'), ('expected_adapter_hash', 'initial_adapter_hash')):
        if cfg.get(field) and cfg[field] != binding[key]:
            raise ValueError('Wrong initial checkpoint for reward control')
    out = Path(cfg['output'])
    if (out / 'DONE').exists():
        receipt = verify_receipt(out)
        if receipt['binding'] != binding or tree_hash(out / 'adapter') != receipt['payload_hash']:
            raise ValueError('Completed reward control changed')
        return
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'binding.json').exists() and json.loads((out / 'binding.json').read_text()) != binding:
        raise ValueError('Partial reward run belongs to another configuration')
    write_json(out / 'binding.json', binding)
    write_json(out / 'environment.json', environment())
    try:
        seed_all(cfg['seed'])
        model, tokenizer, ids = load(cfg['base'], adapter=cfg['adapter'], train=True,
                                     device=cfg['device'], dtype=cfg['dtype'], gradient_checkpointing=False)
        # eval() disables dropout for both discovery and loss; gradients still flow.
        model.eval()
        optimizer_class = torch.optim.SGD if cfg['optimizer'] == 'sgd' else torch.optim.AdamW
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = optimizer_class(parameters, lr=cfg['learning_rate'], weight_decay=0)
        rows = [r for r in read_jsonl(Path(cfg['data']) / 'train.jsonl') if r['depth'] == 3]
        if not rows:
            raise ValueError('No depth-three reward tasks')
        random.Random(cfg['seed']).shuffle(rows)
        probe = rows[0]
        device = next(model.parameters()).device
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            reference = program_scores(model, tokenizer, ids, probe, cfg['prefix_batch'], normalization='local')
            reference = reference.log_softmax(0).detach()
            repeat = program_scores(model, tokenizer, ids, probe, cfg['prefix_batch'], normalization='local').log_softmax(0)
            initial_kl = float((repeat.exp() * (repeat - reference)).sum())
        if abs(initial_kl) > 1e-7:
            raise ValueError('Nonzero initial KL in deterministic reward control')
        started = time.monotonic()
        with (out / 'training_metrics.jsonl').open('w') as log:
            for step in range(cfg['steps']):
                row = rows[step % len(rows)]
                programs = core.enumerate_programs(3)
                mask = torch.tensor([core.verify_program(row['start'], row['target'], p, row['p']) for p in programs], device=device)
                optimizer.zero_grad(set_to_none=True)
                local = program_scores(model, tokenizer, ids, row, cfg['prefix_batch'], normalization='local')
                correction = float(torch.logsumexp(local.detach(), 0))
                if abs(correction) > 1e-4:
                    raise ValueError('Local tree probabilities do not sum to one')
                logq = local.log_softmax(0)  # floating-point roundoff only, measured above
                mass = float(logq[mask].exp().sum().detach())
                indices = torch.multinomial(logq.exp().detach(), cfg['group_size'], replacement=True) if cfg['method'] == 'sampled' else None
                loss = sampled_reward_loss(logq, mask, indices) if indices is not None else exact_reward_loss(logq, mask, cfg['group_size'])
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, math.inf if cfg['optimizer'] == 'sgd' else 1., error_if_nonfinite=True)
                optimizer.step()
                metric = {'step': step + 1, 'task_id': row['task_id'], 'loss': float(loss.detach()),
                    'correct_mass': mass, 'iid_pass@32': -math.expm1(32 * math.log1p(-mass)) if mass < 1 else 1.,
                    'gradient_norm': float(norm), 'normalization_roundoff': correction,
                    'sampled_programs': [list(programs[i]) for i in indices.tolist()] if indices is not None else None,
                    'positive_fraction': float(mask[indices].float().mean()) if indices is not None else None,
                    'seconds': time.monotonic() - started}
                log.write(json.dumps(metric, allow_nan=False) + '\n')
                log.flush()
                print(f"step {step + 1}/{cfg['steps']} mass={mass:.6f}", flush=True)
        with torch.no_grad():
            before = program_scores(model, tokenizer, ids, probe, cfg['prefix_batch'], normalization='local').cpu()
        model.save_pretrained(out / 'adapter')
        tokenizer.save_pretrained(out / 'adapter')
        write_json(out / 'budget.json', {'optimizer_steps': cfg['steps'], 'task_exposures': cfg['steps'],
            'group_size': cfg['group_size'], 'wall_seconds': time.monotonic() - started,
            'peak_memory_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
            'dropout_mode': 'disabled for sampling and loss', 'temperature_sampling': 1, 'temperature_loss': 1,
            'initial_kl': initial_kl, 'kl_beta': 0, 'reward_normalization': 'binary, population std, floor 1e-6',
            'optimizer': cfg['optimizer'], 'gradient_clip': None if cfg['optimizer'] == 'sgd' else 1,
            'objective': 'reward part of finite-group GRPO, task mean; exact expectation is before optimizer/clipping'})
        del model, optimizer, parameters, loss, logq, local
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        model, tokenizer, ids = load(cfg['base'], adapter=str(out / 'adapter'), device=cfg['device'], dtype=cfg['dtype'])
        with torch.no_grad():
            after = program_scores(model, tokenizer, ids, probe, cfg['prefix_batch'], normalization='local').cpu()
        tolerance = .1 if cfg['dtype'] == 'bfloat16' else 1e-4
        torch.testing.assert_close(before, after, rtol=0, atol=tolerance)
        write_json(out / 'reload_check.json', {'max_error': float((before - after).abs().max()), 'tolerance': tolerance})
        write_json(out / 'DONE', {'binding': binding, 'payload_hash': tree_hash(out / 'adapter'), 'files': {
            p.name: file_hash(p) for p in out.iterdir() if p.is_file() and p.name not in ('DONE', 'FAILED')}})
        (out / 'FAILED').unlink(missing_ok=True)
    except Exception:
        (out / 'FAILED').write_text(traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choices = parser.add_mutually_exclusive_group(required=True)
    choices.add_argument('--check', action='store_true')
    choices.add_argument('--config', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.check:
        if not args.out:
            parser.error('--check requires --out')
        print(json.dumps(gradient_check(args.out)))
    else:
        run(json.loads(args.config.read_text()))


if __name__ == '__main__':
    main()
