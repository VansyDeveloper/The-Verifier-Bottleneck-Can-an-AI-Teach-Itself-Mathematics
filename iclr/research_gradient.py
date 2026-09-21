"""Fixed-checkpoint MC/exact gradients and actual isolated Adam update diagnostics."""

import argparse
import json
import math
from pathlib import Path

import torch

import composition_core as core
from .common import read_jsonl, write_json
from .modeling import load, seed_all
from .research_data import check_panels
from .research_io import start_run, finish_run
from .research_model import deterministic_training, score_task
from .research_objectives import assignment_cycles
from .reward_control import exact_reward_loss, sampled_reward_loss, gradient_check


def vector_gradient(loss, parameters, retain=False):
    values = torch.autograd.grad(loss, parameters, retain_graph=retain, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().cpu().flatten()
                      for p, g in zip(parameters, values)])


def dev_function(model, tokenizer, ids, panels, prefix_batch):
    values = []
    for panel in panels:
        candidates = [row['witness'] for row in panel]
        matrix = torch.stack([score_task(model, tokenizer, ids, row, prefix_batch, programs=candidates)[1] for row in panel])
        values.append(assignment_cycles(matrix).mean())
    return torch.stack(values).mean()


def isolated_update(model, tokenizer, ids, parameters, original, gradient, dev_gradient,
                    dev_before, dev_panels, train_row, learning_rate, prefix_batch):
    offset = 0
    with torch.no_grad():
        for parameter, initial in zip(parameters, original):
            parameter.copy_(initial)
            parameter.grad = gradient[offset:offset + parameter.numel()].reshape_as(parameter).to(parameter)
            offset += parameter.numel()
    # Every isolated step starts with identical weights AND empty Adam moments.
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.,
                                 fused=parameters[0].device.type == 'cuda')
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    optimizer.step()
    delta = torch.cat([(p.detach().float().cpu() - initial.float().cpu()).flatten() for p, initial in zip(parameters, original)])
    predicted = float(dev_gradient.double().dot(delta.double()))
    with torch.no_grad():
        actual = float(dev_function(model, tokenizer, ids, dev_panels, prefix_batch)) - dev_before
        _, local, _ = score_task(model, tokenizer, ids, train_row, prefix_batch)
        programs = list(core.enumerate_programs(train_row['depth']))
        mask = torch.tensor([list(p) in train_row['correct_programs'] for p in programs], device=local.device)
        mass_after = float(local.log_softmax(0)[mask].exp().sum())
        scores_after = local.cpu().tolist()
        for parameter, initial in zip(parameters, original):
            parameter.copy_(initial)
            parameter.grad = None
    return {'learning_rate': learning_rate, 'pre_clip_norm': float(norm), 'parameter_delta_norm': float(delta.norm()),
            'predicted_dev_delta': predicted, 'actual_dev_delta': actual, 'taylor_absolute_error': abs(actual - predicted),
            'train_mass_after': mass_after, 'train_scores_after': scores_after,
            'optimizer_state': 'identical empty moments, AdamW weight_decay=0, clip=1'}


def run(config):
    cfg = {**dict(seed=0, device='cuda', dtype='bfloat16', prefix_batch=16, train_tasks=4,
                  dev_panels_per_family=2, mc_groups=128, mc_blocks=8, group_size=32,
                  learning_rates=[1e-5, 3e-6, 1e-6]), **config}
    if (any(type(cfg[k]) is not int or cfg[k] < 1 for k in ('train_tasks', 'dev_panels_per_family', 'mc_groups', 'mc_blocks', 'prefix_batch'))
            or cfg['mc_groups'] % cfg['mc_blocks'] or cfg['mc_blocks'] < 2 or cfg['group_size'] < 2
            or any(not math.isfinite(lr) or lr <= 0 for lr in cfg['learning_rates'])):
        raise ValueError('Invalid gradient diagnostic budget')
    _, binding, done = start_run(cfg, 'reward_diagnostic')
    if done:
        return
    out, data = Path(cfg['output']), Path(cfg['data'])
    gradient_check(out / 'kernel')
    dev_panels = [panel for f in ('B', 'D') for panel in
                  check_panels(read_jsonl(data / f'dev_panel_{f}.jsonl'))[:cfg['dev_panels_per_family']]]
    train_rows = read_jsonl(data / 'train.jsonl')[:cfg['train_tasks']]
    if not train_rows or not dev_panels:
        raise ValueError('Need disjoint train tasks and held-out dev panels')
    dtypes = ['float32', 'bfloat16'] if cfg['device'] == 'cuda' else ['float32']
    reports = []
    for dtype in dtypes:
        seed_all(cfg['seed'])
        model, tokenizer, ids = load(cfg['base'], adapter=cfg.get('adapter'), train=True,
            device=cfg['device'], dtype=dtype, lora_dropout=0., gradient_checkpointing=True)
        deterministic_training(model)
        parameters = [p for p in model.parameters() if p.requires_grad]
        original = [p.detach().cpu().clone() for p in parameters]
        functional = dev_function(model, tokenizer, ids, dev_panels, cfg['prefix_batch'])
        before = float(functional.detach())
        dev_gradient = vector_gradient(functional, parameters)
        for row in train_rows:
            _, local, _ = score_task(model, tokenizer, ids, row, cfg['prefix_batch'])
            logq = local.log_softmax(0)
            programs = list(core.enumerate_programs(row['depth']))
            correct = torch.tensor([list(p) in row['correct_programs'] for p in programs], device=local.device)
            mass = float(logq[correct].detach().exp().sum())
            exact = vector_gradient(exact_reward_loss(logq, correct, cfg['group_size']), parameters, retain=True)
            draws = torch.multinomial(logq.detach().exp(), cfg['mc_groups'] * cfg['group_size'], replacement=True).reshape(-1, cfg['group_size'])
            rewards = correct[draws].to(logq.dtype)
            advantages = (rewards - rewards.mean(1, keepdim=True)) / rewards.std(1, unbiased=False, keepdim=True).clamp_min(1e-6)
            mean, square = torch.zeros_like(exact), torch.zeros_like(exact)
            first = None
            per_block = cfg['mc_groups'] // cfg['mc_blocks']
            for block in range(cfg['mc_blocks']):
                idx = slice(block * per_block, (block + 1) * per_block)
                loss = -(advantages[idx].detach() * logq[draws[idx]]).mean()
                estimate = vector_gradient(loss, parameters, retain=True)
                mean += estimate / cfg['mc_blocks']; square += estimate.square() / cfg['mc_blocks']
                if first is None:
                    first = vector_gradient(sampled_reward_loss(logq, correct, draws[0]), parameters, retain=True)
            variance = ((square - mean.square()).clamp_min(0) * cfg['mc_blocks'] / (cfg['mc_blocks'] - 1))
            standard_error = float((variance.sum() / cfg['mc_blocks']).sqrt())
            error = float((mean - exact).norm())
            steps = []
            initial_scores = local.detach().cpu().tolist()
            del local, logq, loss
            for method, gradient in (('exact', exact), ('sampled_one_group', first), ('sampled_mc_mean', mean)):
                for learning_rate in cfg['learning_rates']:
                    update = isolated_update(model, tokenizer, ids, parameters, original, gradient, dev_gradient,
                        before, dev_panels, row, learning_rate, cfg['prefix_batch'])
                    update.update(method=method, train_mass_delta=update['train_mass_after'] - mass,
                        maximum_score_delta=max(abs(a - b) for a, b in zip(initial_scores, update.pop('train_scores_after'))))
                    # BF16 resolution is reported separately; it cannot establish FP32 convergence.
                    tolerance = (2e-3 if dtype == 'bfloat16' else 2e-5) + .25 * abs(update['predicted_dev_delta'])
                    update['taylor_pass'] = update['taylor_absolute_error'] <= tolerance
                    steps.append(update)
            reports.append({'dtype': dtype, 'task_id': row['task_id'], 'correct_mass': mass,
                'mixed_probability': 1 - mass ** cfg['group_size'] - (1 - mass) ** cfg['group_size'],
                'observed_mixed_fraction': float(((rewards.sum(1) > 0) & (rewards.sum(1) < cfg['group_size'])).float().mean()),
                'exact_gradient_norm': float(exact.norm()), 'mc_gradient_norm': float(mean.norm()),
                'mc_error_norm': error, 'mc_standard_error_norm': standard_error,
                'mc_pass': error <= 4 * standard_error + 1e-7,
                'mc_block_noise_norm': float(variance.sum().sqrt()),
                'gradient_cosine': float(torch.nn.functional.cosine_similarity(exact, mean, dim=0)),
                'train_score_before': initial_scores, 'dev_function_before': before, 'isolated_steps': steps})
            print(f'{dtype} {row["task_id"]}: MC error={error:.3g}, SE={standard_error:.3g}', flush=True)
        del model, parameters, original, functional
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # CPU-only runs validate the plumbing but never unlock a BF16 scientific pilot.
    numerical = all(r['mc_pass'] and all(s['taylor_pass'] for s in r['isolated_steps']) for r in reports)
    passed = numerical and set(dtypes) == {'float32', 'bfloat16'} and len(train_rows) >= 4
    write_json(out / 'gradient_diagnostic.json', {'binding': binding, 'reports': reports,
        'train_ids': [r['task_id'] for r in train_rows], 'dev_ids': [r['task_id'] for p in dev_panels for r in p],
        'mc_groups': cfg['mc_groups'], 'mc_blocks': cfg['mc_blocks'], 'group_size': cfg['group_size'],
        'dev_usage': 'measurement only; never added to train loss or used to filter individual updates',
        'exact_expectation_scope': 'raw normalized binary reward gradient, before clipping and Adam',
        'interpretation': 'same frozen SFT initialization; distinct from the historical 400-step Base GRPO series'})
    write_json(out / 'gate.json', {'kind': 'reward', 'pass': passed, 'numerical_checks': numerical,
        'dtypes': dtypes, 'minimum_train_tasks': 4, 'scientific_gpu_gate': set(dtypes) == {'float32', 'bfloat16'}})
    finish_run(out, binding)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))


if __name__ == '__main__':
    main()
