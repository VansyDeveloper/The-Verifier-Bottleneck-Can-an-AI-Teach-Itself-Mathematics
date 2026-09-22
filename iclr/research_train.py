"""Panel-matched CE/counterfactual/entropy and gated reward/state pilots."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import time
import traceback

import torch

import composition_core as core
from .common import read_jsonl, write_json
from .modeling import load, seed_all, encode, collate, ce_sum
from .research_data import check_panels, program_exposure
from .research_dsl import DOMAIN, apply_prompt, plan_prompt, trajectory
from .research_io import start_run, finish_run, require_gate, exposure_ledger
from .research_model import deterministic_training, score_task
from .research_objectives import StateProjection, conditional_loss, entropy, sigreg
from .reward_control import exact_reward_loss, sampled_reward_loss

OBJECTIVES = ('atomic', 'ce', 'ce_entropy', 'ce_cf', 'ce_cf_entropy',
              'sampled', 'sampled_entropy', 'exact', 'exact_entropy',
              'state_ce', 'state', 'sigreg', 'state_sigreg')
DEFAULTS = dict(objective='ce', seed=0, epochs=2, panel_batch=1, prefix_batch=32,
                learning_rate=1e-4, cf_weight=.1, tau=1., entropy_weight=.01,
                state_weight=.1, sigreg_weight=.01, state_dimension=32,
                group_size=32, device='cuda', dtype='bfloat16', reward_steps=32)


def config_checked(config):
    allowed = DEFAULTS.keys() | {'base', 'adapter', 'model', 'data', 'output', 'expected_base_hash',
                                'expected_adapter_hash', 'gate', 'atomic_gate', 'initial_training_receipt', 'smoke'}
    if set(config) - allowed:
        raise ValueError(f'Unknown v4 training settings: {set(config) - allowed}')
    cfg = {**DEFAULTS, **config}
    if cfg['objective'] not in OBJECTIVES:
        raise ValueError('Unknown objective')
    for key in ('seed', 'epochs', 'panel_batch', 'prefix_batch', 'state_dimension', 'group_size', 'reward_steps'):
        if type(cfg[key]) is not int or cfg[key] < (0 if key == 'seed' else 2 if key == 'group_size' else 1):
            raise ValueError(f'Invalid integer {key}')
    if cfg['seed'] >= 2**32:
        raise ValueError('seed is out of range')
    for key in ('learning_rate', 'tau', 'cf_weight', 'entropy_weight', 'state_weight', 'sigreg_weight'):
        if not math.isfinite(cfg[key]) or cfg[key] < 0 or key in ('learning_rate', 'tau') and cfg[key] == 0:
            raise ValueError(f'Invalid {key}')
    return cfg


def run(config):
    cfg = config_checked(config)
    manifest, binding, done = start_run(cfg, 'train')
    if done:
        return
    out, data = Path(cfg['output']), Path(cfg['data'])
    method = cfg['objective']
    if method == 'atomic':
        return atomic_warmup(cfg, manifest, binding)
    reward = method.startswith(('sampled', 'exact'))
    state_path = method in ('state_ce', 'state', 'sigreg', 'state_sigreg')
    if reward or state_path:
        require_gate(cfg.get('gate'), 'reward' if reward else 'state', binding)
    seed_all(cfg['seed'])
    model, tokenizer, ids = load(cfg['base'], adapter=cfg.get('adapter'), train=True,
                                device=cfg['device'], dtype=cfg['dtype'], lora_dropout=0.)
    deterministic_training(model)
    device = next(model.parameters()).device
    head = StateProjection(model.config.hidden_size, cfg['state_dimension']).to(device) if state_path else None
    parameters = [p for p in model.parameters() if p.requires_grad] + (list(head.parameters()) if head else [])
    optimizer = torch.optim.AdamW(parameters, lr=cfg['learning_rate'], weight_decay=0., fused=device.type == 'cuda')
    panels = check_panels(read_jsonl(data / manifest['training_panels']))
    if not panels or any(row['family'] != 'TRAIN' or row['depth'] != 3 for panel in panels for row in panel):
        raise ValueError('Training requires complete train-only depth-three panels')
    groups = []
    for epoch in range(cfg['epochs']):
        order = list(panels)
        random.Random(cfg['seed'] + epoch).shuffle(order)
        groups.extend(order[i:i + cfg['panel_batch']] for i in range(0, len(order), cfg['panel_batch']))
    if reward:
        groups = [groups[i % len(groups)] for i in range(cfg['reward_steps'])]
    counts, started = Counter(), time.monotonic()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    probe = panels[0][0]
    try:
        with (out / 'training_metrics.jsonl').open('w') as log, (out / 'training_stream.jsonl').open('w') as stream:
            for step, group in enumerate(groups):
                optimizer.zero_grad(set_to_none=True)
                metrics = Counter()
                for panel in group:
                    kind = panel[0].get('panel_kind', 'four_target')
                    candidates = [panel[0]['program_A'], panel[0]['program_B']] if kind == 'crossed' else [r['witness'] for r in panel]
                    matrix, losses, zs, states, fields, keys = [], [], [], [], [], []
                    for row in panel:
                        programs = list(core.enumerate_programs(3))
                        full, local, prefixes = score_task(model, tokenizer, ids, row, cfg['prefix_batch'], head=head, accounting=counts)
                        witness = programs.index(tuple(row['witness']))
                        logq = local.log_softmax(0)
                        if abs(float(torch.logsumexp(local.detach(), 0))) > 1e-4:
                            raise ValueError('Local policy failed normalization')
                        h = entropy(logq)
                        if reward:
                            correct = torch.tensor([list(p) in row['correct_programs'] for p in programs], device=device)
                            mass = float(logq[correct].exp().sum().detach())
                            draw = torch.multinomial(logq.detach().exp(), cfg['group_size'], replacement=True)
                            loss = (exact_reward_loss(logq, correct, cfg['group_size']) if method.startswith('exact') else
                                    sampled_reward_loss(logq, correct, draw))
                            counts['verifier_queries'] += len(programs) if method.startswith('exact') else len(set(draw.tolist()))
                            counts['diagnostic_reward_labels'] += len(programs)
                            metrics['correct_mass'] += mass / 4
                            metrics['mixed_probability'] += (1 - mass ** cfg['group_size'] - (1 - mass) ** cfg['group_size']) / 4
                        else:
                            loss = -full[witness] / row['depth']
                            counts['supervised_operation_tokens'] += row['depth']
                            counts['training_label_queries'] += 1
                        if method.endswith('entropy'):
                            loss = loss - cfg['entropy_weight'] * h
                        matrix.append(local[[programs.index(tuple(p)) for p in candidates]])
                        losses.append(loss)
                        metrics['entropy'] += float(h.detach()) / 4
                        if head:
                            true_states = trajectory(row, row['witness'])
                            lookup = {r['prefix']: r['representation'] for r in prefixes}
                            for i in range(row['depth']):
                                zs.append(lookup[tuple(row['witness'][:i])])
                                states.append(true_states[i]); fields.append(row['p'])
                                keys.append((row['p'], tuple(true_states[i])))
                        stream.write(json.dumps({'step': step, 'panel_id': row['panel_id'], 'task_id': row['task_id'],
                            'program': row['witness'], 'depth': row['depth'], 'kind': kind,
                            'sampled_programs': [list(programs[i]) for i in draw.tolist()] if reward and method.startswith('sampled') else None}) + '\n')
                        counts['example_exposures'] += 1
                    loss = torch.stack(losses).mean()
                    if '_cf' in method:
                        cf = conditional_loss(torch.stack(matrix), kind, cfg['tau'])
                        loss = loss + cfg['cf_weight'] * cf
                        metrics['counterfactual_loss'] += float(cf.detach())
                    if head:
                        z = torch.stack(zs)
                        auxiliary, accuracy = head.state_loss(z, states, fields)
                        regularizer = sigreg(z, keys, cfg['seed'] + step)
                        if method in ('state', 'state_sigreg'):
                            loss = loss + cfg['state_weight'] * auxiliary
                        if method in ('sigreg', 'state_sigreg'):
                            loss = loss + cfg['sigreg_weight'] * regularizer
                        metrics.update(accuracy)
                        metrics['state_loss'] += float(auxiliary.detach())
                        metrics['sigreg'] += float(regularizer.detach())
                    (loss / len(group)).backward()
                    metrics['loss'] += float(loss.detach()) / len(group)
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
                log.write(json.dumps({'step': step + 1, 'gradient_norm': float(norm),
                                       'seconds': time.monotonic() - started, **metrics}, allow_nan=False) + '\n')
                log.flush(); stream.flush()
                print(f'{method}: step {step + 1}/{len(groups)} loss={metrics["loss"]:.5f}', flush=True)
        with torch.no_grad():
            before = score_task(model, tokenizer, ids, probe, cfg['prefix_batch'], head=head)[1].cpu()
        model.save_pretrained(out / 'adapter'); tokenizer.save_pretrained(out / 'adapter')
        if head:
            write_json(out / 'adapter/state_projection.json', head.config)
            torch.save(head.state_dict(), out / 'adapter/state_projection.pt')
        write_json(out / 'budget.json', {**counts, 'optimizer_steps': len(groups),
            'wall_seconds': time.monotonic() - started, 'policy': 'local_prefix_v1', 'temperature': 1,
            'training_panels': manifest['training_panels'], 'initialization_hash': binding['base_hash'],
            'initial_adapter_hash': binding['adapter_hash'], 'continuation_seed': cfg['seed'],
            'max_training_depth': 3, 'ce': 'mean full-vocabulary operation-token NLL; fixed-length program, no EOS target',
            'dropout': 0, 'kl_beta': 0, 'clip_norm': 1, 'optimizer': 'AdamW, zero initial moments',
            'peak_memory_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
            'runtime_verifier_calls': 0,
            'verifier_scope': 'objective label lookups in precomputed labels; diagnostic full-set lookups are separate; dataset-generation interpreter work is in the CPU audit',
            'exact_is_diagnostic_oracle': reward})
        del optimizer, parameters, model, head, loss, full, local, prefixes, logq, matrix, losses
        if state_path:
            del z, zs, auxiliary, regularizer, lookup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        from .research_model import load_head
        model, tokenizer, ids = load(cfg['base'], adapter=str(out / 'adapter'), device=cfg['device'], dtype=cfg['dtype'])
        head = load_head(out / 'adapter', device)
        with torch.no_grad():
            after = score_task(model, tokenizer, ids, probe, cfg['prefix_batch'], head=head)[1].cpu()
        tolerance = .1 if cfg['dtype'] == 'bfloat16' else 1e-4
        torch.testing.assert_close(before, after, atol=tolerance, rtol=0)
        write_json(out / 'reload_check.json', {'maximum_error': float((before - after).abs().max()), 'tolerance': tolerance})
        stage = {'stage': 'current_continuation', 'domain': manifest.get('domain', 'affine_polynomial_v1'),
            'objective': method, **program_exposure(r['witness'] for group in groups for panel in group for r in panel)}
        write_json(out / 'exposure.json', exposure_ledger(manifest, [*binding['exposure']['stages'], stage]))
        finish_run(out, binding, payload=True)
    except Exception:
        (out / 'FAILED').write_text(traceback.format_exc())
        raise


def atomic_warmup(cfg, manifest, binding):
    """One shared list-DSL PLAN/APPLY initialization, without any composition labels."""
    rows = read_jsonl(Path(cfg['data']) / 'atomic_train.jsonl')
    if (manifest.get('domain') != DOMAIN or not rows
            or any(r['depth'] != 1 or r['family'] != 'ATOMIC' or r['p'] not in manifest['atomic_warmup_fields'] for r in rows)
            or {r['witness'][0] for r in rows} != set(core.OPS)):
        raise ValueError('Atomic warm-up needs all list-DSL operations on allowed train fields only')
    seed_all(cfg['seed'])
    model, tokenizer, ids = load(cfg['base'], adapter=cfg.get('adapter'), train=True,
                               device=cfg['device'], dtype=cfg['dtype'], lora_dropout=0.)
    deterministic_training(model)
    device = next(model.parameters()).device
    examples = [encode(tokenizer, {'prompt': prompt({**row, 'program': row['witness']}), 'answer': answer,
                'task_id': row['task_id'], 'kind': kind}) for row in rows for kind, prompt, answer in
                [('PLAN', plan_prompt, core.program_answer(row['witness'])), ('APPLY', apply_prompt, core.format_state(row['target']))]]
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg['learning_rate'], weight_decay=0., fused=device.type == 'cuda')
    out, counts, started, steps = Path(cfg['output']), Counter(), time.monotonic(), 0
    with (out / 'training_stream.jsonl').open('w') as stream:
        for epoch in range(cfg['epochs']):
            order = list(examples); random.Random(cfg['seed'] + epoch).shuffle(order)
            for offset in range(0, len(order), cfg['prefix_batch']):
                group = order[offset:offset + cfg['prefix_batch']]
                optimizer.zero_grad(set_to_none=True)
                loss = ce_sum(model, collate(tokenizer, group, device, include_target_types=True), counts) / sum(e['target_tokens'] for e in group)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step(); steps += 1
                for e in group:
                    stream.write(json.dumps({'epoch': epoch, 'step': steps, 'task_id': e['task_id'], 'kind': e['kind']}) + '\n')
    with torch.no_grad():
        before = score_task(model, tokenizer, ids, rows[0], cfg['prefix_batch'])[1].cpu()
    model.save_pretrained(out / 'adapter'); tokenizer.save_pretrained(out / 'adapter')
    write_json(out / 'budget.json', {**counts, 'optimizer_steps': steps, 'example_exposures': len(examples) * cfg['epochs'],
        'epochs': cfg['epochs'], 'fields': sorted({r['p'] for r in rows}), 'max_training_depth': 1,
        'wall_seconds': time.monotonic() - started, 'purpose': 'shared atomic PLAN/APPLY preparation, no composition supervision'})
    # Polynomial history describes different operation semantics; retain it explicitly outside this domain's pair ledger.
    ledger = exposure_ledger(manifest, [{'stage': 'list_dsl_atomic_warmup', 'domain': DOMAIN,
        **program_exposure(r['witness'] for _ in range(cfg['epochs'] * 2) for r in rows)}])
    ledger['previous_domain_exposure'] = binding['exposure']
    write_json(out / 'exposure.json', ledger)
    del loss, optimizer, parameters, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    model, tokenizer, ids = load(cfg['base'], adapter=str(out / 'adapter'), device=cfg['device'], dtype=cfg['dtype'])
    with torch.no_grad():
        after = score_task(model, tokenizer, ids, rows[0], cfg['prefix_batch'])[1].cpu()
    tolerance = .1 if cfg['dtype'] == 'bfloat16' else 1e-4
    torch.testing.assert_close(before, after, atol=tolerance, rtol=0)
    write_json(out / 'reload_check.json', {'maximum_error': float((before - after).abs().max()), 'tolerance': tolerance})
    finish_run(out, binding, payload=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))


if __name__ == '__main__':
    main()
