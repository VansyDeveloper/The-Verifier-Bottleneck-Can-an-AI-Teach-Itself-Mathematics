"""Panel-matched CE/counterfactual/entropy and gated reward/state pilots."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time
import traceback

import torch

import composition_core as core
from .common import file_hash, read_jsonl, tree_hash, verify_receipt, write_json
from .modeling import load, seed_all, encode, collate, ce_sum
from .research_data import check_panels, program_exposure
from .research_dsl import DOMAIN, apply_prompt, plan_prompt, trajectory
from .research_io import start_run, finish_run, require_gate, exposure_ledger
from .research_model import deterministic_training, evaluation_context, restore_rng, rng_state, score_task
from .research_objectives import StateProjection, conditional_loss, entropy, sigreg
from .reward_control import exact_reward_loss, sampled_reward_loss

OBJECTIVES = ('atomic', 'ce', 'ce_entropy', 'ce_cf', 'ce_cf_entropy',
              'sampled', 'sampled_entropy', 'exact', 'exact_entropy',
              'state_ce', 'state', 'sigreg', 'state_sigreg')
DEFAULTS = dict(objective='ce', seed=0, epochs=2, panel_batch=1, prefix_batch=32,
                learning_rate=1e-4, cf_weight=.1, tau=1., entropy_weight=.01,
                state_weight=.1, sigreg_weight=.01, state_dimension=32,
                group_size=32, device='cuda', dtype='bfloat16', reward_steps=32,
                replay_weight=0., max_steps=None, checkpoint_steps=[], monitor=False)
LOG_NAMES = ('training_metrics.jsonl', 'training_stream.jsonl', 'replay_stream.jsonl')


def snapshot_logs(out):
    snapshot = {}
    for name in LOG_NAMES:
        path = Path(out) / name
        contents = path.read_bytes() if path.exists() else b''
        snapshot[name] = {'bytes': len(contents), 'sha256': hashlib.sha256(contents).hexdigest()}
    return snapshot


def restore_logs(out, snapshot):
    if not isinstance(snapshot, dict) or set(snapshot) != set(LOG_NAMES):
        raise ValueError('Checkpoint has no complete committed-log snapshot; do not guess recovery boundaries')
    # Validate every committed prefix before changing any live log.
    for name in LOG_NAMES:
        entry = snapshot[name]
        if not isinstance(entry, dict) or type(entry.get('bytes')) is not int or entry['bytes'] < 0:
            raise ValueError(f'Invalid committed log boundary: {name}')
        path, size = Path(out) / name, entry['bytes']
        if path.exists():
            with path.open('rb') as stream:
                prefix = stream.read(size)
        else:
            prefix = b''
        if len(prefix) != size or hashlib.sha256(prefix).hexdigest() != entry.get('sha256'):
            raise ValueError(f'Committed log prefix is missing or damaged: {name}')
    for name in LOG_NAMES:
        path = Path(out) / name
        if path.exists():
            with path.open('r+b') as stream:
                stream.truncate(snapshot[name]['bytes'])


def complete_checkpoint_receipts(out):
    return sorted(p for p in (Path(out) / 'checkpoints').glob('step_*/DONE')
                  if p.parent.name.removeprefix('step_').isdigit())


def config_checked(config):
    allowed = DEFAULTS.keys() | {'base', 'adapter', 'model', 'data', 'output', 'expected_base_hash',
                                'expected_adapter_hash', 'gate', 'atomic_gate', 'initial_training_receipt', 'smoke',
                                'plan', 'amendment', 'base_training_receipt'}
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
    for key in ('learning_rate', 'tau', 'cf_weight', 'entropy_weight', 'state_weight', 'sigreg_weight', 'replay_weight'):
        if not math.isfinite(cfg[key]) or cfg[key] < 0 or key in ('learning_rate', 'tau') and cfg[key] == 0:
            raise ValueError(f'Invalid {key}')
    if cfg['max_steps'] is not None and (type(cfg['max_steps']) is not int or cfg['max_steps'] < 1):
        raise ValueError('max_steps must be a positive composition update budget')
    if (not isinstance(cfg['checkpoint_steps'], list) or any(type(s) is not int or s < 0 for s in cfg['checkpoint_steps'])
            or cfg['checkpoint_steps'] != sorted(set(cfg['checkpoint_steps']))):
        raise ValueError('checkpoint_steps must be sorted distinct nonnegative integers')
    if (cfg['monitor'] or cfg['replay_weight']) and not cfg.get('amendment'):
        raise ValueError('Replay/monitor requires a frozen amendment')
    if cfg['replay_weight'] and (cfg['panel_batch'] != 1 or cfg['objective'] not in ('ce', 'ce_cf', 'ce_entropy', 'ce_cf_entropy')):
        raise ValueError('Replay pilot requires one panel and a supervised CE/CF/H objective')
    if cfg['objective'] == 'atomic' and (cfg['checkpoint_steps'] or cfg['max_steps'] or cfg['monitor']):
        raise ValueError('Step checkpoints apply to panel continuation, not the legacy atomic warmup')
    return cfg


def checkpoint(out, step, model, tokenizer, head, optimizer, counts, binding, manifest, planned_steps):
    directory = out / 'checkpoints' / f'step_{step:06d}'
    if directory.exists():
        raise FileExistsError(f'Checkpoint is immutable: {directory}')
    temporary = directory.with_name(directory.name + '.partial')
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    state = rng_state()
    try:
        model.save_pretrained(temporary / 'adapter'); tokenizer.save_pretrained(temporary / 'adapter')
        if head is not None:
            write_json(temporary / 'adapter/state_projection.json', head.config)
            torch.save(head.state_dict(), temporary / 'adapter/state_projection.pt')
        torch.save({'optimizer': optimizer.state_dict(), 'rng': state, 'next_step': step,
                    'counts': dict(counts), 'logs': snapshot_logs(out)}, temporary / 'resume_state.pt')
        write_json(temporary / 'binding.json', binding)
        write_json(temporary / 'budget.json', {**counts, 'optimizer_steps': step, 'completed': False,
                    'planned_steps': planned_steps, 'cursor': {'next_composition_step': step, 'next_replay_step': step}})
        rows = read_jsonl(out / 'training_stream.jsonl') if (out / 'training_stream.jsonl').exists() else []
        stage = {'stage': 'current_continuation', 'provenance_status': 'verified',
                 **program_exposure(r['program'] for r in rows if r['step'] < step)}
        if binding['config']['replay_weight']:
            stage['atomic_replay_examples'] = counts['replay_examples']
            stage['depths'][1] = counts['replay_examples']
        write_json(temporary / 'exposure.json', exposure_ledger(manifest, [*binding['exposure']['stages'], stage]))
        finish_run(temporary, binding, payload=True)
        receipt = json.loads((temporary / 'DONE').read_text())
        receipt.update(checkpoint_step=step, completed_training=False,
                       parent_initialization={'base_hash': binding['base_hash'], 'adapter_hash': binding['adapter_hash']},
                       resume_state_hash=file_hash(temporary / 'resume_state.pt'))
        write_json(temporary / 'DONE', receipt)
        temporary.rename(directory)
    finally:
        restore_rng(state)
    return directory


def run(config, *, resume_from=None, stop_after=None):
    config = dict(config)
    configured_resume = config.pop('resume_from', None)
    resume_from = resume_from if resume_from is not None else configured_resume
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
    if resume_from == 'latest':
        snapshots = complete_checkpoint_receipts(out)
        resume_from = str(snapshots[-1].parent) if snapshots else None
    resumed = None
    if resume_from:
        if Path(resume_from).name.endswith('.partial'):
            raise ValueError('Resume requires a published checkpoint, not a .partial directory')
        resumed = verify_receipt(resume_from)
        if (resumed['binding'] != binding
                or tree_hash(Path(resume_from) / 'adapter') != resumed['payload_hash']
                or file_hash(Path(resume_from) / 'resume_state.pt') != resumed['resume_state_hash']):
            raise ValueError('Resume checkpoint differs from the exact run or its saved state')
        later = complete_checkpoint_receipts(out)
        if later and later[-1].parent.name > Path(resume_from).name:
            raise ValueError('Resume the latest immutable checkpoint; use a new run for an independent restart')
    elif (out / 'training_metrics.jsonl').exists() and (out / 'training_metrics.jsonl').stat().st_size:
        raise ValueError('Partial run exists: explicitly resume a checkpoint or choose a new output for a restart')
    seed_all(cfg['seed'])
    model, tokenizer, ids = load(cfg['base'], adapter=str(Path(resume_from) / 'adapter') if resumed else cfg.get('adapter'), train=True,
                                device=cfg['device'], dtype=cfg['dtype'], lora_dropout=0.)
    deterministic_training(model)
    device = next(model.parameters()).device
    head = StateProjection(model.config.hidden_size, cfg['state_dimension']).to(device) if state_path else None
    if resumed and head is not None:
        head.load_state_dict(torch.load(Path(resume_from) / 'adapter/state_projection.pt', weights_only=True, map_location=device))
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
    if cfg['max_steps']:
        if cfg['max_steps'] > len(groups):
            raise ValueError('max_steps exceeds the declared epoch stream; no implicit repeated training')
        groups = groups[:cfg['max_steps']]
    replay = []
    if cfg['replay_weight']:
        for row in read_jsonl(Path(cfg['amendment']).parent / 'atomic_replay.jsonl'):
            replay.append([encode(tokenizer, {'prompt': prompt({**row, 'program': row['witness']}), 'answer': answer,
                'task_id': row['task_id'], 'kind': kind, 'operation': row['witness'][0], 'p': row['p']})
                for kind, prompt, answer in [('PLAN', plan_prompt, core.program_answer(row['witness'])),
                                             ('APPLY', apply_prompt, core.format_state(row['target']))]])
    counts, started = Counter(), time.monotonic()
    first_step = 0
    if resumed:
        state = torch.load(Path(resume_from) / 'resume_state.pt', weights_only=False, map_location='cpu')
        if state['next_step'] != resumed['checkpoint_step'] or not 0 <= state['next_step'] <= len(groups):
            raise ValueError('Invalid saved stream cursor')
        optimizer.load_state_dict(state['optimizer'])
        first_step, counts = state['next_step'], Counter(state['counts'])
        restore_rng(state['rng'])
        restore_logs(out, state.get('logs'))
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    probe = panels[0][0]
    def monitor_step(path, step):
        if cfg['monitor']:
            from .research_evaluate import monitor_model
            monitor_out = out / 'monitors' / f'step_{step:06d}'
            monitor_model(model, tokenizer, ids, cfg, monitor_out, head)
            finish_run(monitor_out, {**binding, 'kind': 'monitor', 'checkpoint_step': step,
                'training_receipt_hash': file_hash(path / 'DONE'), 'payload_hash': tree_hash(path / 'adapter')})
    def save_step(step):
        path = checkpoint(out, step, model, tokenizer, head, optimizer, counts, binding, manifest, len(groups))
        monitor_step(path, step)
    try:
        if resumed and cfg['monitor'] and not (out / 'monitors' / f'step_{first_step:06d}' / 'DONE').exists():
            monitor_step(Path(resume_from), first_step)
        if not resumed and 0 in cfg['checkpoint_steps']:
            save_step(0)
        with (out / 'training_metrics.jsonl').open('a' if resumed else 'w') as log, (out / 'training_stream.jsonl').open('a' if resumed else 'w') as stream, (out / 'replay_stream.jsonl').open('a' if resumed else 'w') as replay_log:
            for step in range(first_step, len(groups)):
                group = groups[step]
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
                        metrics['ce_full_vocab'] += float((-full[witness] / row['depth']).detach()) / 4 / len(group)
                        metrics['ce_local_actions'] += float((-local[witness] / row['depth']).detach()) / 4 / len(group)
                        metrics['ce_legal_gate'] += float((-(full[witness]-local[witness]) / row['depth']).detach()) / 4 / len(group)
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
                        metrics['entropy_raw'] += float(h.detach()) / 4 / len(group)
                        metrics['entropy_weighted'] += (cfg['entropy_weight'] if method.endswith('entropy') else 0.) * float(h.detach()) / 4 / len(group)
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
                    cf = conditional_loss(torch.stack(matrix), kind, cfg['tau'])
                    metrics['counterfactual_loss_raw'] += float(cf.detach()) / len(group)
                    metrics['counterfactual_loss_weighted'] += (cfg['cf_weight'] if '_cf' in method else 0.) * float(cf.detach()) / len(group)
                    if '_cf' in method:
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
                metrics['composition_loss'] = metrics['loss']
                if replay:
                    atomic_losses = []
                    for example in replay[step % len(replay)]:
                        loss_atomic = ce_sum(model, collate(tokenizer, [example], device, include_target_types=True)) / example['target_tokens']
                        atomic_losses.append(loss_atomic)
                        kind = example['kind'].lower()
                        counts[f'replay_{kind}_tokens'] += example['target_tokens']
                        counts['replay_forward_tokens'] += len(example['input_ids'])
                        counts['total_forward_tokens'] += len(example['input_ids'])
                        counts['replay_examples'] += 1
                        counts[f'replay_{kind}_examples'] += 1
                        metrics[f'replay_{kind}_ce'] = float(loss_atomic.detach())
                        replay_log.write(json.dumps({'step': step + 1, **{k: example[k] for k in ('task_id', 'kind', 'operation', 'p', 'target_tokens')}}) + '\n')
                    atomic_loss = torch.stack(atomic_losses).mean() * cfg['replay_weight']
                    atomic_loss.backward()
                    metrics['replay_weighted'] = float(atomic_loss.detach())
                    metrics['loss'] += float(atomic_loss.detach())
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                before_update = [p.detach().clone() for p in parameters]
                optimizer.step()
                update_norm = math.sqrt(sum(float((p.detach().float() - old.float()).square().sum()) for p, old in zip(parameters, before_update)))
                del before_update
                log.write(json.dumps({'step': step + 1, 'gradient_norm': float(norm), 'gradient_norm_before_clip': float(norm),
                                       'clipping_factor': min(1., 1. / (float(norm) + 1e-6)), 'actual_parameter_update_norm': update_norm,
                                       'seconds': time.monotonic() - started, **metrics}, allow_nan=False) + '\n')
                log.flush(); stream.flush(); replay_log.flush()
                print(f'{method}: step {step + 1}/{len(groups)} loss={metrics["loss"]:.5f}', flush=True)
                if step + 1 in cfg['checkpoint_steps'] or stop_after == step + 1:
                    save_step(step + 1)
                if stop_after == step + 1 and step + 1 < len(groups):
                    return  # Deliberate interruption leaves checkpoint receipts, no completed run receipt.
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
            'replay_weight': cfg['replay_weight'], 'amendment_hash': binding.get('amendment_hash'),
            'composition_examples': counts['example_exposures'],
            'replay_example_fraction': counts['replay_examples'] / (counts['replay_examples'] + counts['example_exposures']),
            'replay_loss_tokens': counts['replay_plan_tokens'] + counts['replay_apply_tokens'],
            'replay_objective': 'weight * 0.5 * (mean-token PLAN CE + mean-token APPLY CE), each includes EOS',
            'dropout': 0, 'kl_beta': 0, 'clip_norm': 1, 'optimizer': 'AdamW; moments restored on resume',
            'peak_memory_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
            'runtime_verifier_calls': 0,
            'verifier_scope': 'objective label lookups in precomputed labels; diagnostic full-set lookups are separate; dataset-generation interpreter work is in the CPU audit',
            'exact_is_diagnostic_oracle': reward})
        del optimizer, parameters, model, head
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
        stage = {'stage': 'current_continuation', 'provenance_status': 'verified', 'domain': manifest.get('domain', 'affine_polynomial_v1'),
            'objective': method, **program_exposure(r['witness'] for group in groups for panel in group for r in panel)}
        if replay:
            stage['depths'][1] = counts['replay_examples']
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
    ledger = exposure_ledger(manifest, [{'stage': 'list_dsl_atomic_warmup', 'domain': DOMAIN, 'provenance_status': 'verified',
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
    parser.add_argument('--resume-from', help='Exact checkpoint directory, or latest in the same output')
    parser.add_argument('--stop-after', type=int, help='Plumbing/interruption check; leaves an incomplete run')
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), resume_from=args.resume_from, stop_after=args.stop_after)


if __name__ == '__main__':
    main()
