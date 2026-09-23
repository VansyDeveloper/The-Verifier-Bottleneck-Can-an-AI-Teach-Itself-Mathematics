"""V4 complete-space metrics, entropy accounting, and paired state-access diagnosis."""

import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

import composition_core as core
from .common import ROOT, code_hash, file_hash, read_jsonl, write_json
from .modeling import load, seed_all
from .research_data import check_panels
from .research_dsl import DOMAIN, apply_prompt, plan_prompt, trajectory, verify
from .research_io import start_run, finish_run
from .research_model import evaluation_context, load_head, score_task, solve
from .research_objectives import assignment_cycles, conditional_loss, entropy_report, interaction
from .upgrade_evaluate import execute_programs


def panel_metrics(rows):
    result = []
    for panel in check_panels(rows):
        crossed = panel[0].get('panel_kind') == 'crossed'
        candidates = [panel[0]['program_A'], panel[0]['program_B']] if crossed else [r['witness'] for r in panel]
        programs = list(core.enumerate_programs(panel[0]['depth']))
        indices = [programs.index(tuple(p)) for p in candidates]
        for policy in ('local', 'full'):
            matrix = torch.tensor([[row[policy][i] for i in indices] for row in panel], dtype=torch.float64)
            item = {k: panel[0][k] for k in ('panel_id', 'family', 'depth', 'p')}
            item.update(policy=policy, kind='crossed' if crossed else 'four_target',
                        starts=[row['start'] for row in panel], task_ids=[row['task_id'] for row in panel],
                        score_matrix=matrix.tolist())
            if crossed:
                desired = torch.tensor([1., -1., -1., 1.])
                margin = (matrix[:, 0] - matrix[:, 1]) * desired
                credit = (margin > 1e-8).double() + .5 * (margin.abs() <= 1e-8)
                item.update(interaction=float(interaction(matrix)), cell_accuracy=float(credit.mean()),
                            joint_accuracy=float(credit.prod()), all_cells_strict=bool((margin > 1e-8).all()))
                d = matrix[:, 0] - matrix[:, 1]
                item.update(signed_margins=margin.tolist(), minimum_margin=float(margin.min()),
                    interaction_positive=bool(interaction(matrix) > 1e-8),
                    nuisance=dict(a=float(d.sum()/4), b=float((d[0]+d[1]-d[2]-d[3])/4),
                                  c=float((d[0]-d[1]+d[2]-d[3])/4), h=float(interaction(matrix)/4)),
                    scalar_offset_gap=float(torch.min(d[[0, 3]]) - torch.max(d[[1, 2]])))
            else:
                cycles = assignment_cycles(matrix)
                item['pair_accuracy'] = float(((cycles > 1e-8).double() + .5 * (cycles.abs() <= 1e-8)).mean())
                item['mean_cycle'] = float(cycles.mean())
                assignments = [sum(matrix[i, j] for i, j in enumerate(order)).item() for order in itertools.permutations(range(4))]
                best = max(assignments)
                item['assignment_accuracy'] = (1 / sum(abs(v - best) <= 1e-8 for v in assignments)
                                               if abs(assignments[0] - best) <= 1e-8 else 0.)
            result.append(item)
    return result


def bootstrap(values, seed=20260922, draws=2000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError('Cannot bootstrap empty panels')
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(draws, len(values)))].mean(1)
    return list(map(float, np.quantile(means, [.025, .975])))


def evaluation_rows(data, phase):
    rows = []
    for pattern in (f'{phase}_[ABCD].jsonl', f'{phase}4_[ABCD].jsonl',
                    f'{phase}_panel_[ABCD].jsonl', f'{phase}_crossed_[ABCD].jsonl'):
        for path in sorted(Path(data).glob(pattern)):
            rows.extend(read_jsonl(path))
    return rows


def atomic_results(model, tokenizer, ids, rows, prefix_batch, head=None):
    with torch.no_grad():
        planned = [{**row, 'operation': row['witness'][0], 'prompt': plan_prompt(row),
                    'prompt_token_ids': tokenizer.encode(plan_prompt(row), add_special_tokens=False),
                    **solve(model, tokenizer, ids, row, head=head)} for row in rows]
        applied = execute_programs(model, tokenizer, [{**r, 'kind': 'atomic', 'program': r['witness']} for r in rows],
                                   min(8, prefix_batch), prompt_builder=apply_prompt)
        for row in applied:
            row['prompt'] = apply_prompt({**row, 'program': row['witness']})
            row['prompt_token_ids'] = tokenizer.encode(row['prompt'], add_special_tokens=False)
    return {'plan': planned, 'apply': applied}


def atomic_summary(result):
    return {op: {mode: {'n': len(rows),
        'accuracy': float(np.mean([r['correct' if mode == 'plan' else 'semantic_correct'] for r in rows])) if rows else None,
        'parse_rate': float(np.mean([r['parse_ok'] for r in rows])) if rows else None}
        for mode in ('plan', 'apply')
        for rows in [[r for r in result[mode] if (r.get('operation') or r['witness'][0]) == op]]}
        for op in core.OPS}


def monitor_model(model, tokenizer, ids, cfg, out, head=None):
    data, out = Path(cfg['data']), Path(out)
    manifest = json.loads((data / 'manifest.json').read_text())
    selected = json.loads(Path(cfg['amendment']).read_text())['monitor']
    wanted = set(selected['ordinary'] + selected['crossed'])
    rows = [r for r in evaluation_rows(data, 'dev') if r['task_id'] in wanted]
    rows += [r for r in read_jsonl(data / manifest['training_panels']) if r['task_id'] in set(selected['train'])]
    saved, metrics = [], defaultdict(list)
    with evaluation_context(model, head):
        for row in rows:
            full, local, _ = score_task(model, tokenizer, ids, row, cfg['prefix_batch'], head=head)
            programs = list(core.enumerate_programs(row['depth']))
            correct = np.array([verify(row, p) for p in programs])
            scores = local.cpu().numpy()
            q = torch.softmax(local, 0).cpu().numpy()
            order = sorted(range(len(programs)), key=lambda i: (-scores[i], tuple(programs[i])))
            rank = min(np.flatnonzero(correct[order])) + 1
            record = {**row, 'full': full.cpu().tolist(), 'local': scores.tolist(),
                      'hit1': float(rank <= 1), 'hit8': float(rank <= 8), 'correct_mass': float(q[correct].sum())}
            saved.append(record)
            kind = 'train_probe' if row['family'] == 'TRAIN' else 'crossed' if row.get('panel_id') else 'ordinary'
            metrics[row['family'], kind].append(record)
        atomic_rows = [r for r in read_jsonl(data / 'dev_atomic.jsonl') if r['task_id'] in set(selected['atomic'])]
        atomic = atomic_results(model, tokenizer, ids, atomic_rows, cfg['prefix_batch'], head)
    panels = panel_metrics([r for r in saved if r.get('panel_id')])
    for panel in panels:
        panel['cf_loss'] = float(conditional_loss(torch.tensor(panel['score_matrix'], dtype=torch.float64), panel['kind'], cfg.get('tau', 1.)))
    report = {'phase': 'dev', 'mode': 'monitor', 'selection': selected,
        'scope': 'TRAIN probes measure fitting; dev families remain separate; constrained PLAN parse is imposed',
        'atomic': atomic_summary(atomic), 'panels': panels,
        'ordinary': [{'family': f, 'kind': k, 'n': len(v), **{metric: float(np.mean([r[metric] for r in v]))
                     for metric in ('hit1', 'hit8', 'correct_mass')}} for (f, k), v in metrics.items()]}
    write_json(out / 'monitor.json', report)
    write_json(out / 'atomic_retention.json', {'phase': 'dev', **atomic})
    (out / 'rankings.jsonl').write_bytes(core.canonical_jsonl_bytes(saved))
    return report


def run(config):
    cfg = {**dict(phase='dev', seed=0, arm='baseline', device='cuda', dtype='bfloat16', prefix_batch=32,
                  solve_per_family=8, mode='full'), **config}
    if cfg['mode'] not in ('full', 'monitor') or cfg['mode'] == 'monitor' and (cfg['phase'] != 'dev' or not cfg.get('amendment')):
        raise ValueError('monitor needs a frozen dev amendment; full uses the original complete evaluator')
    manifest, binding, done = start_run(cfg, 'evaluate')
    if done:
        return
    if cfg['phase'] not in ('dev', 'final'):
        raise ValueError('Unknown evaluation phase')
    if cfg['phase'] == 'final':
        if not cfg.get('analysis_lock'):
            raise ValueError('Final requires a v4 immutable analysis lock')
        lock = json.loads(Path(cfg['analysis_lock']).read_text())
        if (lock['schema'] != 'iclr.research.lock.' + cfg.get('plan', 'v4') or lock['code_hash'] != code_hash()
                or lock['plan_hash'] != binding['plan_hash']
                or binding['data_hash'] not in lock['dataset_hashes']
                or [binding['base_hash'], binding['adapter_hash']] not in lock['checkpoints']):
            raise ValueError('Final lock differs from data, code, or selected checkpoints')
    data, out = Path(cfg['data']), Path(cfg['output'])
    seed_all(0 if cfg['seed'] == -1 and cfg['arm'] in ('baseline', 'initial') else cfg['seed'])
    model, tokenizer, ids = load(cfg['base'], adapter=cfg.get('adapter'), device=cfg['device'], dtype=cfg['dtype'])
    head = load_head(cfg.get('adapter'), next(model.parameters()).device)
    if cfg['mode'] == 'monitor':
        monitor_model(model, tokenizer, ids, cfg, out, head)
        finish_run(out, binding)
        return
    rows, summaries, budget = [], defaultdict(list), defaultdict(int)
    started = time.monotonic()
    with torch.no_grad(), (out / 'rankings.jsonl').open('w') as raw, (out / 'prefixes.jsonl').open('w') as prefixes_file:
        for row in evaluation_rows(data, cfg['phase']):
            full, local, prefixes = score_task(model, tokenizer, ids, row, cfg['prefix_batch'], head=head, accounting=budget)
            saved = {**row, 'full': full.cpu().tolist(), 'local': local.cpu().tolist()}
            programs = core.enumerate_programs(row['depth'])
            correct = [verify(row, p) for p in programs]
            saved['entropy'] = {policy: entropy_report(saved[policy], correct, programs, row['p'], row['degree'], affine=row.get('domain') != DOMAIN)
                                for policy in ('local', 'full')}
            if head:
                ablated_full, ablated, _ = score_task(model, tokenizer, ids, row, cfg['prefix_batch'], head=head, ablate=True, accounting=budget)
                saved['ablated_local'] = ablated.cpu().tolist()
                saved['ablated_full'] = ablated_full.cpu().tolist()
                chosen = tuple(row['witness'])
                selected = {r['prefix']: r['representation'] for r in prefixes}
                z = torch.stack([selected[chosen[:i]] for i in range(row['depth'])])
                states = trajectory(row, chosen)[:-1]
                _, saved['state_prediction'] = head.state_loss(z, states, [row['p']] * len(states))
            raw.write(json.dumps(saved, allow_nan=False) + '\n')
            prefixes_file.write(json.dumps({'task_id': row['task_id'], 'prefixes': [
                {'prefix': r['prefix'], 'full_action_logprobs': r['full'].cpu().tolist()} for r in prefixes]}) + '\n')
            rows.append(saved)
            kind = row.get('panel_kind', 'four_target' if row.get('panel_id') else 'ordinary')
            for policy, metrics in saved['entropy'].items():
                for item in metrics:
                    summaries[row['family'], row['depth'], kind, policy, item['temperature']].append(item)
    panels = panel_metrics([r for r in rows if r.get('panel_id')])
    write_json(out / 'panel_metrics.json', {'binding': binding, 'panels': panels})
    if head:
        ablated = [{**r, 'local': r['ablated_local'], 'full': r['ablated_full']} for r in rows if r.get('panel_id')]
        write_json(out / 'projection_ablation.json', {'binding': binding, 'panels': panel_metrics(ablated),
            'intervention': 'zero compact state representation only on the action path; same model and inputs'})
    summary = [{'family': f, 'depth': d, 'kind': k, 'policy': p, 'temperature': t, 'n': len(items),
                **{key: float(np.mean([r[key] for r in items])) for key in items[0] if key != 'temperature'}}
               for (f, d, k, p, t), items in summaries.items()]
    write_json(out / 'summary.json', {'binding': binding, 'metrics': summary,
                                    'temperature_control': 'positive full-program score scaling preserves ranks and the sign of I'})
    counts, solutions = defaultdict(int), []
    with torch.no_grad():
        for row in rows:
            key = row['family'], row['depth']
            if row.get('panel_id') or counts[key] >= cfg['solve_per_family']:
                continue
            counts[key] += 1
            solutions.append({'task_id': row['task_id'], 'family': row['family'], 'depth': row['depth'],
                'greedy': solve(model, tokenizer, ids, row, head=head),
                'free': solve(model, tokenizer, ids, row, head=head, free=True)})
        atomic = read_jsonl(data / (cfg['phase'] + '_atomic.jsonl'))
        retained = atomic_results(model, tokenizer, ids, atomic, cfg['prefix_batch'], head)
    write_json(out / 'solutions.json', solutions)
    write_json(out / 'atomic_retention.json', {'phase': cfg['phase'], 'summary': atomic_summary(retained), **retained})
    write_json(out / 'budget.json', {**budget, 'wall_seconds': time.monotonic() - started,
        'scored_tasks': len(rows), 'full_candidate_spaces': sorted({5 ** r['depth'] for r in rows}),
        'solution_forward_tokens': sum(s[k]['forward_tokens'] for s in solutions for k in ('greedy', 'free')),
        'atomic_tasks': len(atomic), 'ordinary_inference_state_access': False})
    finish_run(out, binding)


def atomic_check(config):
    cfg = {**dict(seed=0, phase='dev', device='cuda', dtype='bfloat16', prefix_batch=16), **config}
    manifest, binding, done = start_run(cfg, 'atomic_check')
    if done:
        return
    if cfg['phase'] != 'dev' or manifest.get('domain') != DOMAIN:
        raise ValueError('Atomic preparation check is dev-only in the list DSL')
    rows = read_jsonl(Path(cfg['data']) / 'dev_atomic.jsonl')
    if not rows or any(r['p'] not in manifest['atomic_warmup_fields'] or r['depth'] != 1 for r in rows):
        raise ValueError('Atomic admission must not expose withheld fields or compositions')
    seed_all(cfg['seed'])
    model, tokenizer, ids = load(cfg['base'], adapter=cfg.get('adapter'), device=cfg['device'], dtype=cfg['dtype'])
    result = atomic_results(model, tokenizer, ids, rows, cfg['prefix_batch'])
    checks = []
    for op in core.OPS:
        planned = [r['correct'] for r in result['plan'] if r['operation'] == op]
        applied = [r['semantic_correct'] for r in result['apply'] if r['witness'][0] == op]
        checks.append({'operation': op, 'tasks': len(planned),
                       'plan_accuracy': float(np.mean(planned)) if planned else 0.,
                       'apply_accuracy': float(np.mean(applied)) if applied else 0.})
    passed = all(r['tasks'] >= 20 and min(r['plan_accuracy'], r['apply_accuracy']) >= .8 for r in checks)
    out = Path(cfg['output'])
    write_json(out / 'atomic_retention.json', {'phase': 'dev', **result})
    write_json(out / 'gate.json', {'kind': 'atomic', 'pass': passed, 'checks': checks,
        'status': 'passed' if passed else 'atomic_skills_insufficient',
        'criterion': '>=20 dev tasks per operation; PLAN and APPLY accuracy >=0.8 each, allowed train fields only'})
    finish_run(out, binding)


def state_access(config):
    cfg = {**dict(seed=0, phase='dev', device='cuda', dtype='bfloat16', panels_per_family=8), **config}
    if cfg['phase'] != 'dev':
        raise ValueError('State hypothesis screening is dev-only')
    _, binding, done = start_run(cfg, 'state_access')
    if done:
        return
    out, data = Path(cfg['output']), Path(cfg['data'])
    seed_all(cfg['seed'])
    model, tokenizer, ids = load(cfg['base'], adapter=cfg.get('adapter'), device=cfg['device'], dtype=cfg['dtype'])
    rows, panels, contrasts = [], [], defaultdict(list)
    with torch.no_grad():
        for family in ('B', 'D'):
            source = check_panels(read_jsonl(data / f'dev_panel_{family}.jsonl'))[:cfg['panels_per_family']]
            for panel in source:
                accuracies = defaultdict(list)
                for row in panel:
                    results = {mode: solve(model, tokenizer, ids, row, mode) for mode in ('plain', 'state', 'token_control')}
                    rows.append({'task_id': row['task_id'], 'panel_id': row['panel_id'], **results})
                    for mode, value in results.items():
                        accuracies[mode].append(value['correct'])
                values = {k: float(np.mean(v)) for k, v in accuracies.items()}
                panels.append({'panel_id': panel[0]['panel_id'], 'family': family, **values})
                for control in ('plain', 'token_control'):
                    contrasts[family, control].append(values['state'] - values[control])
    checks = [{'family': f, 'control': c, 'panels': len(v), 'gain': float(np.mean(v)), 'ci95': bootstrap(v)}
              for (f, c), v in contrasts.items()]
    # Require independent held-out panel evidence, not one lucky model completion.
    passed = len(checks) == 4 and all(r['panels'] >= 8 and r['ci95'][0] > 0 for r in checks)
    write_json(out / 'state_access.json', {'binding': binding, 'tasks': rows, 'panels': panels, 'contrasts': checks,
        'control': 'identical annotation token count at each identical prefix; trajectories may diverge',
        'interpretation': 'interpreter-assisted inference diagnosis, not an equal-budget final method'})
    write_json(out / 'gate.json', {'kind': 'state', 'pass': passed, 'checks': checks,
                                  'criterion': '>=8 dev panels per B/D and positive paired CI against both controls'})
    finish_run(out, binding)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--state-access', action='store_true')
    mode.add_argument('--atomic-check', action='store_true')
    args = parser.parse_args()
    (state_access if args.state_access else atomic_check if args.atomic_check else run)(json.loads(args.config.read_text()))


if __name__ == '__main__':
    main()
