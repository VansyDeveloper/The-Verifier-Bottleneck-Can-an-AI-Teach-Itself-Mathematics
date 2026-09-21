"""V3 complete-space metrics, entropy accounting, and paired state-access diagnosis."""

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
from .research_dsl import DOMAIN, apply_prompt, trajectory, verify
from .research_io import start_run, finish_run
from .research_model import load_head, score_task, solve
from .research_objectives import assignment_cycles, entropy_report, interaction
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
                        starts=[row['start'] for row in panel], task_ids=[row['task_id'] for row in panel])
            if crossed:
                desired = torch.tensor([1., -1., -1., 1.])
                margin = (matrix[:, 0] - matrix[:, 1]) * desired
                credit = (margin > 1e-8).double() + .5 * (margin.abs() <= 1e-8)
                item.update(interaction=float(interaction(matrix)), cell_accuracy=float(credit.mean()),
                            joint_accuracy=float(credit.prod()), all_cells_strict=bool((margin > 1e-8).all()))
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


def run(config):
    cfg = {**dict(phase='dev', seed=0, arm='baseline', device='cuda', dtype='bfloat16', prefix_batch=32,
                  solve_per_family=8), **config}
    manifest, binding, done = start_run(cfg, 'evaluate')
    if done:
        return
    if cfg['phase'] not in ('dev', 'final'):
        raise ValueError('Unknown evaluation phase')
    if cfg['phase'] == 'final':
        if not cfg.get('analysis_lock'):
            raise ValueError('Final requires a v3 immutable analysis lock')
        lock = json.loads(Path(cfg['analysis_lock']).read_text())
        if (lock['schema'] != 'iclr.research.lock.v3' or lock['code_hash'] != code_hash()
                or lock['plan_hash'] != file_hash(ROOT / 'plans/research_v3.json')
                or binding['data_hash'] not in lock['dataset_hashes']
                or [binding['base_hash'], binding['adapter_hash']] not in lock['checkpoints']):
            raise ValueError('Final lock differs from data, code, or selected checkpoints')
    data, out = Path(cfg['data']), Path(cfg['output'])
    seed_all(0 if cfg['seed'] == -1 and cfg['arm'] in ('baseline', 'initial') else cfg['seed'])
    model, tokenizer, ids = load(cfg['base'], adapter=cfg.get('adapter'), device=cfg['device'], dtype=cfg['dtype'])
    head = load_head(cfg.get('adapter'), next(model.parameters()).device)
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
        atomic_plan = []
        for row in atomic:
            atomic_plan.append({'task_id': row['task_id'], 'operation': row['witness'][0],
                                **solve(model, tokenizer, ids, row, head=head)})
        applied = execute_programs(model, tokenizer, [{**r, 'kind': 'atomic', 'program': r['witness']} for r in atomic], min(8, cfg['prefix_batch']), prompt_builder=apply_prompt)
    write_json(out / 'solutions.json', solutions)
    write_json(out / 'atomic_retention.json', {'phase': cfg['phase'], 'plan': atomic_plan, 'apply': applied})
    write_json(out / 'budget.json', {**budget, 'wall_seconds': time.monotonic() - started,
        'scored_tasks': len(rows), 'full_candidate_spaces': sorted({5 ** r['depth'] for r in rows}),
        'solution_forward_tokens': sum(s[k]['forward_tokens'] for s in solutions for k in ('greedy', 'free')),
        'atomic_tasks': len(atomic), 'ordinary_inference_state_access': False})
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
    parser.add_argument('--state-access', action='store_true')
    args = parser.parse_args()
    (state_access if args.state_access else run)(json.loads(args.config.read_text()))


if __name__ == '__main__':
    main()
