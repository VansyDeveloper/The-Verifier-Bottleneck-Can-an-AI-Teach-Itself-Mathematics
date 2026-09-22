"""Dev-only retention/progress admission for one shared continuation recipe."""

import argparse
import json
from pathlib import Path

from .common import code_hash, file_hash, verify_receipt, write_json
from .research_evaluate import atomic_summary, run as evaluate
from .research_io import plan_path
from .run import write_config


def evaluate_step(training, step, out):
    path = Path(training) / 'checkpoints' / f'step_{step:06d}'
    receipt = verify_receipt(path)
    if receipt['checkpoint_step'] != step:
        raise ValueError('Checkpoint step differs from its receipt')
    cfg = receipt['binding']['config']
    source = {k: cfg[k] for k in ('base', 'model', 'data', 'device', 'dtype', 'prefix_batch', 'plan',
              'amendment', 'base_training_receipt', 'initial_training_receipt') if k in cfg}
    evaluate({**source, 'adapter': str(path / 'adapter'), 'training_receipt': str(path),
              'arm': cfg['objective'], 'seed': cfg['seed'], 'phase': 'dev', 'mode': 'full', 'output': out})


def admit(queue_path, condition, step, out, full_evaluation=None):
    queue_path = Path(queue_path)
    queue = json.loads(queue_path.read_text())
    plan_hash = file_hash(plan_path('v5'))
    if (queue['schema'] != 'iclr.research.queue.v5' or queue['phase'] != 'dev' or queue['experiments'] != ['stability']
            or queue['source_code_hash'] != code_hash() or queue['research_plan_sha256'] != plan_hash):
        raise ValueError('Select from a v5 stability dev queue produced by this code/plan')
    spec = json.loads(plan_path('v5').read_text())['stability']
    if condition not in spec['conditions'] or step not in spec['checkpoint_steps'][1:]:
        raise ValueError('Choose one prespecified nonzero checkpoint and R0/R1/R2/R3')
    job = next(j for j in queue['jobs'] if j['id'].endswith(f'_{condition}_seed0_train'))
    train = Path(job['result'])
    trained = verify_receipt(train)
    cfg, bound = trained['binding']['config'], trained['binding']
    initial_job = next(j for j in queue['jobs'] if j['id'].endswith('_initial'))
    initial_path = Path(initial_job['result'])
    budget = json.loads((train / 'budget.json').read_text())
    candidate_path = Path(full_evaluation) if full_evaluation else Path(next(j['result'] for j in queue['jobs']
        if j['id'].endswith(f'_{condition}_seed0_eval')))
    initial, candidate = verify_receipt(initial_path), verify_receipt(candidate_path)
    cb, ib = candidate['binding'], initial['binding']
    completed_step = cb.get('checkpoint_step') if cb.get('checkpoint_step') is not None else budget['optimizer_steps']
    expected_receipt = train if cb.get('checkpoint_step') is None else train / 'checkpoints' / f'step_{step:06d}'
    if (completed_step != step or cb.get('training_receipt_hash') != file_hash(expected_receipt / 'DONE')
            or any(cb[k] != ib[k] or cb[k] != bound[k] for k in ('base_hash', 'data_hash', 'code_hash', 'plan_hash', 'amendment_hash'))
            or any(b['config'].get('mode', 'full') != 'full' or b['config']['phase'] != 'dev' for b in (cb, ib))):
        raise ValueError('Selection requires full dev evaluation of this exact step and shared initial/data/plan')
    summaries = [atomic_summary(json.loads((p / 'atomic_retention.json').read_text())) for p in (initial_path, candidate_path)]
    checks = []
    for op, modes in summaries[0].items():
        for mode, baseline in modes.items():
            after = summaries[1][op][mode]
            required = mode == 'plan' or baseline['accuracy'] is not None and baseline['accuracy'] >= spec['initially_strong_apply']
            checks.append({'operation': op, 'mode': mode, 'required': required, 'initial': baseline, 'selected': after,
                'pass': baseline['n'] >= 20 and baseline['n'] == after['n'] and
                    (not required or after['accuracy'] >= baseline['accuracy'] - spec['retention_max_drop'] - 1e-12)})
    monitor_paths = [train / 'monitors' / f'step_{s:06d}' for s in (0, step)]
    monitors = []
    for p in monitor_paths:
        receipt = verify_receipt(p)
        if receipt['binding']['training_receipt_hash'] != file_hash(train / 'checkpoints' / p.name / 'DONE'):
            raise ValueError('Monitor does not bind its checkpoint')
        monitors.append(json.loads((p / 'monitor.json').read_text()))
    def train_mass(monitor):
        return next(r['correct_mass'] for r in monitor['ordinary'] if r['family'] == 'TRAIN')
    def familiar_mass(path):
        rows = json.loads((path / 'summary.json').read_text())['metrics']
        return next(r['correct_mass'] for r in rows if r['family'] == 'A' and r['depth'] == 3
                    and r['kind'] == 'ordinary' and r['policy'] == 'local' and r['temperature'] == 1)
    progress = {'train_correct_mass_gain': train_mass(monitors[1]) - train_mass(monitors[0]),
                'dev_A_correct_mass_gain': familiar_mass(candidate_path) - familiar_mass(initial_path)}
    progress_ok = max(progress.values()) >= spec['progress_min_gain']
    passed = not queue['smoke'] and all(c['pass'] for c in checks) and progress_ok
    sources = [train, initial_path, candidate_path, *monitor_paths]
    write_config(Path(out), {'schema': 'iclr.research.stability_selection.v5', 'pass': passed,
        'condition': condition, 'step': step, 'checks': checks, 'progress': progress, 'progress_pass': progress_ok,
        'code_hash': code_hash(), 'plan_hash': plan_hash, 'source_queue': str(queue_path.resolve()),
        'source_queue_sha256': file_hash(queue_path), 'source_receipts': {str(p.resolve()): file_hash(p / 'DONE') for p in sources},
        'base_hash': bound['base_hash'], 'dtype': cfg['dtype'], 'amendment_hash': bound['amendment_hash'],
        'settings': {**spec['conditions'][condition], 'max_steps': step, 'epochs': 2, 'panel_batch': 1},
        'scope': 'engineering dev admission, not a scientific success test; no B/D metric used for selection'})
    return passed


def verify_selection(path, plan_hash):
    if not path:
        raise ValueError('Matched queue requires --stability-selection after full-dev retention/progress admission')
    selected = json.loads(Path(path).read_text())
    if (selected.get('schema') != 'iclr.research.stability_selection.v5' or selected.get('pass') is not True
            or selected['code_hash'] != code_hash() or selected['plan_hash'] != plan_hash):
        raise ValueError('Stability admission missing, failed, or from another code/plan')
    if file_hash(selected['source_queue']) != selected['source_queue_sha256']:
        raise ValueError('Stability source queue changed')
    for directory, digest in selected['source_receipts'].items():
        verify_receipt(directory)
        if file_hash(Path(directory) / 'DONE') != digest:
            raise ValueError('Stability evidence changed')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('evaluate-step')
    p.add_argument('--training', required=True); p.add_argument('--step', type=int, required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('select')
    p.add_argument('--queue', required=True); p.add_argument('--condition', choices=['R0', 'R1', 'R2', 'R3'], required=True)
    p.add_argument('--step', type=int, required=True); p.add_argument('--full-evaluation'); p.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.command == 'evaluate-step':
        evaluate_step(args.training, args.step, args.out)
    else:
        passed = admit(args.queue, args.condition, args.step, args.out, args.full_evaluation)
        print('PASS: matched CE/CF admitted' if passed else 'NOT ADMITTED: inspect retention/progress; do not scale CF')


if __name__ == '__main__':
    main()
