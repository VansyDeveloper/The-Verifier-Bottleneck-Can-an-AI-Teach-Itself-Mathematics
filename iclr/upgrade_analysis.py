"""CPU-only controls and prior-invariant alignment from saved complete rankings."""

import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path

import numpy as np

from .calibration import calibrated_rows, summarize, unpack, write_csv
from .common import file_hash, read_jsonl, verify_data, verify_receipt, write_json

SCORE_KEYS = ('score', 'local_score', 'format_score')


def measure(matrix, atol=1e-8):
    """Reviewer's cycle/assignment diagnostic, including fractional tie credit."""
    s = np.asarray(matrix, dtype=np.float64)
    if s.ndim != 2 or s.shape[0] != s.shape[1] or not 2 <= len(s) <= 7 or not np.isfinite(s).all():
        raise ValueError('Expected a finite square matrix of 2..7 targets')
    s = s - s.mean(1, keepdims=True) - s.mean(0, keepdims=True) + s.mean()
    values = np.array([sum(s[i, j] for i, j in enumerate(perm))
                       for perm in itertools.permutations(range(len(s)))])
    tied = np.isclose(values, values.max(), rtol=0, atol=atol)
    cycle = np.array([s[i, i] + s[j, j] - s[i, j] - s[j, i]
                      for i, j in itertools.combinations(range(len(s)), 2)])
    return {'assignment_credit': float(1 / tied.sum()) if tied[0] else 0.,
            'assignment_margin': float(values[0] - values[1:].max()),
            'pair_accuracy_tie_half': float(np.mean((cycle > atol) + .5 * (np.abs(cycle) <= atol))),
            'cycle_contrasts': cycle.tolist(), 'maximizers': int(tied.sum()), 'tie_tolerance': atol}


def alignment(rows, key):
    grouped, output, skipped = defaultdict(list), [], []
    for row in rows:
        if row.get('panel_id'):
            grouped[row['panel_id']].append(row)
    for pid, cell in sorted(grouped.items()):
        cell.sort(key=lambda r: r['task_id'])
        if (len(cell) != 4 or len({(r['p'], tuple(r['start']), r['depth'], r['split']) for r in cell}) != 1
                or len({tuple(r['target']) for r in cell}) != 4):
            raise ValueError(f'Invalid four-target same-START panel: {pid}')
        for row in cell:
            unpack(row, key)
        correct = [[tuple(x['program']) for x in r['ranking'] if x['correct']] for r in cell]
        if any(len(p) != 1 for p in correct):
            skipped.append({'panel_id': pid, 'evaluation_set': cell[0]['split'],
                            'reason': 'nonunique_solution', 'correct_counts': list(map(len, correct))})
            continue
        programs = [p[0] for p in correct]
        if len(set(programs)) != 4:
            raise ValueError('Different deterministic targets share a solution')
        table = [{tuple(x['program']): x[key] for x in r['ranking']} for r in cell]
        output.append({'panel_id': pid, 'evaluation_set': cell[0]['split'], 'family': cell[0]['family'],
                       'p': cell[0]['p'], 'start': cell[0]['start'], 'depth': cell[0]['depth'],
                       'score_key': key, 'task_ids': [r['task_id'] for r in cell],
                       **measure([[t[p] for p in programs] for t in table])})
    return {'panels': output, 'skipped': skipped, 'eligible': len(output), 'total': len(grouped),
            'static_prior_chance': {'pair_accuracy_tie_half': .5, 'assignment_credit': 1 / 24}}


def analyze(calibration, rows, training, out, binding, reliable_ops=('AC1', 'AX1', 'REV')):
    out = Path(out)
    metrics, panels = [], {}
    for key in SCORE_KEYS:
        computed, _ = calibrated_rows(calibration, rows, score_key=key, training=training)
        metrics.extend({**r, **{k: binding[k] for k in ('model', 'seed', 'arm')},
                        'phase': binding['phase']} for r in computed)
        panels[key] = alignment(rows, key)
    write_csv(out / 'task_metrics.csv', metrics)
    summaries = summarize(metrics)
    write_json(out / 'hitk.json', summaries)
    write_csv(out / 'summary.csv', [{k: v for k, v in r.items() if k != 'hitk'} for r in summaries])
    write_json(out / 'target_alignment.json', {'binding': binding, 'scores': panels,
        'primary': 'local_score pair_accuracy_tie_half; composition minus atomic_control; equal B/D',
        'invariance': 'all column program priors and row prompt offsets cancel; do not count corrections as replications'})
    own, donors = {}, defaultdict(list)
    for row in metrics:
        identity = row['task_id'], row['score_key'], row['method']
        if row['target_control'] == 'own_target' and row['panel_id']:
            own[identity] = row
        elif row['target_control'] == 'wrong_target':
            donors[identity].append(row)
    advantages = []
    for identity, cell in donors.items():
        if len(cell) != 3:
            raise ValueError('Each four-target panel requires all three wrong donors')
        row = own[identity]
        advantages.append({k: row[k] for k in ('task_id', 'panel_id', 'family', 'evaluation_set', 'score_key', 'method')})
        advantages[-1].update({m + '_own_minus_wrong': row[m] - float(np.mean([r[m] for r in cell]))
                              for m in ('correct_mass', 'mrr', 'hit@1', 'hit@8', 'hit@32')})
    write_csv(out / 'own_target_advantage.csv', advantages)
    eligible = {r['task_id'] for r in rows if all(set(x['program']) <= set(reliable_ops)
                for x in r['ranking'] if x['correct'])}
    sets = defaultdict(list)
    for row in rows:
        sets[row['split']].append(row)
    coverage = [{'evaluation_set': split, 'total': len(cell),
                 'eligible': sum(r['task_id'] in eligible for r in cell),
                 'coverage': sum(r['task_id'] in eligible for r in cell) / len(cell),
                 'status': 'nonempty' if any(r['task_id'] in eligible for r in cell) else 'empty'}
                for split, cell in sorted(sets.items())]
    write_json(out / 'reliable_ops_stratum.json', {'operations': reliable_ops, 'phase': binding['phase'],
        'interpretation': 'candidate stratum only; baseline atomic and true-intermediate competence required',
        'coverage': coverage, 'task_ids': sorted(eligible),
        'summary': summarize([r for r in metrics if r['task_id'] in eligible])})
    # A fixed equal mixture, never weighted by the number of tasks/donors in a family.
    mixtures = defaultdict(dict)
    for row in summaries:
        key = tuple(row[k] for k in ('model', 'seed', 'arm', 'score_key', 'method', 'target_control'))
        if row['evaluation_set'] == binding['phase'] + '_' + row['family']:
            mixtures[key][row['family']] = row
    write_json(out / 'equal_family_mixture.json', [{**dict(zip(
        ('model', 'seed', 'arm', 'score_key', 'method', 'target_control'), key)),
        'weights': {f: .25 for f in 'ABCD'}, **{m: float(np.mean([cell[f][m] for f in 'ABCD']))
            for m in ('mrr', 'correct_mass', 'hit@1', 'hit@8', 'hit@32')}}
        for key, cell in mixtures.items() if set(cell) == set('ABCD')])


def reanalyze(args):
    source, out, data = Path(args.source), Path(args.out), Path(args.data)
    receipt = verify_receipt(source)
    binding = {**receipt['binding']}
    verify_data(data)
    if file_hash(data / 'manifest.json') != binding['data_hash']:
        raise ValueError('Analysis training pool differs from the scored dataset')
    binding.setdefault('phase', 'final')  # v1 evaluator scored final only, atomic was dev (documented bug).
    if out.exists():
        raise FileExistsError('Choose a new analysis output; preserve the scored snapshot')
    out.mkdir(parents=True)
    training_path = source / 'control_training_pool.jsonl'
    if not training_path.exists():
        training_path = data / 'train.jsonl'
    if binding.get('training_pool_sha256', file_hash(training_path)) != file_hash(training_path):
        raise ValueError('Training pool changed')
    analyze(read_jsonl(source / 'calibration_rankings.jsonl'), read_jsonl(source / 'rankings.jsonl'),
            read_jsonl(training_path), out, binding)
    write_json(out / 'binding.json', binding)
    write_json(out / 'DONE', {'binding': binding, 'model_inference_repeated': False,
        'source_receipt_sha256': file_hash(source / 'DONE'),
        'files': {p.name: file_hash(p) for p in out.iterdir() if p.is_file()}})


def compare(args):
    """Paired START-cluster bootstrap, conditional on these checkpoints and masks."""
    runs = [json.loads(Path(p).read_text()) for p in args.alignment]
    hashes = {(r['binding']['data_hash'], r['binding']['base_hash'], r['binding']['phase']) for r in runs}
    if len(hashes) != 1:
        raise ValueError('Compare one mask/model/phase with a common atomic initialization at a time')
    paired = defaultdict(dict)
    for run in runs:
        b = run['binding']
        arm = b['arm']
        if arm not in (args.treatment, args.control):
            continue
        if arm in paired[b['seed']]:
            raise ValueError('Duplicate seed/arm')
        paired[b['seed']][arm] = {r['panel_id']: r for r in run['scores']['local_score']['panels']}
    differences, identities = [], None
    for seed, arms in sorted(paired.items()):
        if set(arms) != {args.treatment, args.control}:
            raise ValueError('Missing paired arm for a seed')
        treatment, control = arms[args.treatment], arms[args.control]
        if treatment.keys() != control.keys() or (identities is not None and identities != treatment.keys()):
            raise ValueError('Arms/seeds have different panels')
        identities = treatment.keys()
        delta = {}
        for pid in sorted(identities):
            a, b = treatment[pid], control[pid]
            if any(a[k] != b[k] for k in ('p', 'start', 'family', 'task_ids')):
                raise ValueError('Paired panel identity changed')
            if a['family'] in 'BD':
                delta[pid] = (a['family'], (a['p'], *a['start']),
                              a['pair_accuracy_tie_half'] - b['pair_accuracy_tie_half'])
        differences.append(delta)
    if not differences or not differences[0]:
        raise ValueError('No paired B/D panels')
    # Average continuation seeds before resampling START. Seeds are not fresh atomic initializations.
    clusters = {f: defaultdict(list) for f in 'BD'}
    for pid, (family, start, _) in differences[0].items():
        clusters[family][start].append(np.mean([d[pid][2] for d in differences]))
    values = [np.array([np.mean(v) for v in clusters[f].values()]) for f in 'BD']
    if any(not len(v) for v in values):
        raise ValueError('Both B and D need eligible START clusters')
    rng = np.random.default_rng(20260921)
    draws = np.mean([rng.choice(v, size=(2000, len(v)), replace=True).mean(1) for v in values], axis=0)
    write_json(args.out, {'primary_contrast': f'{args.treatment} - {args.control}',
        'metric': 'local target alignment, pair tie=.5, equal B/D',
        'estimate': float(np.mean([v.mean() for v in values])), 'ci95_start_bootstrap': np.quantile(draws, [.025, .975]).tolist(),
        'continuation_seeds': sorted(paired), 'start_clusters': {f: len(clusters[f]) for f in 'BD'},
        'per_seed': [{'seed': seed, 'estimate': float(np.mean([np.mean([r[2] for r in delta.values() if r[0] == f]) for f in 'BD']))}
                     for seed, delta in zip(sorted(paired), differences)],
        'scope': 'conditional on one mask and shared atomic checkpoint; not a mask-population CI',
        'four_mask_sign_flip_minimum_two_sided_p': .125,
        'sources': {str(p): file_hash(p) for p in args.alignment}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser('rescore')
    for name in ('source', 'data', 'out'):
        p.add_argument('--' + name, required=True)
    p.set_defaults(function=reanalyze)
    p = sub.add_parser('compare')
    p.add_argument('--alignment', nargs='+', required=True)
    p.add_argument('--treatment', default='composition')
    p.add_argument('--control', default='atomic_control')
    p.add_argument('--out', required=True)
    p.set_defaults(function=compare)
    args = parser.parse_args()
    args.function(args)


if __name__ == '__main__':
    main()
