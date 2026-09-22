"""Paired v4 endpoints: crossed interaction, ordinary solving, and secondary alignment."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .common import ROOT, file_hash, read_jsonl, verify_data, verify_receipt, write_json

ENDPOINTS = ('crossed_joint_accuracy_strict', 'crossed_cell_accuracy', 'crossed_interaction',
             'ordinary_hit1', 'ordinary_hit8', 'ordinary_correct_mass', 'four_target_pair_accuracy',
             'ordinary_multiple_solution_correct_entropy')


def observations(path, panels):
    values = defaultdict(list)
    for row in panels:
        if row['policy'] != 'local' or row['family'] not in ('B', 'D') or row['depth'] != 3:
            continue
        metrics = ({'crossed_joint_accuracy_strict': float(row['all_cells_strict']),
                    'crossed_cell_accuracy': row['cell_accuracy'], 'crossed_interaction': row['interaction']}
                   if row['kind'] == 'crossed' else {'four_target_pair_accuracy': row['pair_accuracy']})
        identity = json.dumps([row['task_ids'], row['starts'], row['family']], sort_keys=True)
        for name, value in metrics.items():
            values[name, row['family'], row['panel_id']].append((identity, float(value)))
    for row in read_jsonl(path / 'rankings.jsonl'):
        if row.get('panel_id') or row['family'] not in ('B', 'D') or row['depth'] != 3:
            continue
        metrics = next(m for m in row['entropy']['local'] if m['temperature'] == 1)
        identity = json.dumps([row['task_id'], row['p'], row['start'], row['target'], row['correct_programs']], sort_keys=True)
        unit = json.dumps([row['p'], row['start']])
        for name in ('hit1', 'hit8', 'correct_mass'):
            values['ordinary_' + name, row['family'], unit].append((identity, metrics[name]))
        if row['correct_count'] > 1:
            values['ordinary_multiple_solution_correct_entropy', row['family'], unit].append((identity, metrics['correct_entropy']))
    return {key: {'identities': tuple(i for i, _ in sorted(rows)), 'value': float(np.mean([v for _, v in rows]))}
            for key, rows in values.items()}


def compare(paths, control, treatment, out):
    if control == treatment:
        raise ValueError('Choose distinct conditions')
    plan_hash = file_hash(ROOT / 'plans/research_v4.json')
    groups, common, names, masks, initializations, exposures = defaultdict(dict), set(), {}, {}, defaultdict(set), {}
    dataset_statuses = {}
    for path in map(Path, paths):
        receipt = verify_receipt(path)
        result = json.loads((path / 'panel_metrics.json').read_text())
        binding, cfg = result['binding'], result['binding']['config']
        if binding != receipt['binding'] or cfg['arm'] not in (control, treatment) or binding['plan_hash'] != plan_hash:
            raise ValueError('Wrong comparison arm, result binding or v4 analysis plan')
        dataset = path.parent.parent / 'data' / Path(cfg['data']).name
        if not (dataset / 'manifest.json').is_file():
            dataset = Path(cfg['data'])
        manifest = verify_data(dataset)
        digest = binding['data_hash']
        if file_hash(dataset / 'manifest.json') != digest:
            raise ValueError('Comparison dataset differs from the evaluated dataset')
        masks[digest] = json.dumps(manifest['constraints'], sort_keys=True)
        dataset_statuses[digest] = manifest.get('status')
        common.add((cfg['phase'], binding['code_hash'], binding['plan_hash'], binding['base_hash'],
                    cfg['dtype'], manifest.get('domain', 'affine_polynomial_v1')))
        initializations[digest].add(binding.get('initial_adapter_hash'))
        exposures[str(path)] = binding.get('exposure')
        names[digest] = Path(cfg['data']).name
        key = cfg['seed'], cfg['arm']
        if key in groups[digest]:
            raise ValueError('Duplicate dataset/seed/arm')
        groups[digest][key] = observations(path, result['panels'])
    if len(common) != 1 or not groups:
        raise ValueError('Comparison mixes domains, phases, code, plans, model bases or precision')
    if any(len(v) != 1 for v in initializations.values()):
        raise ValueError('Paired conditions must share the same initialization within each mask')
    if len(set(masks.values())) != len(masks):
        raise ValueError('Different datasets use the same mask; do not count them as independent masks')
    rng, reports, samples, seeds_by_mask = np.random.default_rng(20260922), defaultdict(list), defaultdict(list), {}
    for dataset, runs in sorted(groups.items()):
        seeds = sorted({seed for seed, _ in runs})
        seeds_by_mask[dataset] = seeds
        if set(runs) != {(s, a) for s in seeds for a in (control, treatment)}:
            raise ValueError('Unpaired continuation seeds')
        keys = set(next(iter(runs.values())))
        if not keys or any(set(values) != keys for values in runs.values()):
            raise ValueError('Missing or unpaired panels/tasks/endpoints')
        for endpoint in ENDPOINTS:
            by_family, per_seed = defaultdict(list), defaultdict(list)
            for key in sorted(k for k in keys if k[0] == endpoint):
                rows = [runs[s, a][key] for s in seeds for a in (control, treatment)]
                if any(r['identities'] != rows[0]['identities'] for r in rows):
                    raise ValueError('Panel/task identity differs')
                delta = [runs[s, treatment][key]['value'] - runs[s, control][key]['value'] for s in seeds]
                by_family[key[1]].append(float(np.mean(delta)))
                for seed, value in zip(seeds, delta):
                    per_seed[seed, key[1]].append(value)
            if set(by_family) != {'B', 'D'}:
                continue
            draws = []
            for family in ('B', 'D'):
                values = np.asarray(by_family[family])
                draws.append(values[rng.integers(len(values), size=(2000, len(values)))].mean(1))
            bootstrap = np.mean(draws, axis=0)
            samples[endpoint].append(bootstrap)
            reports[endpoint].append({'data_hash': dataset, 'dataset': names[dataset], 'seeds': seeds,
                'estimate': float(np.mean([np.mean(v) for v in by_family.values()])),
                'ci95': list(map(float, np.quantile(bootstrap, [.025, .975]))),
                'families': {f: {'units': len(v), 'estimate': float(np.mean(v))} for f, v in by_family.items()},
                'per_seed': {str(s): float(np.mean([np.mean(per_seed[s, f]) for f in ('B', 'D')])) for s in seeds}})
    endpoints = {}
    for name in ENDPOINTS:
        complete = len(reports[name]) == len(groups)
        endpoints[name] = {'available_in_all_masks': complete, 'masks': reports[name],
            'estimate': float(np.mean([r['estimate'] for r in reports[name]])) if complete else None,
            'ci95': list(map(float, np.quantile(np.mean(samples[name], axis=0), [.025, .975]))) if complete else None,
            'unit': 'whole panel' if name.startswith(('crossed', 'four_target')) else 'shared (field, START) task cluster'}
    def positive(name, confidence=False):
        metric = endpoints[name]
        return metric['available_in_all_masks'] and (metric['ci95'][0] > 0 if confidence else metric['estimate'] >= 0)
    crossed_test = positive('crossed_joint_accuracy_strict', True)
    conditional = crossed_test and positive('crossed_cell_accuracy')
    solving = conditional and positive('ordinary_hit1', True) and positive('ordinary_hit8') and positive('ordinary_correct_mass')
    phase, _, _, base, _, domain = next(iter(common))
    confirmation = (phase == 'final' and len(groups) == 2 and set(names.values()) == {'mask1', 'mask2'}
        and all(v in ('frozen', 'frozen_new_evaluation') for v in dataset_statuses.values())
        and all(v == [0, 1, 2] for v in seeds_by_mask.values()))
    write_json(out, {'schema': 'iclr.research.comparison.v4', 'control': control, 'treatment': treatment,
        'plan_hash': plan_hash, 'endpoints': endpoints, 'distinct_masks': len(groups),
        'dataset_statuses': dataset_statuses,
        'initializations': {d: {'base_hash': base, 'adapter_hash': next(iter(v))} for d, v in initializations.items()},
        'success_rule': {'conditional_pattern': conditional, 'ordinary_solving_pattern': solving,
            'ordinary_hit1_confirmatory_test_open': crossed_test, 'confirmation_scope_complete': confirmation,
            'supports_joint_conditioning_claim': conditional and confirmation,
            'supports_task_solving_claim': solving and confirmation,
            'scope': 'fixed sequence: crossed joint then ordinary Hit@1; other intervals descriptive; dev is screening',
            'domain': domain, 'missing_crossed': 'no START x TARGET claim; external DSL reports domain adaptation only'},
        'interval_scope': 'paired panels/task clusters conditional on these checkpoints and masks; seeds averaged first, equal B/D and mask weights',
        'replication_scope': 'continuation seeds are not independent full training replications',
        'exposure_ledgers': exposures, 'sources': {str(p): file_hash(Path(p) / 'DONE') for p in paths}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--control', default='ce')
    parser.add_argument('--treatment', default='ce_cf')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    compare(args.runs, args.control, args.treatment, args.out)


if __name__ == '__main__':
    main()
