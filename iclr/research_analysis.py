"""Paired panel bootstrap; seeds and masks remain explicit units."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .common import file_hash, verify_data, verify_receipt, write_json


def compare(paths, control, treatment, out):
    if control == treatment:
        raise ValueError('Choose distinct conditions')
    groups, initialization, common, names, masks = defaultdict(dict), set(), set(), {}, {}
    for path in map(Path, paths):
        receipt = verify_receipt(path)
        result = json.loads((path / 'panel_metrics.json').read_text())
        binding, cfg = result['binding'], result['binding']['config']
        if binding != receipt['binding'] or cfg['arm'] not in (control, treatment):
            raise ValueError('Wrong comparison arm or result binding')
        dataset_path = path.parent.parent / 'data' / Path(cfg['data']).name
        if not (dataset_path / 'manifest.json').is_file():
            dataset_path = Path(cfg['data'])
        manifest = verify_data(dataset_path)
        if file_hash(dataset_path / 'manifest.json') != binding['data_hash']:
            raise ValueError('Comparison dataset differs from the evaluated dataset')
        masks[binding['data_hash']] = json.dumps(manifest['constraints'], sort_keys=True)
        common.add((cfg['phase'], binding['code_hash'], binding['plan_hash'], binding['base_hash'],
                    binding.get('initial_adapter_hash'), cfg['dtype'], manifest.get('domain', 'affine_polynomial_v1')))
        initialization.add(binding['base_hash'])
        names[binding['data_hash']] = Path(cfg['data']).name
        key = binding['data_hash'], cfg['seed'], cfg['arm']
        if key[1:] in groups[key[0]]:
            raise ValueError('Duplicate dataset/seed/arm')
        groups[key[0]][key[1:]] = {r['panel_id']: r for r in result['panels']
            if r['policy'] == 'local' and r['kind'] == 'four_target' and r['family'] in ('B', 'D') and r['depth'] == 3}
    if len(common) != 1 or not groups:
        raise ValueError('Comparison mixes domains, phases, code, plans, model bases or precision')
    if len(set(masks.values())) != len(masks):
        raise ValueError('Different datasets use the same mask; do not count them as independent masks')
    rng, reports = np.random.default_rng(20260922), []
    for dataset, runs in sorted(groups.items()):
        seeds = sorted({seed for seed, _ in runs})
        if set(runs) != {(s, a) for s in seeds for a in (control, treatment)}:
            raise ValueError('Unpaired continuation seeds')
        identities = [set(v) for v in runs.values()]
        if not identities[0] or any(ids != identities[0] for ids in identities):
            raise ValueError('Missing or unpaired panels')
        by_family, per_seed = defaultdict(list), defaultdict(list)
        for panel_id in sorted(identities[0]):
            rows = [runs[s, a][panel_id] for s in seeds for a in (control, treatment)]
            if any((r['task_ids'], r['starts'], r['family']) != (rows[0]['task_ids'], rows[0]['starts'], rows[0]['family']) for r in rows):
                raise ValueError('Panel identity or family differs')
            values = [runs[s, treatment][panel_id]['pair_accuracy'] - runs[s, control][panel_id]['pair_accuracy'] for s in seeds]
            by_family[rows[0]['family']].append(float(np.mean(values)))
            for seed, value in zip(seeds, values):
                per_seed[seed, rows[0]['family']].append(value)
        if set(by_family) != {'B', 'D'}:
            raise ValueError('Both primary families are required')
        draws = []
        for values in by_family.values():
            values = np.asarray(values)
            draws.append(values[rng.integers(len(values), size=(2000, len(values)))].mean(1))
        bootstrap = np.mean(draws, axis=0)
        reports.append({'data_hash': dataset, 'dataset': names[dataset], 'seeds': seeds,
            'estimate': float(np.mean([np.mean(v) for v in by_family.values()])),
            'ci95_panel_bootstrap': list(map(float, np.quantile(bootstrap, [.025, .975]))),
            'families': {f: {'panels': len(v), 'estimate': float(np.mean(v))} for f, v in by_family.items()},
            'per_seed': {str(s): float(np.mean([np.mean(per_seed[s, f]) for f in ('B', 'D')])) for s in seeds}})
    write_json(out, {'control': control, 'treatment': treatment, 'metric': 'local_pair_accuracy_tie_half',
        'masks': reports, 'equal_mask_mean': float(np.mean([r['estimate'] for r in reports])),
        'distinct_masks': len(reports), 'initializations': sorted(initialization),
        'interval_scope': 'paired panels conditional on these checkpoints and masks; seeds averaged before bootstrap',
        'replication_scope': 'continuation seeds sharing a base are not independent full training replications',
        'sources': {str(p): file_hash(Path(p) / 'DONE') for p in paths}})


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
