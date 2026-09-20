"""Inventory saved ICLR runs; verifies raw results without claiming weight replay."""

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean

from .calibration import family, score_metrics, unpack
from .common import file_hash, read_jsonl, verify_receipt, write_json
from .run import recipe_cells


def inventory(root, external_csv=None):
    root = Path(root)
    comparisons = list(root.glob('comparison_*.json'))
    if len(comparisons) != 1:
        raise ValueError('Expected one primary comparison manifest')
    entries = json.loads(comparisons[0].read_text())
    cells, runs, summaries = set(), [], {}
    fields = ('method', 'supervision', 'replay_fraction', 'budget_mode', 'depth3_only')
    task_identity = None
    for entry in entries:
        run = (root / entry['metrics']).parent.parent
        done = verify_receipt(run)
        trained = verify_receipt(run, 'TRAINED')
        config = done['binding']['config']
        if config != json.loads((run / 'config.resolved.json').read_text())['config']:
            raise ValueError('Run config and completion receipt differ')
        if config['seed'] != entry['seed'] or trained['payload_hash'] != done['payload_hash']:
            raise ValueError('Seed or payload receipt mismatch')
        for directory in (run / 'mid_eval', run / 'eval'):
            if directory.name == 'mid_eval':
                verify_receipt(directory)
            summary = json.loads((directory / 'summary.json').read_text())
            raw = read_jsonl(directory / 'rankings.jsonl')
            saved = read_jsonl(directory / 'metrics.jsonl')
            identity = [(r['task_id'], r['p'], r['start'], r['target'], r['split']) for r in raw]
            if len({r['task_id'] for r in raw}) != len(raw) or [r['task_id'] for r in saved] != [r['task_id'] for r in raw]:
                raise ValueError('Duplicate or mismatched task coverage')
            if task_identity is not None and identity != task_identity:
                raise ValueError('Different tasks between arms, seeds or checkpoints')
            task_identity = identity
            by_family = {}
            for row, metric in zip(raw, saved):
                scores, mask = unpack(row)
                derived = score_metrics(scores, mask)
                for key in ('best_rank', 'hit@32', 'mrr', 'correct_mass'):
                    if abs(derived[key] - metric[key]) > 1e-9:
                        raise ValueError(f'Raw/metric disagreement: {run.name} {key}')
                by_family.setdefault(family(row), []).append(derived['hit@32'])
            if set(by_family) != set('ABCD'):
                raise ValueError('Missing A/B/C/D family')
            for f, values in by_family.items():
                if abs(fmean(values) - summary['families'][f]['hit@32']) > 1e-12:
                    raise ValueError('Raw/summary disagreement')
            if directory.name == 'eval':
                summaries[entry['seed'], entry['arm']] = summary
        cells.add(tuple(config[k] for k in fields))
        runs.append({'seed': entry['seed'], 'arm': entry['arm'], 'run': run.relative_to(root).as_posix(),
                     'model': config['model'], 'historical_config': config,
                     'base_hash': done['binding']['base_hash'], 'data_hash': done['binding']['data_hash'],
                     'code_hash': done['binding']['code_hash'], 'adapter_hash': done['payload_hash'],
                     'done_sha256': file_hash(run / 'DONE'), 'hashed_files_checked': len(done['files']),
                     'weights_present': any((run / 'adapter').glob('*.safetensors')),
                     'tasks_per_family': {f: len(v) for f, v in by_family.items()},
                     'raw_hit32': {f: fmean(v) for f, v in by_family.items()}})
    base = next(root.glob('baseline_*'))
    verify_receipt(base)
    base_summary = json.loads((base / 'summary.json').read_text())
    csv_path = next(root.glob('results_*.csv'))
    differences = []
    for path in (csv_path, *([Path(external_csv)] if external_csv else [])):
        rows = list(csv.DictReader(path.open()))
        if len(rows) != len(entries):
            raise ValueError(f'CSV row count differs: {path}')
        for row in rows:
            arm = 'atomic_control' if float(row['replay_fraction']) == 1 else 'composition'
            for f in 'ABCD':
                expected = summaries[int(row['seed']), arm]['families'][f]['hit@32']
                actual = float(row[f'{f}_hit@32'])
                if abs(actual - expected) > 1e-12:
                    if path == csv_path:
                        raise ValueError('Internal CSV differs from raw rankings')
                    differences.append({'seed': int(row['seed']), 'arm': arm, 'family': f,
                                        'external_csv': actual, 'raw_package': expected})
    planned = recipe_cells(['replay', 'trace', 'set'])
    return {'status': 'raw_results_and_receipts_verified', 'source': str(root.resolve()),
            'comparison_sha256': file_hash(comparisons[0]), 'model_inference_repeated': False,
            'weights_verified': False, 'runs': runs, 'actual_training_runs': len(runs),
            'unobserved_old_cells': [c for c in planned if tuple(c[k] for k in fields) not in cells],
            'baseline_atomic_plan': base_summary['atomic_plan'],
            'baseline_atomic_parse': base_summary['atomic_parse'],
            'internal_csv_sha256': file_hash(csv_path),
            'external_csv_sha256': file_hash(external_csv) if external_csv else None,
            'external_csv_differences': differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--external-csv', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.root, args.external_csv)
    write_json(args.out, result)
    print(json.dumps({'out': str(args.out), 'runs': result['actual_training_runs'],
                      'external_csv_differences': len(result['external_csv_differences'])}))


if __name__ == '__main__':
    main()
