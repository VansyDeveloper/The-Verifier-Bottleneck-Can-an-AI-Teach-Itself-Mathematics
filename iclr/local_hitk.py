"""Reanalyze the saved three-seed Qwen3-0.6B-Base series from this computer."""

import argparse
import json
import math
import re
from pathlib import Path
from statistics import fmean

import composition_core as core
from .analyze import interval, plot_curves, sign_p, write_csv
from .common import file_hash, read_jsonl, write_json


def load_local(manifest_path):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    if (manifest['schema'] != 'iclr.local-series-ranking.v1'
            or manifest['model'] != 'Qwen/Qwen3-0.6B-Base'
            or manifest['historical_main_six_seed_series'] is not False):
        raise ValueError('Expected the local second-series bundle')
    for name, binding in manifest['files'].items():
        if file_hash(path.parent / name) != binding['sha256']:
            raise ValueError(f'Changed source file: {name}')
    runs, data = {}, {}
    for entry in manifest['runs']:
        split, seed, arm = entry['split'], entry['training_seed'], entry['arm']
        key = split, seed, arm
        if key in runs or type(seed) is not int or arm not in ('atomic_control', 'composition'):
            raise ValueError('Duplicate or invalid seed/arm')
        training = entry['training']
        config = (path.parent / training['files']['config.resolved.yaml']).read_text()
        recorded_seed = re.search(r'^seed: (\d+)$', config, re.MULTILINE)
        if recorded_seed is None or int(recorded_seed[1]) != seed or training['seed'] != seed:
            raise ValueError('Training seed differs from its original resolved config')
        if entry['recorded_input_sha256'] and entry['recorded_input_sha256'] != file_hash(path.parent / entry['data']):
            raise ValueError('Dataset differs from its hash recorded at ranking time')
        if entry['recorded_adapter_sha256'] and entry['recorded_adapter_sha256'] != training['weights_sha256']:
            raise ValueError('Adapter differs from its hash recorded at ranking time')
        tasks = {task['task_id']: task for task in read_jsonl(path.parent / entry['data'])}
        if len(tasks) != 300 or (split in data and data[split] != tasks):
            raise ValueError('Different task sets across paired runs')
        data[split] = tasks
        raw = read_jsonl(path.parent / entry['files']['generations.jsonl'])
        rows = {row['task_id']: row for row in raw}
        if len(rows) != len(raw) or rows.keys() != tasks.keys():
            raise ValueError('Ranking task IDs are missing, duplicated or unpaired')
        for task_id, row in rows.items():
            task = tasks[task_id]
            rank = row['best_correct_rank']
            if (row['run_id'] != entry['run_id'] or task['max_steps'] != 3 or row['candidates_ranked'] != 125
                    or type(rank) is not int or not 1 <= rank <= 125):
                raise ValueError(f'Invalid exhaustive rank: {task_id}')
            if not math.isfinite(row['correct_mass']) or not 0 <= row['correct_mass'] <= 1:
                raise ValueError(f'Invalid correct mass: {task_id}')
            top1 = core.verify_program(task['start'], task['target'], row['top1_program'], task['p'])
            if top1 != row['top1_is_correct'] or top1 != (rank == 1):
                raise ValueError(f'Top-1 contradicts exact verification/rank: {task_id}')
        final = json.loads((path.parent / entry['files']['final_metrics.json']).read_text())
        for k in (1, 4, 8, 16, 32, 64):
            if not math.isclose(fmean(row['best_correct_rank'] <= k for row in raw), final[f'hit_at_{k}'], abs_tol=1e-12):
                raise ValueError(f'Raw ranks disagree with stored Hit@{k}')
        if final['tasks'] != len(raw) or final['candidates_per_task'] != 125:
            raise ValueError('Stored aggregate task/candidate count differs')
        runs[key] = rows
    expected = {(split, seed, arm) for split in ('capacity_d3_train', 'capacity_d3_heldout')
                for seed in range(3) for arm in ('atomic_control', 'composition')}
    if set(runs) != expected:
        raise ValueError('Expected both splits and both arms for all three training seeds')
    # Reuse exact polynomial semantics; saved files contain no full score arrays.
    for split, tasks in data.items():
        for task_id, task in tasks.items():
            count = sum(core.verify_program(task['start'], task['target'], program, task['p'])
                        for program in core.enumerate_programs(3))
            if not count or any(runs[split, seed, arm][task_id]['correct_programs'] != count
                                for seed in range(3) for arm in ('atomic_control', 'composition')):
                raise ValueError(f'Correct-program count fails independent enumeration: {task_id}')
    return manifest, runs


def analyze_local(manifest_path, out, plots=False):
    manifest, runs = load_local(manifest_path)
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Choose a new output directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    curves, by_seed, pairs = [], [], []
    for split in ('capacity_d3_train', 'capacity_d3_heldout'):
        for k in range(1, 126):
            control, composition, differences = [], [], []
            for seed in range(3):
                a, c = runs[split, seed, 'atomic_control'], runs[split, seed, 'composition']
                hits_a = sum(row['best_correct_rank'] <= k for row in a.values())
                hits_c = sum(row['best_correct_rank'] <= k for row in c.values())
                control.append(hits_a / len(a))
                composition.append(hits_c / len(c))
                differences.append(hits_c - hits_a)
                by_seed.append(dict(split=split, seed=seed, k=k, n=len(a), atomic_control=control[-1],
                                    composition=composition[-1], delta=(hits_c - hits_a) / len(a)))
            mean, low, high = interval([value / 300 for value in differences])
            curves.append(dict(split=split, depth=3, k=k, atomic_control=fmean(control),
                               composition=fmean(composition), delta=mean,
                               delta_ci95_low=low, delta_ci95_high=high,
                               sign_flip_p=sign_p(differences, 300) if k == 32 else None))
        for seed in range(3):
            a, c = runs[split, seed, 'atomic_control'], runs[split, seed, 'composition']
            for task_id in sorted(a):
                ra, rc = a[task_id]['best_correct_rank'], c[task_id]['best_correct_rank']
                pairs.append(dict(split=split, seed=seed, task_id=task_id, atomic_rank=ra, composition_rank=rc,
                                  hit32_delta=int(rc <= 32) - int(ra <= 32),
                                  correct_mass_delta=c[task_id]['correct_mass'] - a[task_id]['correct_mass']))
    write_csv(out / 'hitk_summary.csv', curves)
    write_csv(out / 'hitk_by_seed.csv', by_seed)
    write_csv(out / 'paired_task_deltas.csv', pairs)
    summary = {'study': manifest['study'], 'model': manifest['model'], 'scorer': manifest['scoring'],
               'source_manifest_sha256': file_hash(manifest_path), 'training_seeds': [0, 1, 2],
               'ranking_cli_seed_is_not_training_seed': True,
               'main_six_seed_series': False, 'full_program_scores_available': False,
               'historical_tokenizer_hash_and_revision': None,
               'source_snapshot_is_current_not_run_bound': True,
               'intervals': '95% pointwise t intervals over three paired training seeds',
               'sign_flip_test': 'Hit@32 only; three seeds imply minimum two-sided p=0.25',
               'validation': 'source hashes, paired IDs, saved Hit@K, exact solution counts and top1 labels',
               'hit32': [row for row in curves if row['k'] == 32]}
    write_json(out / 'summary.json', summary)
    if plots:
        plot_curves(curves, out)
    return curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] / 'evidence/local_series/manifest.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    analyze_local(args.manifest, args.out, args.plots)
    print(f'Local three-seed Hit@K: {args.out}')


if __name__ == '__main__':
    main()
