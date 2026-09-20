"""A-only score calibration. No checkpoint loading and no fitted alpha."""

import argparse
from collections import defaultdict
import csv
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

import composition_core as core
from .common import file_hash, read_jsonl, write_json

METHODS = ('raw', 'unigram', 'bigram', 'full')
KS = (1, 8, 16, 32, 64, 100, 125)


def features(programs, pairs=True):
    """Position-independent operation and adjacent-pair counts, with intercept."""
    ops = list(core.OPS)
    result = np.zeros((len(programs), 1 + 5 + (25 if pairs else 0)))
    result[:, 0] = 1
    for i, program in enumerate(programs):
        for op in program:
            result[i, 1 + ops.index(op)] += 1
        if pairs:
            for a, b in zip(program, program[1:]):
                result[i, 6 + 5 * ops.index(a) + ops.index(b)] += 1
    return result


@lru_cache(maxsize=8192)
def correct_mask(p, start, target, depth):
    return np.array([core.verify_program(start, target, program, p)
                     for program in sorted(core.enumerate_programs(depth))], dtype=bool)


def unpack(row, score_key='score'):
    programs = sorted(core.enumerate_programs(row['depth']))
    candidates = row['ranking']
    by_program = {tuple(item['program']): item for item in candidates}
    if len(candidates) != len(programs) or set(by_program) != set(programs):
        raise ValueError(f"Incomplete or duplicate candidates: {row['task_id']}")
    scores = np.array([by_program[p][score_key] for p in programs], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError(f"Nonfinite score: {row['task_id']}")
    mask = correct_mask(row['p'], tuple(row['start']), tuple(row['target']), row['depth'])
    if not mask.any() or not np.array_equal(mask, [by_program[p]['correct'] for p in programs]):
        raise ValueError(f"Incorrect solution labels: {row['task_id']}")
    fingerprint = core.canonical_task_fingerprint(row['p'], row['start'], row['target'], row['depth'])
    if row.get('task_fingerprint', fingerprint) != fingerprint:
        raise ValueError(f"Task fingerprint mismatch: {row['task_id']}")
    return scores, mask


def family(row):
    return row.get('family') or row['split'].rsplit('_', 1)[-1]


def fit_bias(rows, score_key='score'):
    if not rows or any(family(row) != 'A' or row['depth'] != 3 for row in rows):
        raise ValueError('Calibration requires only A tasks at depth 3')
    if len({r['task_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate calibration task IDs')
    programs = sorted(core.enumerate_programs(3))
    # Correct labels and target families never enter this estimate.
    means = np.mean([unpack(row, score_key)[0] for row in rows], axis=0)
    result = {'task_ids': sorted(r['task_id'] for r in rows), 'score_key': score_key,
              'alpha': 1, 'training_depth': 3, 'full': means.tolist()}
    for name, pairs in (('unigram', False), ('bigram', True)):
        design = features(programs, pairs)
        weights, _, rank, singular = np.linalg.lstsq(design, means, rcond=None)
        result[name] = {'weights': weights.tolist(), 'rank': int(rank),
                        'singular_values': singular.tolist(),
                        'rmse': float(np.sqrt(np.mean((design @ weights - means) ** 2)))}
    return result


def bias_vector(fit, method, depth):
    programs = sorted(core.enumerate_programs(depth))
    if method == 'raw':
        return np.zeros(len(programs))
    if method == 'full':
        if depth != fit['training_depth']:
            raise ValueError('A per-program table cannot transfer to another depth')
        return np.asarray(fit['full'])
    if method not in ('unigram', 'bigram'):
        raise ValueError(f'Unknown calibration: {method}')
    return features(programs, method == 'bigram') @ np.asarray(fit[method]['weights'])


def score_metrics(scores, mask):
    # Canonical lexical tie break matches the frozen evaluator.
    order = np.argsort(-scores, kind='stable')
    rank = int(np.flatnonzero(mask[order])[0]) + 1
    shifted = scores - np.max(scores)
    q = np.exp(shifted) / np.exp(shifted).sum()
    mass = float(np.clip(q[mask].sum(), 0, 1))
    cq = q[mask] / mass if mass else np.zeros(mask.sum())
    entropy = float(-np.sum(cq[cq > 0] * np.log(cq[cq > 0])))
    return {'best_rank': rank, 'correct_count': int(mask.sum()), 'correct_mass': mass,
            'mrr': 1 / rank, 'correct_conditional_entropy': entropy,
            **{f'hit@{k}': int(rank <= k) for k in KS},
            **{f'iid_pass@{k}': float(-np.expm1(k * np.log1p(-mass))) if mass < 1 else 1.
               for k in KS}}


def calibrated_rows(calibration, evaluation, *, crossfit=False, score_key='score'):
    """Closed evaluation uses disjoint A; archived A uses fixed five-fold crossfit."""
    fit = fit_bias(calibration, score_key)
    cal_ids = set(fit['task_ids'])
    if not crossfit and cal_ids & {r['task_id'] for r in evaluation}:
        raise ValueError('Calibration and evaluation task IDs overlap')
    fold_by_id = {tid: i % 5 for i, tid in enumerate(sorted(cal_ids))}
    folds = {}
    if crossfit:
        if len(calibration) < 5:
            raise ValueError('Five-fold calibration needs at least five A tasks')
        folds = {fold: fit_bias([r for r in calibration if fold_by_id[r['task_id']] != fold], score_key)
                 for fold in range(5)}
    output = []
    grouped = defaultdict(list)
    for row in evaluation:
        if row.get('panel_id'):
            grouped[row['panel_id']].append(row)
    wrong_target = {}
    for panel, rows in grouped.items():
        rows = sorted(rows, key=lambda r: r['task_id'])
        if len(rows) < 2 or len({(r['p'], tuple(r['start']), r['depth']) for r in rows}) != 1:
            raise ValueError(f'Invalid matched START panel: {panel}')
        if len({tuple(r['target']) for r in rows}) != len(rows):
            raise ValueError(f'Duplicate TARGET in panel: {panel}')
        wrong_target.update({r['task_id']: rows[(i + 1) % len(rows)] for i, r in enumerate(rows)})
    for row in evaluation:
        scores, mask = unpack(row, score_key)
        selected_fit = folds[fold_by_id[row['task_id']]] if row['task_id'] in cal_ids and crossfit else fit
        variants = {'own_target': scores}
        if row['task_id'] in wrong_target:
            variants['wrong_target'] = unpack(wrong_target[row['task_id']], score_key)[0]
        for target_control, values in variants.items():
            for method in METHODS:
                if method == 'full' and row['depth'] != 3:
                    continue
                corrected = values - bias_vector(selected_fit, method, row['depth'])
                output.append({'task_id': row['task_id'], 'family': family(row), 'depth': row['depth'],
                               'evaluation_set': row['split'],
                               'p': row['p'], 'degree': len(row['start']) - 1,
                               'panel_id': row.get('panel_id'), 'method': method, 'score_key': score_key,
                               'target_control': target_control, **score_metrics(corrected, mask)})
    return output, {'all_A': fit, 'crossfit': folds}


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    groups = defaultdict(list)
    keys = ('model', 'seed', 'arm', 'evaluation_set', 'family', 'depth', 'score_key', 'method', 'target_control')
    for row in rows:
        groups[tuple(row.get(k) for k in keys)].append(row)
    result = []
    for values, cell in groups.items():
        summary = {**dict(zip(keys, values)), 'tasks': len(cell)}
        for metric in ('mrr', 'correct_mass', *(f'hit@{k}' for k in KS), *(f'iid_pass@{k}' for k in KS)):
            summary[metric] = float(np.mean([r[metric] for r in cell]))
        # Full H(K), not just selected K. Each task contributes once.
        summary['hitk'] = [float(np.mean([r['best_rank'] <= k for r in cell]))
                           for k in range(1, 5 ** cell[0]['depth'] + 1)]
        result.append(summary)
    return result


def archive_analysis(comparison, out):
    comparison, out = Path(comparison), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite analysis: {out}')
    out.mkdir(parents=True, exist_ok=True)
    all_rows, fits, sources, identity = [], {}, [], None
    entries = json.loads(comparison.read_text())
    if not entries or len({(e['arm'], e['seed']) for e in entries}) != len(entries):
        raise ValueError('Empty or duplicate comparison entries')
    for entry in entries:
        metrics = comparison.parent / entry['metrics']
        ranking = metrics.with_name('rankings.jsonl')
        rows = read_jsonl(ranking)
        if len({r['task_id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate evaluation task IDs')
        current = {(r['task_id'], r['p'], tuple(r['start']), tuple(r['target']), r['depth'], r['split']) for r in rows}
        if identity is not None and identity != current:
            raise ValueError('Arms/seeds evaluate different tasks')
        identity = current
        config = json.loads((metrics.parent.parent / 'config.resolved.json').read_text())['config']
        result, fit = calibrated_rows([r for r in rows if family(r) == 'A'], rows, crossfit=True)
        all_rows.extend({**r, **{k: entry[k] for k in ('arm', 'seed')}, 'model': config['model']} for r in result)
        fits[f"{entry['arm']}_seed{entry['seed']}"] = fit
        sources.append({'path': str(ranking.resolve()), 'sha256': file_hash(ranking),
                        'base_hash': rows[0]['base_hash'], 'model_hash': rows[0]['model_hash'],
                        'code_hash': rows[0]['code_hash'], 'data_hash': rows[0]['data_hash']})
    summaries = summarize(all_rows)
    write_csv(out / 'calibration_by_seed.csv', [{k: v for k, v in r.items() if k != 'hitk'} for r in summaries])
    write_csv(out / 'task_metrics.csv', all_rows)
    write_json(out / 'hitk.json', summaries)
    write_json(out / 'coefficients.json', fits)
    write_json(out / 'receipt.json', {'status': 'development_reanalysis', 'model_inference_repeated': False,
        'calibration': 'A-only, alpha=1, five-fold crossfit for A, no B/C/D fitting',
        'probability_scope': 'global distribution over enumerated leaves after score intervention; not local action policy',
        'comparison_sha256': file_hash(comparison), 'sources': sources,
        'analysis_code_sha256': file_hash(Path(__file__)),
        'files': {p.name: file_hash(p) for p in sorted(out.iterdir()) if p.is_file()}})
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    summaries = archive_analysis(args.comparison, args.out)
    print(json.dumps({'out': str(args.out), 'cells': len(summaries), 'status': 'development_reanalysis'}))


if __name__ == '__main__':
    main()
