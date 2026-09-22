"""Post-hoc additive score diagnostic on raw, receipt-verified Q1 exports (CPU only)."""

import argparse
from collections import defaultdict
import gzip
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import iclr  # registers the frozen composition interpreter
import composition_core as core
import numpy as np
from iclr.common import file_hash, read_jsonl, verify_data, verify_receipt, write_json
from iclr.data import constraint_hits
from iclr.research_data import check_panels
from iclr.research_io import plan_path


def probabilities(scores):
    q = np.exp(np.asarray(scores, dtype=float) - np.max(scores))
    return q / q.sum()


def crossed(matrix):
    d = np.asarray(matrix)[:, 0] - np.asarray(matrix)[:, 1]
    margins = d * [1, -1, -1, 1]
    a, b, c, h = np.array([[1,1,1,1], [1,1,-1,-1], [1,-1,1,-1], [1,-1,-1,1]]) @ d / 4
    return dict(score_matrix=np.asarray(matrix).tolist(), signed_margins=margins.tolist(), minimum_margin=float(min(margins)),
        a=float(a), b=float(b), c=float(c), h=float(h), interaction=float(4*h), interaction_positive=bool(4*h > 1e-8),
        joint=float(np.all(margins > 1e-8)), cell=float(np.mean((margins > 1e-8) + .5*(np.abs(margins) <= 1e-8))),
        scalar_offset_gap=float(min(d[[0,3]]) - max(d[[1,2]])))


def pool_decomposition(pi_c, r_c, pi_t, r_t):
    between = (pi_t-pi_c)*(r_t+r_c)/2
    within = (r_t-r_c)*(pi_t+pi_c)/2
    assert abs(between + within - (pi_t*r_t - pi_c*r_c)) < 1e-12
    return between, within


def analyze(results, out):
    results, out = Path(results), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    compact = (results / 'input_manifest.json').exists()
    inputs = json.loads((results / 'input_manifest.json').read_text()) if compact else None
    data = results / 'data/original'
    manifest = inputs['parent_manifest'] if compact else verify_data(data)
    data_hash = inputs['parent_manifest_hash'] if compact else file_hash(data / 'manifest.json')
    compact_rows = []
    if compact:
        if file_hash(results / 'source_scores.jsonl.gz') != inputs['source_scores_sha256']:
            raise ValueError('Compact Q1 source scores changed')
        compact_rows = read_jsonl(results / 'source_scores.jsonl.gz')
    programs = list(core.enumerate_programs(3))
    pool = np.array([constraint_hits(p, manifest['constraints']) > 0 for p in programs])
    reference = inputs['reference'] if compact else sorted(read_jsonl(data / 'dev_A.jsonl'), key=lambda r: r['task_id'])[:64]
    if len(reference) != 64 or any(r['p'] not in core.KNOWN_FIELDS or r['family'] != 'A' for r in reference):
        raise ValueError('Need the fixed 64 known-field ordinary dev A reference inputs')
    ref_ids = [r['task_id'] for r in reference]
    summaries, nuisances, pool_rows, priors, sources = [], [], [], [], dict(inputs['sources']) if compact else {}
    exported = []
    for seed in (0, 1, 2):
        paired = []
        for arm in ('atomic_control', 'composition'):
            if compact:
                paired.append({r['row']['task_id']: r['row'] for r in compact_rows if r['seed'] == seed and r['arm'] == arm})
                continue
            directory = results / 'evaluations' / f'q06_{arm}_seed{seed}'
            receipt = verify_receipt(directory)
            if (receipt['binding']['data_hash'] != data_hash
                    or receipt['binding']['config']['phase'] != 'dev'):
                raise ValueError('Wrong Q1 source data or phase')
            sources[directory.name] = {'receipt_sha256': file_hash(directory / 'DONE'), 'binding': receipt['binding']}
            rows = [{k:v for k,v in r.items() if k != 'entropy'} for r in read_jsonl(directory / 'rankings.jsonl')
                    if r['depth'] == 3 and (r['task_id'] in ref_ids or r['family'] in 'BD')]
            paired.append({r['task_id']: r for r in rows})
            exported.extend({'seed': seed, 'arm': arm, 'row': r} for r in rows)
        control, treatment = paired
        if control.keys() != treatment.keys():
            raise ValueError('Unpaired source tasks')
        for key in control:
            if any(control[key][k] != treatment[key][k] for k in ('p', 'start', 'target', 'correct_programs', 'family')):
                raise ValueError('Task identity changed between arms')
        for policy in ('local', 'full'):
            delta = np.mean([np.array(treatment[k][policy])-control[k][policy] for k in ref_ids], axis=0)
            priors.append({'seed': seed, 'policy': policy, 'delta_b': delta.tolist()})
            variants = {
                'control': lambda k: np.array(control[k][policy]),
                'composition': lambda k: np.array(treatment[k][policy]),
                'control_plus_prior': lambda k: np.array(control[k][policy]) + delta,
                'composition_minus_prior': lambda k: np.array(treatment[k][policy]) - delta,
                'prior_only': lambda k: delta,
            }
            for variant, score in variants.items():
                values = defaultdict(list)
                for row in control.values():
                    if row.get('panel_id') or row['family'] not in 'BD':
                        continue
                    q = probabilities(score(row['task_id']))
                    correct = np.array([list(p) in row['correct_programs'] for p in programs])
                    s = score(row['task_id'])
                    order = sorted(range(len(programs)), key=lambda i: (-s[i], tuple(programs[i])))
                    rank = min(np.flatnonzero(correct[order])) + 1
                    mass = float(q[correct].sum())
                    for name, value in {'hit1': rank <= 1, 'hit8': rank <= 8, 'hit32': rank <= 32,
                                        'correct_mass': mass, 'iid_success32': -np.expm1(32*np.log1p(-mass))}.items():
                        values[row['family'], name].append(float(value))
                panels = check_panels([r for r in control.values() if r.get('panel_id') and r['family'] in 'BD'])
                for panel in panels:
                    is_crossed = panel[0].get('panel_kind') == 'crossed'
                    candidates = [panel[0]['program_A'], panel[0]['program_B']] if is_crossed else [r['witness'] for r in panel]
                    indices = [programs.index(tuple(p)) for p in candidates]
                    matrix = np.array([score(r['task_id'])[indices] for r in panel])
                    if is_crossed:
                        result = crossed(matrix)
                        for name in ('joint', 'cell', 'interaction', 'interaction_positive'):
                            values[panel[0]['family'], name].append(float(result[name]))
                        nuisances.append({'seed': seed, 'policy': policy, 'variant': variant,
                            'panel_id': panel[0]['panel_id'], 'family': panel[0]['family'], **result})
                    else:
                        cycles = np.array([matrix[i,i]+matrix[j,j]-matrix[i,j]-matrix[j,i] for i,j in itertools.combinations(range(4),2)])
                        values[panel[0]['family'], 'four_target_pair'].append(float(np.mean((cycles > 1e-8) + .5*(np.abs(cycles) <= 1e-8))))
                summaries += [{'seed': seed, 'policy': policy, 'variant': variant, 'family': f, 'metric': m,
                               'n': len(v), 'mean': float(np.mean(v))} for (f,m),v in values.items()]
            for key, row in control.items():
                if row.get('panel_id') or row['family'] not in 'BD':
                    continue
                correct = np.array([list(p) in row['correct_programs'] for p in programs])
                if np.any(correct & ~pool):
                    raise ValueError('Correct set is not contained in the diagnostic held-pair pool')
                qc, qt = (probabilities(rows[key][policy]) for rows in paired)
                pi_c, pi_t = float(qc[pool].sum()), float(qt[pool].sum())
                mc, mt = float(qc[correct].sum()), float(qt[correct].sum())
                between, within = pool_decomposition(pi_c, mc/pi_c, pi_t, mt/pi_t)
                pool_rows.append(dict(seed=seed, policy=policy, task_id=key, family=row['family'],
                    pi_control=pi_c, pi_composition=pi_t, r_control=mc/pi_c, r_composition=mt/pi_t,
                    delta_mass=mt-mc, between=between, within=within))
    grouped = defaultdict(list)
    for r in summaries:
        grouped[r['policy'], r['variant'], r['metric']].append(r['mean'])
    pool_means = {policy: {k: float(np.mean([r[k] for r in pool_rows if r['policy'] == policy]))
                         for k in ('delta_mass', 'between', 'within')} for policy in ('local', 'full')}
    transitions = {}
    for policy in ('local', 'full'):
        index = {(r['seed'], r['panel_id'], r['variant']): r for r in nuisances if r['policy'] == policy}
        pairs = [(r, index[s, p, 'composition']) for (s,p,v),r in index.items() if v == 'control']
        transitions[policy] = {'panel_seed_records': len(pairs), 'unique_panels': len({p for _,p,_ in index}),
            'I_nonpositive_to_positive': sum(not a['interaction_positive'] and b['interaction_positive'] for a,b in pairs),
            'I_positive_to_nonpositive': sum(a['interaction_positive'] and not b['interaction_positive'] for a,b in pairs)}
        for (s,p,v), r in index.items():
            origin = 'control' if v == 'control_plus_prior' else 'composition' if v == 'composition_minus_prior' else None
            if origin:
                assert abs(r['interaction']-index[s,p,origin]['interaction']) < 1e-10
    write_json(out / 'summary.json', {'status': 'POST_HOC_CPU_REPLAY', 'alpha': 1,
        'amendment_plan_hash': file_hash(plan_path('v5')), 'source_manifest_hash': data_hash,
        'reference_ids': ref_ids, 'reference_scope': 'fixed first 64 dev A ordinary tasks, known fields; chosen post-hoc; no B/D outcomes used',
        'metrics': [{'policy': p, 'variant': v, 'metric': m, 'mean_over_seeds_and_BD': float(np.mean(x))} for (p,v,m),x in grouped.items()],
        'per_seed_family': summaries, 'pool_decomposition': pool_means, 'interaction_transitions': transitions,
        'limitations': 'additive score model, not causal; no alpha tuning; no new training; same historical masks/checkpoints; provenance gaps remain', 'sources': sources})
    for name, rows in [('crossed_decomposition', nuisances), ('pool_decomposition', pool_rows)]:
        with gzip.open(out / (name + '.jsonl.gz'), 'wt', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
    write_json(out / 'program_prior.json', {'programs': programs, 'priors': priors})
    if not compact:
        with gzip.open(out / 'source_scores.jsonl.gz', 'wt', encoding='utf-8') as stream:
            for row in exported:
                stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
        write_json(out / 'input_manifest.json', {'parent_manifest': manifest, 'parent_manifest_hash': data_hash,
            'source_scores_sha256': file_hash(out / 'source_scores.jsonl.gz'), 'reference': reference, 'sources': sources,
            'scope': 'derived lossless subset of receipt-verified depth3 scores; reference A and evaluation B/D, both policies; original full archives retain original receipt/rankings verification'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True); parser.add_argument('--out', required=True)
    args = parser.parse_args()
    analyze(args.results, args.out)
