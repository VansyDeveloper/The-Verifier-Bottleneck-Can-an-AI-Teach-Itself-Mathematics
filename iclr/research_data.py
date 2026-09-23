"""Frozen v4 panels. Crossed panels are a diagnostic unless support passes preflight."""

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import random

import numpy as np

import composition_core as core
from .common import ROOT, file_hash, read_jsonl, verify_data, write_json
from .data import _full_states, constraint_hits, mixture
from .upgrade_data import affine_maps, generate_rows, reference_registry, signatures, reachable


def program_exposure(programs):
    pairs, depths = Counter(), Counter()
    for program in programs:
        depths[len(program)] += 1
        pairs.update('>'.join(pair) for pair in zip(program, program[1:]))
    return {'pairs': dict(sorted(pairs.items())), 'depths': dict(sorted(depths.items()))}


def historical_exposure(reference):
    """Reconstruct the recorded SFT mixtures; this is not an observed historical log."""
    digest = file_hash(reference / 'manifest.json')
    rows, atomic = read_jsonl(reference / 'train.jsonl'), read_jsonl(reference / 'atomic_train.jsonl')
    result = {}
    for path in sorted((ROOT / 'evidence/upgrade').glob('previous_q*.json')):
        for run in json.loads(path.read_text())['runs']:
            if run['data_hash'] != digest:
                continue
            cfg = run['historical_config']
            source = [r for r in rows if not cfg['depth3_only'] or r['depth'] == 3]
            examples = mixture(source, atomic, cfg['num_examples'], cfg['replay_fraction'], cfg['seed'],
                               'program_trace' if cfg['supervision'] == 'trace' else 'program_only')
            programs = [e['row']['witness'] for e in examples]
            counts = program_exposure(programs * cfg['epochs'])
            per_epoch = program_exposure(programs)
            if cfg['budget_mode'] == 'target_tokens':
                # Historical atomic control resampled to a token budget, but has no pairs.
                if cfg['replay_fraction'] != 1:
                    raise ValueError('Cannot infer mixed target-token exposure without its training stream')
                counts = per_epoch = {'pairs': {}, 'depths': {'1': None}}
            result[run['adapter_hash']] = {'stage': 'historical_sft_reconstruction',
                **counts, 'per_epoch': per_epoch,
                'epochs': cfg['epochs'], 'reference_manifest_sha256': digest, 'config': cfg,
                'history_receipt_sha256': file_hash(path),
                'evidence_scope': 'deterministic mixture reconstruction, not an independently observed training stream'}
    return result


def modular_solutions(matrix, rhs, p):
    """Particular solution and null-space basis by exact elimination in GF(p)."""
    a = np.column_stack((matrix, rhs)).astype(np.int64) % p
    row, pivots = 0, []
    for column in range(a.shape[1] - 1):
        choices = np.flatnonzero(a[row:, column])
        if not len(choices):
            continue
        pivot = row + int(choices[0])
        a[[row, pivot]] = a[[pivot, row]]
        a[row] = a[row] * pow(int(a[row, column]), -1, p) % p
        for other in range(len(a)):
            if other != row:
                a[other] = (a[other] - a[other, column] * a[row]) % p
        pivots.append(column)
        row += 1
        if row == len(a):
            break
    if any(not r[:-1].any() and r[-1] for r in a):
        return None
    n = a.shape[1] - 1
    origin = np.zeros(n, dtype=np.int64)
    origin[pivots] = a[:len(pivots), -1]
    basis = []
    for column in sorted(set(range(n)) - set(pivots)):
        vector = np.zeros(n, dtype=np.int64)
        vector[column] = 1
        vector[pivots] = -a[:len(pivots), column] % p
        basis.append(vector)
    return origin, np.asarray(basis, dtype=np.int64).reshape(-1, n)


def crossed_rows(name, count, spec, registry, seed, family, attempts_per_pair=24):
    """Search exact affine crossing equations; report bounded-search coverage honestly."""
    rng, rows, report = random.Random(seed), [], []
    fields = core.TRANSFER_FIELDS if family in ('C', 'D') else core.KNOWN_FIELDS
    shapes = list(itertools.product(fields, core.DEGREES))
    rng.shuffle(shapes)
    wanted = 1 if family in ('B', 'D') else 0
    candidates = []
    for p, degree in shapes:
        programs, matrices, offsets = affine_maps(p, degree, 3)
        sig = signatures(p, degree, 3)
        shorter = {s for d in range(3) for s in signatures(p, degree, d).values()}
        multiplicity = Counter(sig.values())
        eligible = [i for i, program in enumerate(programs) if multiplicity[sig[program]] == 1
                    and sig[program] not in shorter and constraint_hits(program, spec) == wanted]
        pairs = list(itertools.combinations(eligible, 2))
        rng.shuffle(pairs)
        stats = {'p': p, 'degree': degree, 'eligible_programs': len(eligible),
                 'candidate_pairs': len(pairs), 'solvable_pairs': 0, 'draws': 0, 'accepted': 0}
        for i, j in pairs:
            if sum(a != b for a, b in zip(programs[i], programs[j])) < 2:
                continue
            aa, bb, ac, bc = matrices[i], matrices[j], offsets[i], offsets[j]
            solution = modular_solutions(np.block([[aa, -bb], [bb, -aa]]),
                                         np.concatenate((bc - ac, ac - bc)), p)
            if solution is None:
                continue
            stats['solvable_pairs'] += 1
            candidates.append((p, degree, programs[i], programs[j], solution, stats))
        report.append(stats)
    rng.shuffle(candidates)
    for p, degree, a, b, (origin, basis), stats in candidates:
        if len(rows) // 4 >= count:
            break
        for _ in range(attempts_per_pair):
            stats['draws'] += 1
            weights = np.array([rng.randrange(p) for _ in basis], dtype=np.int64)
            state = (origin + weights @ basis) % p
            s1, s2 = map(tuple, np.split(state, 2))
            if s1 == s2:
                continue
            t1, t2 = core.trajectory(s1, a, p)[-1], core.trajectory(s1, b, p)[-1]
            if t1 == t2:
                continue
            cells = [(s1, t1, a), (s1, t2, b), (s2, t1, b), (s2, t2, a)]
            states, selected = set(), []
            for index, (start, target, program) in enumerate(cells):
                correct = reachable(start, p, 3)[target]
                if correct != [program] or any(target in reachable(start, p, d) for d in range(3)):
                    break
                fp = core.canonical_task_fingerprint(p, start, target, 3)
                row = {**spec, 'schema': core.TASK_SCHEMA, 'p': p, 'degree': degree, 'depth': 3,
                       'start': list(map(int, start)), 'target': list(target), 'witness': list(program),
                       'correct_programs': [list(program)], 'correct_count': 1, 'shortest_depth': 3,
                       'task_fingerprint': fp, 'task_id': 's4c-' + fp[:24], 'split': name,
                       'family': family, 'panel_kind': 'crossed', 'cell': index,
                       'program_A': list(a), 'program_B': list(b)}
                states.update(_full_states(row, [program]))
                selected.append(row)
            if len(selected) != 4 or states & registry.states or any(r['task_fingerprint'] in registry.tasks for r in selected):
                continue
            panel_id = hashlib.sha256(json.dumps([name, p, list(map(int, state)), a, b]).encode()).hexdigest()[:24]
            for row in selected:
                row['panel_id'] = panel_id
            rows.extend(selected)
            registry.states.update(states)
            registry.tasks.update(r['task_fingerprint'] for r in selected)
            registry.shape_counts[p, degree] += len(states)
            stats['accepted'] += 1
            break
    return rows, {'requested_panels': count, 'accepted_panels': len(rows) // 4,
                  'search': 'all eligible affine program pairs; bounded random null-space draws, not an exhaustive state search',
                  'attempts_per_pair': attempts_per_pair, 'strata': report}


def check_panels(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['panel_id']].append(row)
    for panel in grouped.values():
        if len(panel) != 4 or any(r['correct_count'] != 1 for r in panel):
            raise ValueError('Panels require four uniquely solvable cells')
        if len({(r['p'], len(r['start']), r['depth']) for r in panel}) != 1:
            raise ValueError('Panel cells must share field, dimension, and depth')
        if panel[0].get('panel_kind') == 'crossed':
            panel.sort(key=lambda r: r['cell'])
            if [r['cell'] for r in panel] != list(range(4)):
                raise ValueError('Crossed cell order is incomplete')
            a, b = panel[0]['program_A'], panel[0]['program_B']
            if ([r['witness'] for r in panel] != [a, b, b, a]
                    or panel[0]['start'] != panel[1]['start'] or panel[2]['start'] != panel[3]['start']
                    or panel[0]['start'] == panel[2]['start']
                    or panel[0]['target'] != panel[2]['target'] or panel[1]['target'] != panel[3]['target']
                    or panel[0]['target'] == panel[1]['target']):
                raise ValueError('Invalid START x TARGET crossing')
        elif len({tuple(r['start']) for r in panel}) != 1 or len({tuple(r['target']) for r in panel}) != 4:
            raise ValueError('Invalid four-TARGET panel')
    return list(grouped.values())


def audit_rows(splits, inherited=None):
    owners, tasks, support = {}, set(), {}
    for split, rows in splits.items():
        counts, pairs, strata = Counter(), Counter(), Counter()
        for row in rows:
            correct = [p for p in core.enumerate_programs(row['depth'])
                       if core.verify_program(row['start'], row['target'], p, row['p'])]
            if sorted(correct) != sorted(map(tuple, row['correct_programs'])):
                raise ValueError('Correct set differs from independent interpreter enumeration')
            if len(correct) != row['correct_count'] or tuple(row['witness']) not in correct:
                raise ValueError('Wrong witness or correct count')
            if any(tuple(row['target']) in reachable(row['start'], row['p'], d) for d in range(row['depth'])):
                raise ValueError('Task has a shorter solution')
            wanted = 1 if row['family'] in ('B', 'D') else 0
            if row['family'] != 'ATOMIC' and {constraint_hits(p, row) for p in correct} != {wanted}:
                raise ValueError('Withheld semantics violated by a correct program')
            fp = core.canonical_task_fingerprint(row['p'], row['start'], row['target'], row['depth'])
            if fp != row['task_fingerprint'] or fp in tasks:
                raise ValueError('Changed identity or duplicated task')
            tasks.add(fp)
            states = _full_states(row, correct)
            owner = (split, row.get('panel_id', row['task_id']))
            if inherited and (states & inherited.states or fp in inherited.tasks):
                raise ValueError('New task overlaps inherited data')
            if any(s in owners and owners[s] != owner for s in states):
                raise ValueError('Shared state outside one panel')
            owners.update(dict.fromkeys(states, owner))
            counts.update(' '.join(p) for p in correct)
            pairs.update(f'{i}:{a}>{b}' for p in correct for i, (a, b) in enumerate(zip(p, p[1:])))
            strata[f"{row['p']}:{row['degree']}:{row['depth']}:{len(correct)}"] += 1
        if rows and rows[0].get('panel_id'):
            check_panels(rows)
        support[split] = {'tasks': len(rows), 'panels': len({r['panel_id'] for r in rows if 'panel_id' in r}),
                          'program_union': len(counts), 'program_counts': counts,
                          'position_pairs': pairs, 'strata': strata}
    return {'status': 'PASS', 'tasks': len(tasks), 'states': len(owners), 'support': support}


def prepare(source, reference, out, train_panels=64, eval_panels=16, seed=20260923, smoke=False):
    source, reference, out = map(Path, (source, reference, out))
    if min(train_panels, eval_panels) < 1 or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Positive panel counts and a valid seed required')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Frozen datasets are immutable; choose another --out')
    protocol = json.loads((source / 'protocol.json').read_text())
    if protocol['schema'] != 'iclr.upgrade.protocol.v2' or protocol['reference_manifest_sha256'] != file_hash(reference / 'manifest.json'):
        raise ValueError('Expected v2 data and their original reference')
    if bool(protocol['smoke']) != smoke:
        raise ValueError('Smoke and scientific input manifests cannot be mixed')
    if smoke:
        train_panels, eval_panels = 2, 1
    out.mkdir(parents=True, exist_ok=True)
    inherited = reference_registry(reference)
    history = historical_exposure(reference)
    manifests, reports = {}, {}
    for index, name in enumerate(('original', 'mask1', 'mask2')):
        directory, prior = out / name, source / name
        manifest = verify_data(prior)
        if file_hash(prior / 'manifest.json') != protocol['datasets'][name]['manifest_sha256']:
            raise ValueError('Source dataset changed')
        spec = manifest['constraints']
        registry = reference_registry(prior)
        registry.states.update(inherited.states)
        registry.tasks.update(inherited.tasks)
        registry.shape_counts = Counter()
        # Counts are a capacity heuristic only; exact state sets enforce exclusions.
        for origin in (reference, prior):
            for filename in verify_data(origin)['files']:
                for row in read_jsonl(origin / filename):
                    registry.shape_counts[row['p'], len(row['start']) - 1] += len(_full_states(row, row.get('correct_programs', [row['witness']])))
        # Avoid double-counting reused atomic states in the finite-capacity heuristic.
        registry.shape_counts = Counter({shape: min(n, shape[0] ** (shape[1] + 1)) for shape, n in registry.shape_counts.items()})
        before = core.FingerprintRegistry(set(registry.states), set(registry.tasks))
        fresh, searches = {}, {}
        local_seed = seed + index * 100
        fresh['train_four'] = generate_rows('train_four', train_panels * 4, spec, registry, local_seed,
                                            family='TRAIN', panel_targets=4, correct_count=1)
        for j, (phase, family) in enumerate(itertools.product(('dev', 'final'), 'ABCD')):
            split = f'{phase}4_{family}'
            if split + '.jsonl' not in manifest['files']:
                fresh[split] = generate_rows(split, eval_panels, spec, registry, local_seed + 30 + j,
                                            family=family, depth=4)
        for split, family, count, offset in [('train_crossed', 'TRAIN', train_panels, 1),
                *[(f'{phase}_crossed_{family}', family, eval_panels, 2 + i)
                  for i, (phase, family) in enumerate(itertools.product(('dev', 'final'), 'ABCD'))]]:
            fresh[split], searches[split] = crossed_rows(split, count, spec, registry, local_seed + offset, family)
        report = audit_rows(fresh, before)
        cross = report['support']['train_crossed']
        use_crossed = cross['panels'] >= train_panels and cross['program_union'] >= 8
        files = {}
        directory.mkdir()
        # Reuse v2 ordinary tasks and held-out four-TARGET panels byte-for-byte.
        for filename, entry in manifest['files'].items():
            (directory / filename).write_bytes((prior / filename).read_bytes())
            files[filename] = entry
        for split, rows in fresh.items():
            path = directory / (split + '.jsonl')
            path.write_bytes(core.canonical_jsonl_bytes(rows))
            files[path.name] = {'rows': len(rows), 'sha256': file_hash(path)}
        training = 'train_crossed.jsonl' if use_crossed else 'train_four.jsonl'
        write_json(directory / 'manifest.json', {**manifest, 'schema': 'iclr.research.data.v4', 'files': files,
            'source_manifest_sha256': file_hash(prior / 'manifest.json'), 'training_panels': training,
            'historical_exposure': history,
            'crossed_training_gate': {'pass': use_crossed, 'minimum_panels': train_panels, 'minimum_programs': 8},
            'length_transfer': 'new continuations: atomic base and depth3-only train; historical SFT saw depth4'})
        reports[name] = {**report, 'crossed_search': searches, 'selected_training_panels': training}
        manifests[name] = {'manifest_sha256': file_hash(directory / 'manifest.json'), 'constraints': spec}
        print(f'{name}: train={training}; crossed support={cross["panels"]} panels/{cross["program_union"]} programs', flush=True)
    write_json(out / 'audit.json', {'schema': 'iclr.research.audit.v4', 'status': 'PASS', 'datasets': reports,
        'scope': 'new v4 panels: independent full interpreter, shortest depth, all-solution states, input/split exclusion; reused v2 files: exact hashes',
        'smoke': smoke})
    write_json(out / 'protocol.json', {'schema': 'iclr.research.protocol.v4', 'smoke': smoke, 'seed': seed,
        'datasets': manifests, 'source_protocol_sha256': file_hash(source / 'protocol.json'),
        'reference_manifest_sha256': file_hash(reference / 'manifest.json'),
        'analysis_plan_sha256': file_hash(ROOT / 'plans/research_v4.json'), 'audit_sha256': file_hash(out / 'audit.json')})
    return reports


def monitor_selection(data, manifest):
    data = Path(data)
    atomic = read_jsonl(data / 'dev_atomic.jsonl')
    train = check_panels(read_jsonl(data / manifest['training_panels']))
    return {
        'atomic': [r['task_id'] for op in core.OPS for r in sorted(
            (r for r in atomic if r['witness'] == [op]), key=lambda r: r['task_id'])[:20]],
        'ordinary': [r['task_id'] for f in 'ABCD' if (data / f'dev_{f}.jsonl').exists()
                     for r in sorted(read_jsonl(data / f'dev_{f}.jsonl'), key=lambda r: r['task_id'])[:16]],
        'crossed': [r['task_id'] for f in 'BD' if (data / f'dev_crossed_{f}.jsonl').exists()
                    for r in read_jsonl(data / f'dev_crossed_{f}.jsonl')],
        'train': [r['task_id'] for panel in sorted(train, key=lambda p: p[0]['panel_id'])[:4] for r in panel],
        'reference': [r['task_id'] for r in sorted(read_jsonl(data / 'dev_A.jsonl'), key=lambda r: r['task_id'])[:64]]}


def prepare_amendment(data, out, per_stratum=8, seed=20260923, plan='v5'):
    """Fresh train-only atomic replay; the parent dataset is never modified."""
    from .research_io import plan_path
    data, out = Path(data), Path(out)
    manifest = verify_data(data)
    if manifest['schema'] != 'iclr.research.data.v4' or manifest.get('domain', 'affine_polynomial_v1') != 'affine_polynomial_v1':
        raise ValueError('This amendment is for the frozen affine v4 inputs')
    if type(per_stratum) is not int or per_stratum < 1:
        raise ValueError('Positive replay count per operation/field required')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Amendments are immutable; choose a new output')
    registry = reference_registry(data)
    rng, rows = random.Random(seed), []
    for index in range(per_stratum):
        strata = list(itertools.product(core.OPS, core.KNOWN_FIELDS))
        rng.shuffle(strata)
        for op, p in strata:
            for _ in range(20000):
                degree = rng.choice(core.DEGREES)
                start = [rng.randrange(p) for _ in range(degree + 1)]
                target = list(core.trajectory(start, [op], p)[-1])
                if start == target or not any(start):
                    continue
                correct = [list(z) for z in core.enumerate_programs(1) if core.verify_program(start, target, z, p)]
                row = dict(p=p, degree=degree, depth=1, start=start, target=target, witness=[op],
                           correct_programs=correct, correct_count=len(correct), family='ATOMIC', split='train_replay')
                fp = core.canonical_task_fingerprint(p, start, target, 1)
                states = _full_states(row, [[op]])
                if correct != [[op]] or states & registry.states or fp in registry.tasks:
                    continue
                rows.append({**row, 'task_id': 'replay-' + fp[:24], 'task_fingerprint': fp})
                registry.states.update(states); registry.tasks.add(fp)
                break
            else:
                raise RuntimeError(f'Replay capacity exhausted for {op}/{p}')
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'atomic_replay.jsonl'
    path.write_bytes(core.canonical_jsonl_bytes(rows))
    write_json(out / 'manifest.json', {'schema': 'iclr.research.amendment.v5',
        'parent_manifest_sha256': file_hash(data / 'manifest.json'), 'plan_hash': file_hash(plan_path(plan)),
        'seed': seed, 'per_operation_field': per_stratum, 'modes': ['PLAN', 'APPLY'],
        'files': {path.name: {'sha256': file_hash(path), 'rows': len(rows)}},
        'monitor': monitor_selection(data, manifest),
        'exclusions': 'all parent task identities and full correct-trajectory states, including final inputs; no final scores or outcomes used'})
    verify_amendment(out / 'manifest.json', data, file_hash(plan_path(plan)))


def verify_amendment(path, data, plan_hash):
    path, data = Path(path), Path(data)
    amended, parent = verify_data(path.parent), verify_data(data)
    if (path.name != 'manifest.json' or amended['schema'] != 'iclr.research.amendment.v5'
            or amended['parent_manifest_sha256'] != file_hash(data / 'manifest.json')
            or amended['plan_hash'] != plan_hash or amended['modes'] != ['PLAN', 'APPLY']
            or amended['monitor'] != monitor_selection(data, parent)):
        raise ValueError('Amendment differs from its frozen parent/plan/monitor')
    registry = reference_registry(data)
    balance = Counter()
    rows = read_jsonl(path.parent / 'atomic_replay.jsonl')
    for row in rows:
        fp = core.canonical_task_fingerprint(row['p'], row['start'], row['target'], 1)
        if (row['split'] != 'train_replay' or row['family'] != 'ATOMIC' or row['depth'] != 1
                or row['witness'][0] not in core.OPS or len(row['witness']) != 1 or row['p'] not in core.KNOWN_FIELDS
                or row['task_fingerprint'] != fp or row['task_id'] != 'replay-' + fp[:24]
                or not core.verify_program(row['start'], row['target'], row['witness'], row['p'])):
            raise ValueError('Invalid train-only replay row')
        states = _full_states(row, [row['witness']])
        if states & registry.states or fp in registry.tasks:
            raise ValueError('Replay overlaps parent or other replay states/tasks')
        registry.states.update(states); registry.tasks.add(fp)
        balance[row['witness'][0], row['p']] += 1
    expected = Counter({(op, p): amended['per_operation_field'] for op in core.OPS for p in core.KNOWN_FIELDS})
    if balance != expected:
        raise ValueError('Replay must balance all five operations and known train fields')
    return amended


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source')
    parser.add_argument('--reference')
    parser.add_argument('--amend-data', help='Prepare v5 replay/monitor beside this immutable v4 dataset')
    parser.add_argument('--amend-plan', choices=['v5', 'v6'], default='v5')
    parser.add_argument('--out', required=True)
    parser.add_argument('--train-panels', type=int, default=64)
    parser.add_argument('--eval-panels', type=int, default=16)
    parser.add_argument('--seed', type=int, default=20260923)
    parser.add_argument('--smoke', action='store_true')
    args = vars(parser.parse_args())
    amended = args.pop('amend_data')
    amendment_plan = args.pop('amend_plan')
    if amended:
        prepare_amendment(amended, args['out'], seed=args['seed'], plan=amendment_plan)
    else:
        if not args['source'] or not args['reference']:
            parser.error('--source and --reference are required for v4 generation')
        prepare(**args)


if __name__ == '__main__':
    main()
