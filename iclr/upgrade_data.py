"""Fresh closed panels, exact semantic mask feasibility, and controlled witnesses."""

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import random

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

import composition_core as core
from .common import file_hash, read_jsonl, verify_data, write_json
from .data import _atomic_split, _full_states, constraint_hits, correct_programs


@lru_cache(maxsize=None)
def affine_maps(p, degree, depth):
    """Exact affine maps from their action on zero and basis vectors over F_p."""
    programs = [()] if depth == 0 else sorted(core.enumerate_programs(depth))
    linear, offsets = [], []
    for program in programs:
        values = []
        for source in [np.zeros(degree + 1, dtype=int), *np.eye(degree + 1, dtype=int)]:
            state = tuple(int(x) for x in source)
            for op in program:
                state = core.apply_op(state, op, p)
            values.append(state)
        values = np.asarray(values, dtype=np.int64)
        offsets.append(values[0])
        linear.append(((values[1:] - values[0]) % p).T)
    return programs, np.asarray(linear), np.asarray(offsets)


def signatures(p, degree, depth):
    programs, linear, offsets = affine_maps(p, degree, depth)
    return {program: tuple(np.concatenate([a.ravel(), b]))
            for program, a, b in zip(programs, linear, offsets)}


def feasibility(spec):
    report = []
    for p, degree in itertools.product((*core.KNOWN_FIELDS, *core.TRANSFER_FIELDS), core.DEGREES):
        shorter = {s for d in range(3) for s in signatures(p, degree, d).values()}
        classes = defaultdict(list)
        for program, signature in signatures(p, degree, 3).items():
            classes[signature].append(program)
        counts = Counter()
        for signature, programs in classes.items():
            if signature in shorter:
                continue
            hits = {constraint_hits(program, spec) for program in programs}
            if hits == {0}:
                counts['train_classes'] += 1
                counts['multisolution_train_classes'] += len(programs) >= 2
            if hits == {1}:
                counts['strict_heldout_classes'] += 1
                for edge in spec.get('heldout_motifs', []):
                    if any(tuple(edge) in list(zip(program, program[1:])) for program in programs):
                        counts['edge:' + '>'.join(edge)] += 1
        report.append({'p': p, 'degree': degree, **dict.fromkeys(
            ('train_classes', 'multisolution_train_classes', 'strict_heldout_classes'), 0), **counts})
    return report


def choose_masks(seed=20260921, count=4):
    rng = random.Random(seed)
    pairs = list(itertools.product(core.OPS, repeat=2))
    selected, seen = [], {tuple(sorted(core.HELDOUT_MOTIFS))}
    for _ in range(10000):
        mask = tuple(sorted(rng.sample(pairs, 3)))
        if mask in seen:
            continue
        seen.add(mask)
        spec = {'heldout_motifs': [list(p) for p in mask]}
        report = feasibility(spec)
        if min(r['strict_heldout_classes'] for r in report) < 4:
            continue
        if min(r['multisolution_train_classes'] for r in report) < 4:
            continue
        if any(not any(r.get('edge:' + '>'.join(edge), 0) for r in report) for edge in mask):
            continue
        selected.append(spec)
        if len(selected) == count:
            return selected
    raise RuntimeError('Not enough feasible masks; do not silently weaken the task semantics')


def reachable(start, p, depth):
    programs, linear, offsets = affine_maps(p, len(start) - 1, depth)
    targets = (np.einsum('pij,j->pi', linear, np.asarray(start)) + offsets) % p
    grouped = defaultdict(list)
    for program, target in zip(programs, targets):
        grouped[tuple(int(v) for v in target)].append(program)
    return grouped


def matched_program_schedule(spec, p, degree, count, seed, available=None):
    """Integer quotas for operation marginals and solution counts before sampling START."""
    classes = defaultdict(list)
    shorter = {s for d in range(3) for s in signatures(p, degree, d).values()}
    for program, signature in signatures(p, degree, 3).items():
        classes[signature].append(program)
    candidates = [(program, len(programs)) for signature, programs in classes.items()
                  if signature not in shorter and len(programs) <= 3
                  and all(constraint_hits(program, spec) == 0 for program in programs)
                  for program in programs if available is None or (program, len(programs)) in available]
    if not candidates:
        raise ValueError('No clean programs for matched training')
    matrix = np.array([[1, *[program.count(op) for op in core.OPS],
                       *[int(n == k) for k in (1, 2, 3)]] for program, n in candidates], dtype=float).T
    solution_counts = [round(.6 * count), round(.2 * count)]
    solution_counts.append(count - sum(solution_counts))
    lower = [count, *([np.floor(3 * count / 5)] * 5), *solution_counts]
    upper = [count, *([np.ceil(3 * count / 5)] * 5), *solution_counts]
    # Preserve available pair support in the larger strata; the objective keeps
    # program counts close to uniform instead of selecting a sparse LP solution.
    if count >= len(candidates):
        for pair in itertools.product(core.OPS, repeat=2):
            values = [int(pair in list(zip(program, program[1:]))) for program, _ in candidates]
            if any(values):
                matrix = np.vstack([matrix, values])
                lower.append(1)
                upper.append(np.inf)
    n = len(candidates)
    target = np.full(n, count / n)
    constraints = [LinearConstraint(np.column_stack([matrix, np.zeros_like(matrix)]), lower, upper),
                   LinearConstraint(np.block([[np.eye(n), -np.eye(n)], [-np.eye(n), -np.eye(n)]]),
                                    -np.inf, np.concatenate([target, -target]))]
    rng = np.random.default_rng(seed)
    result = milp(np.concatenate([rng.random(n) * 1e-7, np.ones(n)]),
                  integrality=np.concatenate([np.ones(n), np.zeros(n)]),
                  bounds=Bounds(0, np.inf), constraints=constraints, options={'time_limit': 10})
    if result.x is None:
        raise ValueError(f'No matched operation/solution-count schedule for p={p}, degree={degree}, n={count}')
    quotas = np.rint(result.x[:n]).astype(int)
    achieved = matrix @ quotas
    if np.any(achieved < np.asarray(lower) - 1e-6) or np.any(achieved > np.asarray(upper) + 1e-6):
        raise ValueError('Integer matching returned an infeasible schedule')
    schedule = [candidate for candidate, quota in zip(candidates, quotas) for _ in range(quota)]
    random.Random(seed).shuffle(schedule)
    return schedule


def generate_rows(name, count, spec, registry, seed, *, family='A', depth=3,
                  panel_targets=1, correct_count=None, required_pairs=(), required_positions=(), matched_training=False,
                  excluded_shapes=(), available_programs=None):
    if count < 1 or count % panel_targets:
        raise ValueError('Positive task count divisible by panel size required')
    rng, rows = random.Random(seed), []
    fields = core.TRANSFER_FIELDS if family in ('C', 'D') else core.KNOWN_FIELDS
    shapes = list(itertools.product(fields, core.DEGREES))
    # Equal small-field quotas exhaust finite state spaces. Freeze capacity-aware
    # quotas before drawing any task; never select strata from model outcomes.
    capacities = np.array([0 if (p, d) in excluded_shapes else max(0, p ** (d + 1) - registry.shape_counts[p, d]) // 64
                           for p, d in shapes])
    weights = np.sqrt([p ** (d + 1) for p, d in shapes]) * (capacities > 0)
    group_count = count // panel_targets
    if capacities.sum() < group_count or not weights.sum():
        raise ValueError(f'Insufficient disjoint state capacity for {name}')
    desired = group_count * weights / weights.sum()
    quotas = np.minimum(capacities, np.floor(desired).astype(int))
    while quotas.sum() < group_count:
        index = int(np.argmax(np.where(quotas < capacities, desired - quotas, -np.inf)))
        quotas[index] += 1
    schedule = [shape for shape, quota in zip(shapes, quotas) for _ in range(int(quota))]
    rng.shuffle(schedule)
    if matched_training and (family != 'TRAIN' or depth != 3 or panel_targets != 1 or correct_count is not None):
        raise ValueError('Matched quotas require ordinary depth-three training tasks')
    program_schedules = {shape: matched_program_schedule(spec, *shape, int(quota), seed + i,
                         available=available_programs.get(shape) if available_programs else None)
                         for i, (shape, quota) in enumerate(zip(shapes, quotas)) if quota} if matched_training else {}
    shape_cursors = Counter()
    wanted = 1 if family in ('B', 'D') else 0
    attempts, attempts_for_slot = 0, 0
    while len(rows) < count:
        attempts += 1
        attempts_for_slot += 1
        if attempts_for_slot > 10000 or attempts > count * 2000:
            raise RuntimeError(f'Exhausted strict generation for {name}: {len(rows)}/{count}')
        p, degree = schedule[len(rows) // panel_targets]
        start = [rng.randrange(p) for _ in range(degree + 1)]
        shorter = {target for d in range(depth) for target in reachable(start, p, d)}
        planned = program_schedules[p, degree][shape_cursors[p, degree]] if matched_training else None
        candidates = [(target, programs) for target, programs in reachable(start, p, depth).items()
                      if target not in shorter and (correct_count is None or len(programs) == correct_count)
                      and (planned is None or planned[0] in programs and len(programs) == planned[1])
                      and {constraint_hits(program, spec) for program in programs} == {wanted}]
        rng.shuffle(candidates)
        def eligible_witness(program):
            if planned is not None:
                return program == planned[0]
            slot = len(rows)
            if slot < len(required_pairs):
                return tuple(required_pairs[slot]) in list(zip(program, program[1:]))
            slot -= len(required_pairs)
            if slot < len(required_positions):
                index, a, b = required_positions[slot]
                return tuple(program[index:index + 2]) == (a, b)
            return True
        selected, states = [], set()
        for target, programs in candidates:
            witnesses = [p for p in programs if eligible_witness(p)]
            if not witnesses:
                continue
            row = {'p': p, 'start': start, 'target': list(target), 'depth': depth}
            trajectory_states = _full_states(row, programs)
            if trajectory_states & registry.states:
                continue
            fingerprint = core.canonical_task_fingerprint(p, start, target, depth)
            if fingerprint in registry.tasks:
                continue
            witness = list(rng.choice(witnesses))
            selected.append({**row, **spec, 'schema': core.TASK_SCHEMA,
                'task_id': 's4c-' + fingerprint[:24], 'task_fingerprint': fingerprint,
                'split': name, 'family': family, 'degree': degree, 'witness': witness,
                'states': [list(s) for s in core.trajectory(start, witness, p)],
                'correct_count': len(programs), 'shortest_depth': depth,
                'shortest_solution_count': len(programs),
                'correct_programs': [list(p) for p in programs]})
            states.update(trajectory_states)
            if len(selected) == panel_targets:
                break
        if len(selected) != panel_targets:
            continue
        if panel_targets > 1:
            panel_id = hashlib.sha256(json.dumps([name, p, start]).encode()).hexdigest()[:24]
            for row in selected:
                row['panel_id'] = panel_id
        registry.states.update(states)
        registry.shape_counts[p, degree] += len(states)
        registry.tasks.update(row['task_fingerprint'] for row in selected)
        rows.extend(selected)
        shape_cursors[p, degree] += panel_targets
        attempts_for_slot = 0
    return rows


def reference_registry(data):
    manifest = verify_data(data)
    registry = core.FingerprintRegistry()
    registry.shape_counts = Counter()
    for name in manifest['files']:
        for row in read_jsonl(Path(data) / name):
            programs = correct_programs(row)
            states = _full_states(row, programs)
            registry.shape_counts[row['p'], len(row['start']) - 1] += len(states - registry.states)
            registry.states.update(states)
            registry.tasks.add(row['task_fingerprint'])
    return registry


def preflight_small_strata(registry, specs):
    """Enumerate remaining states where the inherited corpus occupied most of a cell."""
    available = {name: {} for name in specs}
    excluded, evidence = [], []
    for p, degree in itertools.product(core.KNOWN_FIELDS, core.DEGREES):
        total = p ** (degree + 1)
        free = total - registry.shape_counts[p, degree]
        if free < 64 or free >= total / 2:
            continue
        pools = {name: set() for name in specs}
        counts = {name: Counter() for name in specs}
        for start in itertools.product(range(p), repeat=degree + 1):
            if core.state_fingerprint(p, start) in registry.states:
                continue
            shorter = {t for d in range(3) for t in reachable(start, p, d)}
            for target, programs in reachable(start, p, 3).items():
                if target in shorter or len(programs) > 3 or core.state_fingerprint(p, target) in registry.states:
                    continue
                if _full_states({'p': p, 'start': start, 'target': target}, programs) & registry.states:
                    continue
                for name, spec in specs.items():
                    if all(constraint_hits(program, spec) == 0 for program in programs):
                        counts[name][len(programs)] += 1
                        pools[name].update((program, len(programs)) for program in programs)
        blocked = any(set(counts[name]) != {1, 2, 3} for name in specs)
        if blocked:
            excluded.append((p, degree))
        for name in specs:
            available[name][p, degree] = pools[name]
        evidence.append({'p': p, 'degree': degree, 'all_starts_enumerated': total,
                         'remaining_states': free, 'valid_tasks_by_solution_count': counts,
                         'excluded_from_all_matched_training': blocked})
        print(f'Preflight F{p}/degree{degree}: excluded={blocked}', flush=True)
    return excluded, available, evidence


def prepare(out, reference, *, seed=20260921, size=5000, eval_size=200, panels=50, smoke=False):
    out, reference = Path(out), Path(reference)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite a frozen data directory: {out}')
    if smoke:
        size, eval_size, panels = 20, 5, 2
    elif min(size, eval_size, panels) < 1:
        raise ValueError('Positive data sizes required')
    inherited = reference_registry(reference)
    reference_states = len(inherited.states)
    out.mkdir(parents=True, exist_ok=True)
    specs = {'original': {'heldout_motifs': [list(p) for p in core.HELDOUT_MOTIFS]},
             **{f'mask{i + 1}': s for i, s in enumerate(choose_masks(seed))},
             'witness': {'heldout_motifs': [list(p) for p in core.HELDOUT_MOTIFS]},
             'triple': {'heldout_motifs': [], 'heldout_triples': [['SH1', 'SC2', 'REV']]},
             'position': {'heldout_motifs': [], 'heldout_positions': [[0, 'SH1', 'SC2']]}}
    matched_specs = {k: v for k, v in specs.items() if k == 'original' or k.startswith('mask')}
    excluded_shapes, available, strata_evidence = preflight_small_strata(inherited, matched_specs)
    write_json(out / 'finite_state_preflight.json', {'scope': 'exhaustive remaining-START feasibility before training quotas',
                                                  'cells': strata_evidence, 'excluded_training_shapes': excluded_shapes})
    single_edges = [{'heldout_motifs': [list(pair)], 'cells': feasibility({'heldout_motifs': [list(pair)]})}
                    for pair in itertools.product(core.OPS, repeat=2)]
    write_json(out / 'mask_feasibility.json', {'scope': 'affine function classes at minimum depth 3, not counts of tasks',
                                             'single_edges': single_edges,
                                             'selected': {name: feasibility(spec) for name, spec in specs.items()}})
    manifests = {}
    for i, (name, spec) in enumerate(specs.items()):
        # Each mask is an independent experiment sharing only the inherited base.
        registry = core.FingerprintRegistry(set(inherited.states), set(inherited.tasks))
        registry.shape_counts = inherited.shape_counts.copy()
        directory = out / name
        directory.mkdir()
        # Match masks on task count, field/degree, operation marginals and the
        # distribution of full solution counts. Witness selection has its own pool.
        files, rows_by_name = {}, {}
        def save(split, rows):
            path = directory / f'{split}.jsonl'
            path.write_bytes(core.canonical_jsonl_bytes(rows))
            files[path.name] = {'rows': len(rows), 'sha256': file_hash(path),
                'strata': dict(Counter(f"{r['p']}:{r['degree']}:{r['depth']}:{r['correct_count']}" for r in rows))}
            rows_by_name[split] = rows
        local_seed = seed + 100 * i
        save('train', generate_rows('train', size, spec, registry, local_seed, family='TRAIN',
             correct_count=2 if name == 'witness' else None,
             matched_training=name not in ('witness', 'triple', 'position'),
             excluded_shapes=excluded_shapes if name in matched_specs else (),
             available_programs=available.get(name),
             required_pairs=[('SH1', 'SC2'), ('SC2', 'REV')] if name == 'triple' else (),
             required_positions=[(1, 'SH1', 'SC2')] if name == 'position' else ()))
        save('atomic_train', read_jsonl(reference / 'atomic_train.jsonl'))
        for j, family in enumerate('ABCD'):
            save(f'dev_{family}', generate_rows(f'dev_{family}', eval_size, spec, registry, local_seed + j + 1, family=family))
            save(f'final_{family}', generate_rows(f'final_{family}', eval_size, spec, registry, local_seed + j + 11, family=family))
            if name == 'original':
                save(f'final4_{family}', generate_rows(f'final4_{family}', eval_size, spec, registry, local_seed + j + 21, family=family, depth=4))
                save(f'panel_{family}', generate_rows(f'panel_{family}', panels * 4, spec, registry,
                     local_seed + j + 31, family=family, panel_targets=4))
        # A used for calibration is independent of both checkpoint-selection dev and final.
        save('calibration_A', generate_rows('calibration_A', eval_size, spec, registry, local_seed + 41))
        for j, phase in enumerate(('dev', 'final')):
            save(phase + '_atomic', _atomic_split(phase + '_atomic', eval_size, local_seed + 50 + j, registry))
        observed_pairs = {pair for row in rows_by_name['train'] for pair in zip(row['witness'], row['witness'][1:])}
        if name == 'triple' and not {('SH1', 'SC2'), ('SC2', 'REV')} <= observed_pairs:
            raise ValueError('Held triple has a constituent pair absent from training')
        if name == 'position' and not any(row['witness'][1:] == ['SH1', 'SC2'] for row in rows_by_name['train']):
            raise ValueError('Held positional pair never occurs at the other position in training')
        manifest = {'schema': 'iclr.upgrade.data.v1', 'status': 'smoke' if smoke else 'frozen_new_evaluation',
            'config': {'size': size, 'eval_size': eval_size, 'panels': panels, 'seed': local_seed},
            'constraints': spec, 'files': files, 'reference_data_sha256': file_hash(reference / 'manifest.json'),
            'state_exclusion': 'within each mask: all correct trajectories; shared states permitted only inside a matched-START panel; inherited atomic training reused',
            'observed_training_pairs': [list(p) for p in sorted(observed_pairs)]}
        write_json(directory / 'manifest.json', manifest)
        manifests[name] = {'manifest_sha256': file_hash(directory / 'manifest.json'), 'constraints': spec}
        print(f'Prepared {name}: {sum(f["rows"] for f in files.values())} tasks', flush=True)
    write_json(out / 'protocol.json', {'schema': 'iclr.upgrade.protocol.v1', 'seed': seed, 'smoke': smoke,
        'primary_metric': 'Hit@32', 'calibration_alpha': 1, 'calibration_depth': 3,
        'calibration_family': 'A', 'training_seeds': [0, 1, 2], 'mask_training_seeds': [0, 1],
        'mask_count': 4, 'mask_size': 3, 'witness_correct_train_count': 2,
        'mask_matching': {'solution_counts': '60% one, 20% two, 20% three, rounded within field/degree strata',
                          'operation_counts': 'within floor/ceil(3*n/5) in each field/degree stratum',
                          'program_support': 'all available pairs represented in strata with n >= candidate programs'},
        'excluded_matched_training_shapes': excluded_shapes,
        'reference_state_count': reference_states, 'reference_manifest_sha256': file_hash(reference / 'manifest.json'),
        'data_code_sha256': file_hash(Path(__file__)), 'datasets': manifests})
    return manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path, help='original outputs/data, including atomic train')
    parser.add_argument('--seed', type=int, default=20260921)
    parser.add_argument('--size', type=int, default=5000)
    parser.add_argument('--eval-size', type=int, default=200)
    parser.add_argument('--panels', type=int, default=50)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    prepare(**vars(args))


if __name__ == '__main__':
    main()
