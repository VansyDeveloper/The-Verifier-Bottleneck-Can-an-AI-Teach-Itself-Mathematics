"""Independent interpreter audit of every prepared task and every split exclusion."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import composition_core as core
from .calibration import correct_mask
from .common import file_hash, read_jsonl, verify_data, write_json
from .data import _full_states, assign_witnesses, constraint_hits, mixture
from .upgrade_data import reference_registry, signatures


def distribution(programs):
    programs = list(map(tuple, programs))
    counts = Counter(programs)
    pairs = Counter(pair for p in programs for pair in zip(p, p[1:]))
    return {'exposures': len(programs), 'program_union': len(counts),
            'program_entropy': -sum(n / len(programs) * math.log(n / len(programs)) for n in counts.values()),
            'program_counts': {' '.join(p): n for p, n in sorted(counts.items())},
            'unigrams': Counter(op for p in programs for op in p),
            'positions': {str(i): Counter(p[i] for p in programs) for i in range(3)},
            'pairs': {'>'.join(p): n for p, n in sorted(pairs.items())},
            'pair_count_square': sum(n * n for n in pairs.values())}


def witness_preflight(rows, smoke):
    sets = Counter(tuple(sorted(map(tuple, r['correct_programs']))) for r in rows)
    unavoidable = sum(all(p[0] == 'REV' for p in r['correct_programs']) for r in rows) / len(rows)
    families = set()
    for row in rows:
        maps = signatures(row['p'], row['degree'], 3)
        grouped = defaultdict(list)
        for program in row['correct_programs']:
            grouped[maps[tuple(program)]].append(program)
        expected = list(grouped.values())
        if sorted(expected) != sorted(row['equivalence_groups']):
            raise ValueError('Incorrect global-affine equivalence labels')
        if row['equivalence_kind'] != ('global_affine' if len(expected) == 1 else 'start_specific_collision'):
            raise ValueError('Incorrect start-specific coincidence label')
        families.update(tuple(sorted(map(tuple, group))) for group in expected)
    examples = mixture(rows, [], len(rows), 0., 0, 'program_only')
    arms = {policy: distribution(item['row']['witness'] for epoch in range(2)
                for item in assign_witnesses(examples, policy, epoch)) for policy in ('fixed', 'balanced')}
    checks = {'at_least_ten_semantic_families': len(families) >= 10,
              'largest_correct_set_at_most_20_percent': max(sets.values()) / len(rows) <= .2,
              'unavoidable_REV_at_most_half': unavoidable <= .5,
              'balanced_reduces_pair_square': arms['balanced']['pair_count_square'] < arms['fixed']['pair_count_square'],
              'actual_labels_changed': arms['balanced']['program_counts'] != arms['fixed']['program_counts']}
    if not smoke and not all(checks.values()):
        raise ValueError(f'Witness manipulation failed its predeclared CPU checks: {checks}')
    return {'status': 'smoke' if smoke else 'PASS', 'checks': checks, 'semantic_families': len(families),
            'largest_correct_set_share': max(sets.values()) / len(rows), 'unavoidable_first_REV_share': unavoidable,
            'equivalence_kinds': Counter(r['equivalence_kind'] for r in rows), 'epochs': 2, 'seed': 0,
            'arms': arms, 'interpretation': 'pair imbalance intervention, not maximum program diversity'}


def audit(data, reference, out):
    data, reference, out = Path(data), Path(reference), Path(out)
    protocol = json.loads((data / 'protocol.json').read_text())
    if file_hash(reference / 'manifest.json') != protocol['reference_manifest_sha256']:
        raise ValueError('Different inherited dataset')
    inherited = reference_registry(reference)
    reports, matched_strata = {}, None
    for name, entry in protocol['datasets'].items():
        directory = data / name
        if file_hash(directory / 'manifest.json') != entry['manifest_sha256']:
            raise ValueError('Dataset changed after the protocol was frozen')
        manifest = verify_data(directory)
        owners, task_ids, panels = {}, set(), defaultdict(list)
        counts, operations, train_strata = Counter(), defaultdict(Counter), Counter()
        train_rows, support = [], {}
        for filename, info in manifest['files'].items():
            rows = read_jsonl(directory / filename)
            if len(rows) != info['rows']:
                raise ValueError('Manifest row count changed')
            if filename == 'atomic_train.jsonl':
                if file_hash(directory / filename) != file_hash(reference / filename):
                    raise ValueError('Inherited atomic training tasks changed')
                counts['reused_atomic_train'] += len(rows)
                continue
            if filename == 'train.jsonl':
                train_rows = rows
            if rows and rows[0]['depth'] == 3:
                support[filename] = {'tasks': len(rows), 'correct_program_union': len({tuple(p) for r in rows for p in r['correct_programs']}),
                    'unique_solutions': sum(r['correct_count'] == 1 for r in rows)}
            for row in rows:
                p, depth = row['p'], row['depth']
                if row['task_id'] in task_ids:
                    raise ValueError('Task repeated across generated splits')
                task_ids.add(row['task_id'])
                expected_fp = core.canonical_task_fingerprint(p, row['start'], row['target'], depth)
                if expected_fp != row['task_fingerprint'] or expected_fp in inherited.tasks:
                    raise ValueError('Wrong task fingerprint or overlap with inherited tasks')
                if row['task_id'] != 's4c-' + expected_fp[:24]:
                    raise ValueError('Task ID differs from the exact identity')
                programs = sorted(core.enumerate_programs(depth))
                mask = correct_mask(p, tuple(row['start']), tuple(row['target']), depth)
                correct = [prog for prog, yes in zip(programs, mask) if yes]
                if not correct or row['correct_count'] != len(correct):
                    raise ValueError('Incorrect solution count')
                if 'correct_programs' in row and sorted(map(tuple, row['correct_programs'])) != correct:
                    raise ValueError('Incomplete correct set')
                if tuple(row['witness']) not in correct:
                    raise ValueError('Incorrect witness')
                if row['start'] == row['target'] or any(core.verify_program(row['start'], row['target'], prog, p)
                    for d in range(1, depth) for prog in core.enumerate_programs(d)):
                    raise ValueError('Claimed minimum depth is incorrect')
                if row['family'] != 'ATOMIC':
                    wanted = 1 if row['family'] in ('B', 'D') else 0
                    if {constraint_hits(prog, entry['constraints']) for prog in correct} != {wanted}:
                        raise ValueError('Not all correct programs satisfy the heldout semantics')
                    expected_fields = core.TRANSFER_FIELDS if row['family'] in ('C', 'D') else core.KNOWN_FIELDS
                    if p not in expected_fields:
                        raise ValueError('Wrong field family')
                states = _full_states(row, correct)
                if states & inherited.states:
                    raise ValueError('Generated task shares a state with inherited data')
                owner = row.get('panel_id', row['task_id'])
                if any(fp in owners and owners[fp] != owner for fp in states):
                    raise ValueError('Shared state outside a matched-START panel')
                owners.update(dict.fromkeys(states, owner))
                if row.get('panel_id'):
                    panels[row['panel_id']].append(row)
                if row['family'] == 'TRAIN':
                    shape = p, len(row['start']) - 1
                    operations[shape].update(row['witness'])
                    train_strata[(*shape, len(correct))] += 1
                counts[filename] += 1
                counts['programs_checked'] += len(programs)
        for rows in panels.values():
            if len(rows) != 4 or len({(r['p'], tuple(r['start']), r['depth']) for r in rows}) != 1 or len({tuple(r['target']) for r in rows}) != 4:
                raise ValueError('Invalid counterfactual panel')
            if protocol['schema'].endswith('.v2') and any(r['correct_count'] != 1 for r in rows):
                raise ValueError('v2 target alignment panels require unique solutions')
        if name == 'original' or name.startswith('mask'):
            if matched_strata is not None and matched_strata != train_strata:
                raise ValueError('Field, degree or solution counts differ between mask training sets')
            matched_strata = train_strata
            for shape, histogram in operations.items():
                values = [histogram[op] for op in core.OPS]
                if max(values) - min(values) > 1:
                    raise ValueError('Operation marginals are not matched within a training stratum')
        reports[name] = {'counts': counts, 'new_states': len(owners), 'panels': len(panels),
                         'training_distribution': distribution(r['witness'] for r in train_rows),
                         'solution_support': support,
                         'operation_counts_by_shape': {f'{p}:{d}': h for (p, d), h in operations.items()},
                         'training_strata': {':'.join(map(str, k)): v for k, v in train_strata.items()}}
        if name == 'witness' and protocol['schema'].endswith('.v2'):
            reports[name]['manipulation_check'] = witness_preflight(train_rows, protocol['smoke'])
        print(f'Audited {name}: {len(task_ids)} new tasks', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    report = {'status': 'PASS', 'protocol_sha256': file_hash(data / 'protocol.json'),
              'reference_manifest_sha256': file_hash(reference / 'manifest.json'), 'datasets': reports,
              'scope': 'full interpreter enumeration, minimum depths, all-correct-set states, inherited/split exclusions, matched training strata and operation counts',
              'smoke': protocol['smoke']}
    write_json(out / 'audit.json', report)
    write_json(out / 'DONE', {'files': {'audit.json': file_hash(out / 'audit.json')}})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    audit(**vars(args))


if __name__ == '__main__':
    main()
