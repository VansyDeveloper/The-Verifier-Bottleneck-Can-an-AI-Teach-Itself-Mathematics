"""Independent interpreter audit of every prepared task and every split exclusion."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import composition_core as core
from .calibration import correct_mask
from .common import file_hash, read_jsonl, verify_data, write_json
from .data import _full_states, constraint_hits
from .upgrade_data import reference_registry


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
        for filename, info in manifest['files'].items():
            rows = read_jsonl(directory / filename)
            if len(rows) != info['rows']:
                raise ValueError('Manifest row count changed')
            if filename == 'atomic_train.jsonl':
                if file_hash(directory / filename) != file_hash(reference / filename):
                    raise ValueError('Inherited atomic training tasks changed')
                counts['reused_atomic_train'] += len(rows)
                continue
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
        if name == 'original' or name.startswith('mask'):
            if matched_strata is not None and matched_strata != train_strata:
                raise ValueError('Field, degree or solution counts differ between mask training sets')
            matched_strata = train_strata
            for shape, histogram in operations.items():
                values = [histogram[op] for op in core.OPS]
                if max(values) - min(values) > 1:
                    raise ValueError('Operation marginals are not matched within a training stratum')
        reports[name] = {'counts': counts, 'new_states': len(owners), 'panels': len(panels),
                         'operation_counts_by_shape': {f'{p}:{d}': h for (p, d), h in operations.items()},
                         'training_strata': {':'.join(map(str, k)): v for k, v in train_strata.items()}}
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
