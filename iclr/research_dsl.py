"""Small nonlinear list DSL, explicitly separate from RobustFill/DeepCoder."""

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import random

import composition_core as core
from .common import ROOT, file_hash, write_json
from .data import constraint_hits
from .research_data import check_panels

DOMAIN = 'nonlinear_list_v1'
DEFINITIONS = ('LIST_DSL_V1 (all integers reduced modulo FIELD after each operation):\n'
    '<OP0> adds 1 to every entry; <OP1> squares the first entry; <OP2> reverses the list; '
    '<OP3> sorts entries in ascending numeric order; <OP4> rotates the list one place left.\n')


def trajectory(row, program):
    if not program:
        return (tuple(row['start']),)
    if row.get('domain') != DOMAIN:
        return core.trajectory(row['start'], program, row['p'])
    states, p = [tuple(row['start'])], row['p']
    for op in program:
        state = states[-1]
        if op == 'SH1':
            state = tuple((x + 1) % p for x in state)
        elif op == 'SC2':
            state = (state[0] ** 2 % p, *state[1:])
        elif op == 'REV':
            state = tuple(reversed(state))
        elif op == 'AC1':
            state = tuple(sorted(state))
        elif op == 'AX1':
            state = (*state[1:], state[0])
        else:
            raise ValueError('Unknown list operation')
        states.append(state)
    return tuple(states)


def verify(row, program):
    return trajectory(row, program)[-1] == tuple(row['target'])


def plan_prompt(row):
    prompt = core.plan_prompt(row)
    if row.get('domain') == DOMAIN:
        prompt = DEFINITIONS + prompt.replace(f"DEGREE_CAP: {len(row['start']) - 1}", f"LENGTH: {len(row['start'])}")
    return prompt


def apply_prompt(row):
    return (DEFINITIONS if row.get('domain') == DOMAIN else '') + core.apply_prompt(row)


def prompt_prefix(row, prefix):
    return plan_prompt(row) + (' ' + core.program_answer(prefix) if prefix else '')


def prepare(out, train_panels=64, eval_panels=16, ordinary=32, seed=20260924, smoke=False):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Nonlinear DSL inputs are frozen; choose a new directory')
    if min(train_panels, eval_panels, ordinary) < 1:
        raise ValueError('Positive dataset sizes required')
    if smoke:
        train_panels, eval_panels, ordinary = 2, 1, 2
    rng, used, task_ids, files, reports = random.Random(seed), set(), set(), {}, {}
    directory = out / 'listdsl'
    directory.mkdir(parents=True)
    spec = {'heldout_motifs': [['SC2', 'REV'], ['SC2', 'SC2'], ['SH1', 'SC2']]}
    def generate(split, family, count, depth=3, panel=False):
        rows, attempts = [], 0
        while len(rows) < count:
            attempts += 1
            if attempts > 20000:
                raise ValueError(f'Insufficient disjoint nonlinear DSL support: {split} {len(rows)}/{count}')
            p = rng.choice((17, 19) if family in ('C', 'D') else (7, 11, 13))
            start = [rng.randrange(p) for _ in range(4)]
            template = {**spec, 'domain': DOMAIN, 'p': p, 'degree': 3, 'depth': depth, 'start': start}
            grouped = defaultdict(list)
            for program in core.enumerate_programs(depth):
                grouped[trajectory(template, program)[-1]].append(program)
            shorter = {tuple(start)}
            for d in range(1, depth):
                shorter.update(trajectory(template, program)[-1] for program in core.enumerate_programs(d))
            candidates = [(target, programs) for target, programs in grouped.items()
                if target not in shorter and (not panel or len(programs) == 1)
                and (family == 'ATOMIC' or {constraint_hits(p, spec) for p in programs} == {int(family in ('B', 'D'))})]
            rng.shuffle(candidates)
            selected, states = [], set()
            for target, programs in candidates:
                fingerprint = hashlib.sha256(json.dumps([DOMAIN, p, start, target, depth]).encode()).hexdigest()
                trajectories = {(p, s) for program in programs for s in trajectory(template, program)}
                if trajectories & used or fingerprint in task_ids:
                    continue
                selected.append({**template, 'target': list(target), 'correct_programs': list(map(list, programs)),
                    'witness': list(programs[0]), 'correct_count': len(programs), 'shortest_depth': depth,
                    'task_fingerprint': fingerprint, 'task_id': 'list-' + fingerprint[:24], 'split': split, 'family': family})
                states.update(trajectories)
                if len(selected) == (4 if panel else 1):
                    break
            if len(selected) != (4 if panel else 1):
                continue
            if panel:
                panel_id = hashlib.sha256(json.dumps([DOMAIN, split, p, start]).encode()).hexdigest()[:24]
                for row in selected:
                    row['panel_id'] = panel_id
            rows.extend(selected)
            used.update(states); task_ids.update(r['task_fingerprint'] for r in selected)
        if panel:
            check_panels(rows)
        # An independent direct call verifies every stored label at all lengths.
        for row in rows:
            exact = [list(p) for p in core.enumerate_programs(depth) if verify(row, p)]
            if exact != row['correct_programs']:
                raise ValueError('List DSL label audit failed')
        path = directory / (split + '.jsonl')
        path.write_bytes(core.canonical_jsonl_bytes(rows))
        files[path.name] = {'rows': len(rows), 'sha256': file_hash(path)}
        reports[split] = {'tasks': len(rows), 'attempts': attempts,
                         'program_support': dict(Counter(' '.join(p) for r in rows for p in r['correct_programs']))}
        print(f'{DOMAIN} {split}: {len(rows)} tasks', flush=True)
    generate('train_four', 'TRAIN', 4 * train_panels, panel=True)
    # Reward uses the same task identities as the panel arms, not a different pool.
    (directory / 'train.jsonl').write_bytes((directory / 'train_four.jsonl').read_bytes())
    files['train.jsonl'] = dict(files['train_four.jsonl'])
    for phase in ('dev', 'final'):
        for family in 'ABCD':
            generate(f'{phase}_{family}', family, ordinary)
            generate(f'{phase}4_{family}', family, ordinary, depth=4)
            generate(f'{phase}_panel_{family}', family, 4 * eval_panels, panel=True)
        generate(f'{phase}_atomic', 'ATOMIC', 10 if smoke else 100, depth=1)
    write_json(directory / 'manifest.json', {'schema': 'iclr.research.data.v3', 'domain': DOMAIN,
        'files': files, 'status': 'smoke' if smoke else 'frozen', 'constraints': spec,
        'training_panels': 'train_four.jsonl', 'config': {'seed': seed, 'train_panels': train_panels},
        'definitions': DEFINITIONS, 'max_training_depth': 3, 'evaluation_depths': [3, 4],
        'scope': 'new finite nonlinear list DSL; not a full RobustFill, DeepCoder or ExeDec benchmark'})
    write_json(out / 'audit.json', {'status': 'PASS', 'smoke': smoke, 'domain': DOMAIN,
        'scope': 'complete solution sets, minimum lengths, masks, task/state exclusions, unique four-TARGET panels',
        'splits': reports, 'states': len(used), 'tasks': len(task_ids),
        'aliases': 'train.jsonl equals train_four.jsonl intentionally; it is not an additional split'})
    write_json(out / 'protocol.json', {'schema': 'iclr.research.protocol.v3', 'smoke': smoke, 'domain': DOMAIN,
        'reference_manifest_sha256': None, 'seed': seed,
        'datasets': {'listdsl': {'manifest_sha256': file_hash(directory / 'manifest.json'), 'constraints': spec}},
        'audit_sha256': file_hash(out / 'audit.json'), 'analysis_plan_sha256': file_hash(ROOT / 'plans/research_v3.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--train-panels', type=int, default=64)
    parser.add_argument('--eval-panels', type=int, default=16)
    parser.add_argument('--ordinary', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20260924)
    parser.add_argument('--smoke', action='store_true')
    prepare(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
