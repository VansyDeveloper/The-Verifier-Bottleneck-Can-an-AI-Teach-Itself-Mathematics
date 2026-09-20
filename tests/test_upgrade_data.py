from collections import Counter
import itertools
import random

import numpy as np

from iclr.data import assign_witnesses, constraint_hits, mixture
from iclr.upgrade_data import affine_maps, feasibility, generate_rows, matched_program_schedule
import composition_core as core


def test_exact_masks_matched_targets_and_witness_intervention():
    # Independent interpreter agreement across every program, all shapes and depths.
    rng = random.Random(3)
    for p, degree in itertools.product((*core.KNOWN_FIELDS, *core.TRANSFER_FIELDS), core.DEGREES):
        start = [rng.randrange(p) for _ in range(degree + 1)]
        for depth in (1, 3, 4):
            programs, linear, offsets = affine_maps(p, degree, depth)
            calculated = (np.einsum('pij,j->pi', linear, start) + offsets) % p
            expected = [core.trajectory(start, program, p)[-1] for program in programs]
            np.testing.assert_array_equal(calculated, expected)
    impossible = {pair for pair in itertools.product(core.OPS, repeat=2)
                  if not any(r['strict_heldout_classes'] for r in feasibility({'heldout_motifs': [pair]}))}
    assert impossible == {('REV', 'REV'), *[(a, b) for a, b in itertools.permutations(('AC1', 'AX1', 'SC2', 'SH1'), 2) if 'AC1' in (a, b)]}
    registry = core.FingerprintRegistry()
    registry.shape_counts = Counter()
    spec = {'heldout_motifs': [list(p) for p in core.HELDOUT_MOTIFS]}
    matched = matched_program_schedule(spec, 29, 4, 200, 3)
    op_counts = Counter(op for program, _ in matched for op in program)
    assert set(op_counts.values()) == {120}
    assert Counter(n for _, n in matched) == {1: 120, 2: 40, 3: 40}
    train = generate_rows('train', 20, spec, registry, 42, family='TRAIN', correct_count=2)
    train_states = registry.states.copy()
    rows = generate_rows('panel_B', 8, spec, registry, 44, family='B', panel_targets=4)
    assert len({r['panel_id'] for r in rows}) == 2
    for panel in {r['panel_id'] for r in rows}:
        group = [r for r in rows if r['panel_id'] == panel]
        assert len({(r['p'], tuple(r['start'])) for r in group}) == 1
        assert len({tuple(r['target']) for r in group}) == 4
    for row in rows:
        assert all(constraint_hits(p, row) == 1 for p in row['correct_programs'])
        for d in (1, 2):
            assert not any(core.verify_program(row['start'], row['target'], p, row['p']) for p in core.enumerate_programs(d))
        assert not any(core.state_fingerprint(row['p'], state) in train_states for program in row['correct_programs']
                       for state in core.trajectory(row['start'], program, row['p']))
    examples = mixture(train, [], 60, 0, 51, 'program_only')
    answers = []
    for policy in ('fixed', 'uniform', 'balanced'):
        selected = assign_witnesses(examples, policy, 5)
        assert [r['task_id'] for r in selected] == [r['task_id'] for r in examples]
        assert [r['prompt'] for r in selected] == [r['prompt'] for r in examples]
        for item in selected:
            row = item['row']
            assert core.verify_program(row['start'], row['target'], row['witness'], row['p'])
            assert constraint_hits(row['witness'], row) == 0
        answers.append([r['answer'] for r in selected])
    assert answers[0] != answers[1] and answers[1] != answers[2]
