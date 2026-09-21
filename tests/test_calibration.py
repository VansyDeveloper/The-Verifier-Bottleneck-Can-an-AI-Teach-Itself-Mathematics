import copy

import numpy as np
import pytest

from iclr.calibration import bias_vector, calibrated_rows, design, features, fit_bias, score_metrics, tie_hit
from iclr.upgrade_analysis import measure
import composition_core as core


def panel():
    programs = sorted(core.enumerate_programs(3))
    rng = np.random.default_rng(71)
    weights = rng.normal(size=31)
    prior = features(programs) @ weights
    rows = []
    for i in range(10):
        start = [i + 1, 3, 5]
        target = list(core.trajectory(start, programs[i], 29)[-1])
        rows.append({'task_id': str(i), 'family': 'A', 'split': 'calibration_A',
                     'depth': 3, 'p': 29, 'start': start, 'target': target,
                     'ranking': [{'program': list(p), 'score': float(s - i),
                                  'correct': core.verify_program(start, target, p, 29)}
                                 for p, s in zip(programs, prior)]})
    return rows, weights


def test_calibration_is_A_only_transfers_structure_and_excludes_self():
    rows, weights = panel()
    fit = fit_bias(rows)
    # Known pair bias recovered modulo a program-independent offset, including d4.
    for depth in (3, 4):
        true = features(sorted(core.enumerate_programs(depth))) @ weights
        estimated = bias_vector(fit, 'bigram', depth)
        np.testing.assert_allclose(estimated - estimated.mean(), true - true.mean(), atol=1e-12)
    with pytest.raises(ValueError, match='cannot transfer'):
        bias_vector(fit, 'full', 4)
    with pytest.raises(ValueError, match='only A'):
        fit_bias([{**rows[0], 'family': 'B'}])
    with pytest.raises(ValueError, match='overlap'):
        calibrated_rows(rows, rows)
    _, folds = calibrated_rows(rows, rows, crossfit=True)
    for i, tid in enumerate(sorted(r['task_id'] for r in rows)):
        assert tid not in folds['crossfit'][i % 5]['task_ids']
    malformed = copy.deepcopy(rows)
    malformed[0]['ranking'][0]['correct'] = not malformed[0]['ranking'][0]['correct']
    with pytest.raises(ValueError, match='solution labels'):
        fit_bias(malformed)
    # IID success depends on correct mass, not on how it is split across witnesses.
    left = score_metrics(np.log([.1, .3, .6]), np.array([True, True, False]))
    right = score_metrics(np.log([.2, .2, .6]), np.array([True, True, False]))
    assert left['iid_pass@32'] == right['iid_pass@32']
    assert left['correct_conditional_entropy'] != right['correct_conditional_entropy']


def test_static_controls_alignment_invariance_and_positional_confound():
    import itertools
    rng = np.random.default_rng(5)
    matrix = rng.normal(size=(4, 4))
    a = measure(matrix)
    b = measure(matrix + rng.normal(size=(4, 1)) * 100 + rng.normal(size=(1, 4)) * 100)
    np.testing.assert_allclose(a['cycle_contrasts'], b['cycle_contrasts'], atol=1e-12)
    assert a['assignment_credit'] == b['assignment_credit']
    assert a['pair_accuracy_tie_half'] == b['pair_accuracy_tie_half']
    blind = measure(np.broadcast_to(np.arange(4.), (4, 4)))
    assert blind['assignment_credit'] == 1 / 24 and blind['pair_accuracy_tie_half'] == .5
    # Exact tie expectation vs exhaustive permutations, including multiple solutions.
    scores, mask = np.array([2., 1., 1., 1., 0.]), np.array([False, True, True, False, True])
    metrics = score_metrics(scores, mask, random_ties=True)
    ranks = [1 + next(i for i, yes in enumerate(mask[[0, *perm, 4]]) if yes)
             for perm in itertools.permutations([1, 2, 3])]
    assert np.isclose(metrics['mrr'], np.mean([1 / rank for rank in ranks]))
    assert tie_hit(0, 46, 1, 32) == 32 / 46
    for k in range(1, 6):
        assert np.isclose(tie_hit(metrics['tie_above'], metrics['tie_size'], metrics['tie_correct'], k), np.mean([rank <= k for rank in ranks]))
    programs = sorted(core.enumerate_programs(3))
    positional = design(programs, 'positional') @ rng.normal(size=16)
    pooled = features(programs, False)
    assert np.linalg.norm(positional - pooled @ np.linalg.lstsq(pooled, positional, rcond=None)[0]) > 1
    rows, _ = panel()
    fit = fit_bias(rows)
    with pytest.raises(ValueError, match='cannot transfer'):
        bias_vector(fit, 'positional', 4)
