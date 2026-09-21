import json
from pathlib import Path

import numpy as np
import pytest
import torch

import iclr
import composition_core as core
from iclr.research_data import check_panels, modular_solutions
from iclr.research_objectives import (StateProjection, assignment_cycles, conditional_loss,
                                      entropy_report, interaction, sigreg)
from iclr.common import file_hash, write_json
from iclr.research_analysis import compare


def test_crossed_fixture_offsets_gradient_and_modular_solver():
    fixture = json.loads(Path('plans/source/22sept_iclr_feedback/crossed_panel_fixture.json').read_text())
    a, b = map(tuple, (fixture['program_A'], fixture['program_B']))
    for i, start in enumerate(fixture['starts']):
        for j, target in enumerate(fixture['targets']):
            correct = [p for p in core.enumerate_programs(3) if core.verify_program(start, target, p, 5)]
            assert correct == [a if i == j else b]
            assert list(map(list, core.trajectory(start, correct[0], 5))) == fixture['trajectories'][i][j]
            assert start != target
            assert not any(core.verify_program(start, target, p, 5) for d in (1, 2) for p in core.enumerate_programs(d))
    torch.manual_seed(1)
    scores = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    offsets = (torch.randn(2, 1, 2) + torch.randn(1, 2, 2) + torch.randn(2)
               + torch.randn(2, 2, 1)).double().reshape(4, 2)
    torch.testing.assert_close(interaction(scores + offsets), interaction(scores), atol=1e-6, rtol=0)
    old = float(interaction(scores).detach())
    gradient, = torch.autograd.grad(conditional_loss(scores, 'crossed'), scores)
    assert float(interaction(scores - .1 * gradient)) > old
    matrix = torch.randn(4, 4, dtype=torch.float64)
    torch.testing.assert_close(assignment_cycles(matrix + torch.randn(4, 1) + torch.randn(1, 4)),
                               assignment_cycles(matrix), atol=1e-6, rtol=0)
    a = np.array([[1, 2, 0], [2, 4, 0]])
    origin, basis = modular_solutions(a, [3, 1], 5)
    assert np.array_equal(a @ origin % 5, [3, 1])
    assert not (a @ basis.T % 5).any()
    assert modular_solutions(a, [3, 2], 5) is None


def test_entropy_semantics_boundaries_and_temperature():
    programs = core.enumerate_programs(3)
    scores = np.linspace(-10, 2, len(programs))
    for mask in (np.zeros(125, dtype=bool), np.ones(125, dtype=bool), np.arange(125) % 3 == 0):
        reports = entropy_report(scores, mask, programs, 5, 2)
        assert len({r['first_correct_rank'] for r in reports}) == 1
        assert reports[0]['entropy'] < reports[1]['entropy'] < reports[2]['entropy']
        assert all(r['semantic_classes'] < 125 for r in reports)
    with pytest.raises(ValueError, match='positive'):
        entropy_report(scores, mask, programs, 5, 2, temperatures=[0])


def test_state_projection_has_action_path_and_duplicate_safe_sigreg():
    torch.manual_seed(5)
    head = StateProjection(8, dimension=4)
    hidden = torch.randn(3, 8)
    z, action = head(hidden)
    assert not action.any()
    loss, counts = head.state_loss(z, [[0, 1, 2]] * 3, [5] * 3)
    assert counts['coefficients'] == 9 and torch.isfinite(loss)
    with torch.no_grad():
        head.action.weight.fill_(1.)
    assert head(hidden)[1].abs().sum() > 0
    assert not head(hidden, ablate=True)[1].any()
    keys = [(5, (i, 1, 2)) for i in range(3)]
    value = sigreg(z, keys, 0)
    duplicated = sigreg(z.repeat_interleave(2, 0), [k for k in keys for _ in range(2)], 0)
    torch.testing.assert_close(value, duplicated)
    (loss + value).backward()
    assert torch.isfinite(head.project.weight.grad).all()


def test_comparison_pairs_seeds_and_rejects_domain_or_mask_pooling(tmp_path):
    def runs(name, domain, pair, seed=0):
        data = tmp_path / name
        data.mkdir(exist_ok=True)
        (data / 'tasks.jsonl').write_text('{}\n')
        write_json(data / 'manifest.json', {'domain': domain, 'constraints': {'heldout_motifs': [pair]},
            'files': {'tasks.jsonl': {'sha256': file_hash(data / 'tasks.jsonl')}}})
        result = []
        for arm, accuracy in (('ce', .5), ('ce_cf', .75)):
            out = tmp_path / f'{name}_{arm}_{seed}'
            binding = dict(code_hash='same-code', plan_hash='same-plan', base_hash='same-base',
                data_hash=file_hash(data / 'manifest.json'),
                config=dict(data=str(data), seed=seed, arm=arm, dtype='float32', phase='dev'))
            panels = [dict(panel_id=f'{f}-{i}', policy='local', kind='four_target', family=f, depth=3,
                task_ids=[f'{f}-{i}-{j}' for j in range(4)], starts=[[i, 0]], pair_accuracy=accuracy)
                for f in ('B', 'D') for i in range(2)]
            write_json(out / 'panel_metrics.json', {'binding': binding, 'panels': panels})
            write_json(out / 'DONE', {'binding': binding, 'files': {'panel_metrics.json': file_hash(out / 'panel_metrics.json')}})
            result.append(str(out))
        return result
    first = runs('mask1', 'affine', ['SH1', 'SC2'])
    second = runs('mask2', 'affine', ['REV', 'SC2'])
    out = tmp_path / 'contrast.json'
    compare(first + second, 'ce', 'ce_cf', out)
    report = json.loads(out.read_text())
    assert report['equal_mask_mean'] == .25 and report['distinct_masks'] == 2
    with pytest.raises(ValueError, match='Unpaired'):
        compare(first[:1], 'ce', 'ce_cf', out)
    third = runs('external', 'nonlinear', ['AX1', 'SC2'])
    with pytest.raises(ValueError, match='domains'):
        compare(first + third, 'ce', 'ce_cf', out)
    repeated = runs('same_mask', 'affine', ['SH1', 'SC2'], seed=1)
    # Different manifests of the same mask cannot masquerade as another mask.
    manifest = json.loads((tmp_path / 'same_mask/manifest.json').read_text())
    manifest['generation_seed'] = 2
    write_json(tmp_path / 'same_mask/manifest.json', manifest)
    for path in map(Path, repeated):
        result = json.loads((path / 'panel_metrics.json').read_text())
        result['binding']['data_hash'] = file_hash(tmp_path / 'same_mask/manifest.json')
        write_json(path / 'panel_metrics.json', result)
        write_json(path / 'DONE', {'binding': result['binding'], 'files': {'panel_metrics.json': file_hash(path / 'panel_metrics.json')}})
    with pytest.raises(ValueError, match='same mask'):
        compare(first + repeated, 'ce', 'ce_cf', out)
