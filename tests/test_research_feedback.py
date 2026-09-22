"""Regression checks for the independent 22 September second review."""

import argparse
import json
from pathlib import Path

import pytest
import torch

from iclr.common import file_hash, write_json
from iclr.research import plan
from iclr.research_data import program_exposure
from iclr.research_gradient import mc_estimate, mc_status, vector_gradient
from iclr.research_io import exposure_ledger
from iclr.reward_control import exact_reward_loss


def test_rare_reward_is_inconclusive_with_a_bounded_diagnostic_budget(monkeypatch):
    logits = torch.tensor([-13.815509557963773, 0.], dtype=torch.float64, requires_grad=True)
    logq, correct = logits.log_softmax(0), torch.tensor([True, False])
    exact = vector_gradient(exact_reward_loss(logq, correct, 32), [logits], retain=True)
    assert float(exact.norm()) > 1e-7
    # The reported counterexample is the possible event that all draws are incorrect.
    monkeypatch.setattr(torch, 'multinomial', lambda q, count, replacement: torch.ones(count, dtype=torch.long))
    result = mc_estimate(logq, correct, [logits], exact,
                         dict(mc_groups=128, mc_max_groups=2048, mc_blocks=8, group_size=32))
    assert result['status'] == 'mc_inconclusive_rare_reward'
    assert [a['groups'] for a in result['attempts']] == [128, 512, 2048]
    assert result['gradient'].norm() == 0 and result['standard_error'] == 0
    assert mc_status(.1, 20, 8, .1, .1) == 'mc_ok'
    assert mc_status(.1, 20, 8, 10., .1) == 'mc_check_failed'
    assert mc_status(0., 0, 0, 0., 0.) == 'mc_degenerate_zero_signal'


def test_exposure_distinguishes_historical_mask_from_continuation_mask():
    stage = {'stage': 'historical_sft', 'provenance_status': 'verified', **program_exposure([['SC2', 'REV', 'SC2', 'SC2']])}
    new = exposure_ledger({'constraints': {'heldout_motifs': [['SC2', 'REV']]}}, [stage])
    original = exposure_ledger({'constraints': {'heldout_motifs': [['AX1', 'SH1']]}}, [stage])
    assert not new['recorded_history_excludes_mask'] and new['historical_depth4']
    assert original['recorded_history_excludes_mask']
    unknown = exposure_ledger({'constraints': {'heldout_motifs': []}}, [{'pairs': None}])
    assert not unknown['recorded_history_excludes_mask']
    declared = exposure_ledger({'constraints': {'heldout_motifs': []}}, [{'pairs': {}, 'depths': {}, 'provenance_status': 'declared_only'}])
    assert declared['provenance_status'] == 'declared_only' and not declared['recorded_history_excludes_mask']
    assert declared['depth_history_status'] == 'unknown'
    incomplete_depth = exposure_ledger({'constraints': {'heldout_motifs': []}},
        [{'pairs': {}, 'depths': {'4': None}, 'provenance_status': 'verified'}])
    assert incomplete_depth['depth_history_status'] == 'unknown'


def test_reward_defaults_to_original_and_rejects_unmatched_mask(tmp_path):
    inputs = tmp_path / 'inputs'
    datasets = {}
    for name in ('original', 'mask1', 'mask2'):
        directory = inputs / name
        directory.mkdir(parents=True)
        (directory / 'tasks.jsonl').write_text('{}\n')
        write_json(directory / 'manifest.json', {'schema': 'iclr.research.data.v4',
            'constraints': {'heldout_motifs': [['SC2', 'REV']]},
            'files': {'tasks.jsonl': {'sha256': file_hash(directory / 'tasks.jsonl')}}})
        datasets[name] = {'manifest_sha256': file_hash(directory / 'manifest.json')}
    write_json(inputs / 'audit.json', {'status': 'PASS'})
    history = json.loads(Path('evidence/upgrade/previous_q06.json').read_text())
    write_json(inputs / 'protocol.json', {'schema': 'iclr.research.protocol.v4', 'smoke': False,
        'analysis_plan_sha256': file_hash('plans/research_v4.json'), 'audit_sha256': file_hash(inputs / 'audit.json'),
        'reference_manifest_sha256': history['runs'][0]['data_hash'], 'datasets': datasets})
    args = argparse.Namespace(inputs=str(inputs), smoke=False, queue_name='Q2', stage='screen', phase='dev',
        analysis_lock=None, trained_from=None, selection=None, gate=None, mask=None,
        models='plans/upgrade_models.json', model_names=['q06'], device='cuda', dtype='bfloat16',
        prefix_batch=16, out=str(tmp_path / 'default'))
    plan(args)
    queue = json.loads((tmp_path / 'default/queue.json').read_text())
    configs = [json.loads(Path(j['config_path']).read_text()) for j in queue['jobs'] if 'config_path' in j]
    assert len(configs) == 1 and Path(configs[0]['data']).name == 'original'
    args.mask, args.out = 'mask1', str(tmp_path / 'mismatched')
    with pytest.raises(ValueError, match='matching-mask SFT'):
        plan(args)
    args.prepare_matching_sft, args.out = True, str(tmp_path / 'prepare_matching')
    plan(args)
    queue = json.loads((tmp_path / 'prepare_matching/queue.json').read_text())
    trained = [j for j in queue['jobs'] if j['kind'] == 'train']
    assert len(trained) == 1
    init = json.loads(Path(trained[0]['config_path']).read_text())
    assert init['adapter'] is None and init['objective'] == 'ce'
    diagnostic = next(j for j in queue['jobs'] if j['id'].endswith('_diagnostic'))
    assert diagnostic['depends_on'] == [trained[0]['id']]
