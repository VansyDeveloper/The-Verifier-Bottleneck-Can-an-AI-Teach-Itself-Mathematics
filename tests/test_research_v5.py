"""Feedback regressions: actual trajectory/resume, replay exclusions and explicit plans."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

import iclr
import composition_core as core
from iclr.common import code_hash, file_hash, read_jsonl, verify_receipt, write_json
from iclr.research import plan
from iclr.research_data import prepare_amendment, verify_amendment
from iclr.research_io import plan_path, start_run
from iclr.research_model import evaluation_context, rng_state, restore_rng
from iclr.research_train import LOG_NAMES, run, snapshot_logs
from test_research_runtime import tiny_setup


def test_replay_monitor_and_resume_preserve_actual_optimizer_trajectory(tmp_path, monkeypatch):
    data, base = tiny_setup(tmp_path)
    amendment = tmp_path / 'amendment/manifest.json'
    prepare_amendment(data, amendment.parent, per_stratum=1)
    rows = read_jsonl(amendment.parent / 'atomic_replay.jsonl')
    assert Counter((r['witness'][0], r['p']) for r in rows) == Counter({(op,p): 1 for op in core.OPS for p in core.KNOWN_FIELDS})
    assert all(r['split'] == 'train_replay' for r in rows)
    device = os.environ.get('ICLR_SMOKE_DEVICE', 'cpu')
    cfg = dict(base=str(base), data=str(data), device=device, dtype='bfloat16' if device == 'cuda' else 'float32',
               prefix_batch=8, epochs=3, max_steps=3, checkpoint_steps=[0,1,2,3],
               objective='ce_cf_entropy', plan='v5', amendment=str(amendment), replay_weight=.25)
    for name, monitor, interrupted in [('continuous', False, False), ('monitored', True, False),
                                        ('resumed', True, True), ('recovered', True, False)]:
        out = tmp_path / name
        config = {**cfg, 'output': str(out), 'monitor': monitor, 'resume_from': 'latest'}
        if name == 'recovered':
            rename = Path.rename
            def interrupt_publication(path, target):
                if path.name == 'step_000002.partial':
                    raise RuntimeError('Injected interruption before checkpoint publication')
                return rename(path, target)
            with monkeypatch.context() as patch:
                patch.setattr(Path, 'rename', interrupt_publication)
                with pytest.raises(RuntimeError, match='Injected interruption'):
                    run(config)
            assert not (out / 'DONE').exists()
            partial = out / 'checkpoints/step_000002.partial'
            assert (partial / 'DONE').exists()
            for log in LOG_NAMES:
                with (out / log).open('ab') as stream:
                    stream.write(b'{"step": 3, "unfinished":')
            with pytest.raises(ValueError, match='published checkpoint'):
                run(config, resume_from=str(partial))
            # The generated config and explicit CLI argument both request latest.
            run(config, resume_from='latest')
            assert not partial.exists()
        else:
            run(config, stop_after=1 if interrupted else None)
        if interrupted:
            assert not (out / 'DONE').exists()
            run({**config, 'resume_from': 'must-be-overridden'}, resume_from=str(out / 'checkpoints/step_000001'))
        verify_receipt(out)
        # Completed replay is idempotent, not a second training run.
        run(config)
        budget = json.loads((out / 'budget.json').read_text())
        assert budget['optimizer_steps'] == 3 and budget['composition_examples'] == 12 and budget['replay_examples'] == 6
        assert budget['supervised_operation_tokens'] == 36 and budget['replay_plan_tokens'] == 6
        replay = read_jsonl(out / 'replay_stream.jsonl')
        assert budget['replay_loss_tokens'] == sum(r['target_tokens'] for r in replay)
        for metric in read_jsonl(out / 'training_metrics.jsonl'):
            assert metric['ce_full_vocab'] == pytest.approx(metric['ce_local_actions'] + metric['ce_legal_gate'], abs=1e-6)
            assert metric['actual_parameter_update_norm'] > 0
        for step in range(4):
            cp = out / 'checkpoints' / f'step_{step:06d}'
            receipt = verify_receipt(cp)
            assert receipt['checkpoint_step'] == step and not receipt['completed_training']
            assert file_hash(cp / 'resume_state.pt') == receipt['resume_state_hash']
        state = torch.load(cp / 'resume_state.pt', weights_only=False, map_location='cpu')
        assert state['logs'] == snapshot_logs(out)
    expected = load_file(tmp_path / 'continuous/adapter/adapter_model.safetensors')
    expected_state = torch.load(tmp_path / 'continuous/checkpoints/step_000003/resume_state.pt', weights_only=False, map_location='cpu')
    for name in ('monitored', 'resumed', 'recovered'):
        actual = load_file(tmp_path / name / 'adapter/adapter_model.safetensors')
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], atol=1e-6 if device == 'cuda' else 0, rtol=0)
        for log in ('training_stream.jsonl', 'replay_stream.jsonl'):
            assert (tmp_path / 'continuous' / log).read_bytes() == (tmp_path / name / log).read_bytes()
        state = torch.load(tmp_path / name / 'checkpoints/step_000003/resume_state.pt', weights_only=False, map_location='cpu')
        assert state['next_step'] == expected_state['next_step'] and state['counts'] == expected_state['counts']
        torch.testing.assert_close(state['optimizer'], expected_state['optimizer'], atol=1e-6 if device == 'cuda' else 0, rtol=0)
    # A final receipt must not be substituted for a step-1 adapter.
    with pytest.raises(ValueError, match='training receipt'):
        start_run({**cfg, 'output': str(tmp_path / 'wrong_receipt'), 'arm': cfg['objective'], 'seed': 0,
            'adapter': str(tmp_path / 'continuous/checkpoints/step_000001/adapter'),
            'training_receipt': str(tmp_path / 'continuous')}, 'evaluate')
    # Weight zero is the original objective and does not consume the replay stream.
    for name, extra in [('legacy', {}), ('zero', {'amendment': str(amendment), 'replay_weight': 0.})]:
        run({k: v for k, v in {**cfg, 'epochs': 1, 'max_steps': 1, 'checkpoint_steps': [],
            'replay_weight': 0., 'output': str(tmp_path / name), **extra}.items() if k != 'amendment' or name == 'zero'})
    zero = load_file(tmp_path / 'zero/adapter/adapter_model.safetensors')
    for key, value in load_file(tmp_path / 'legacy/adapter/adapter_model.safetensors').items():
        torch.testing.assert_close(value, zero[key], atol=0, rtol=0)


def test_evaluation_restores_rng_modes_and_replay_rejects_changed_inputs(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(2,2), torch.nn.Dropout(.2))
    model.train(); model[0].eval()
    state = rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(1))
    restore_rng(state)
    with evaluation_context(model):
        random.random(); np.random.rand(7); torch.rand(8)
        assert not model.training
    actual = (random.random(), np.random.rand(), torch.rand(1))
    assert actual[0:2] == expected[0:2]
    assert torch.equal(actual[2], expected[2]) and model.training and not model[0].training and model[1].p == .2
    data, _ = tiny_setup(tmp_path)
    path = tmp_path / 'amendment/manifest.json'
    prepare_amendment(data, path.parent, per_stratum=1)
    amended = json.loads(path.read_text())
    # A maliciously relabelled dev row with a refreshed file hash still fails the semantic exclusion audit.
    rows = read_jsonl(path.parent / 'atomic_replay.jsonl')
    dev = read_jsonl(data / 'dev_atomic.jsonl')[0]
    rows[0] = {**dev, 'split': 'train_replay', 'task_id': 'replay-' + dev['task_fingerprint'][:24]}
    (path.parent / 'atomic_replay.jsonl').write_bytes(core.canonical_jsonl_bytes(rows))
    amended['files']['atomic_replay.jsonl']['sha256'] = file_hash(path.parent / 'atomic_replay.jsonl')
    write_json(path, amended)
    with pytest.raises(ValueError, match='overlaps'):
        verify_amendment(path, data, file_hash(plan_path('v5')))


def test_stability_queue_is_bounded_and_matched_needs_admission(tmp_path):
    data, base = tiny_setup(tmp_path)
    inputs = tmp_path / 'inputs'
    import shutil
    shutil.copytree(data, inputs / 'mask1')
    write_json(inputs / 'audit.json', {'status': 'PASS'})
    write_json(inputs / 'protocol.json', {'schema': 'iclr.research.protocol.v4', 'smoke': True,
        'analysis_plan_sha256': file_hash(plan_path('v4')), 'audit_sha256': file_hash(inputs / 'audit.json'),
        'reference_manifest_sha256': 'tiny', 'datasets': {'mask1': {'manifest_sha256': file_hash(inputs / 'mask1/manifest.json')}}})
    models = tmp_path / 'models.json'
    write_json(models, [{'name': 'tiny', 'model': 'tiny', 'base': str(base)}])
    prepare_amendment(data, tmp_path / 'amendments/mask1', per_stratum=1)
    args = argparse.Namespace(inputs=str(inputs), smoke=True, queue_name='stability', stage='screen', phase='dev',
        analysis_lock=None, trained_from=None, selection=None, gate=None, mask=None,
        models=str(models), model_names=['tiny'], device='cpu', dtype='float32', prefix_batch=8,
        out=str(tmp_path / 'queue'), amendments=str(tmp_path / 'amendments'))
    plan(args)
    queue = json.loads((tmp_path / 'queue/queue.json').read_text())
    configs = [json.loads(Path(j['config_path']).read_text()) for j in queue['jobs'] if j['kind'] == 'train']
    assert len(configs) == 4 and {c['objective'] for c in configs} == {'ce'}
    assert {(c['learning_rate'],c['replay_weight']) for c in configs} == {(1e-4,0),(3e-5,0),(1e-4,.25),(3e-5,.25)}
    assert all(c['max_steps'] == 1 and c['checkpoint_steps'] == [0,1] and c['plan'] == 'v5' for c in configs)
    args.queue_name = 'matched'
    with pytest.raises(ValueError, match='stability-selection'):
        plan(args)


def test_posthoc_algebra_changes_joint_without_changing_interaction():
    from analysis.q1_prior_residual import crossed, pool_decomposition
    scores = np.column_stack(([3,1,1,3], np.zeros(4)))
    before, after = crossed(scores), crossed(scores + [-2,0])
    assert before['joint'] == 0 and after['joint'] == 1 and before['interaction'] == after['interaction'] == 4
    between, within = pool_decomposition(.2,.3,.5,.4)
    assert between + within == pytest.approx(.5*.4-.2*.3)


def test_stability_admission_uses_full_retention_progress_and_exact_step(tmp_path):
    """Synthetic counts test the admission rule; this is not scientific evidence."""
    from iclr.research_stability import admit, verify_selection
    from iclr.research import freeze
    common = dict(code_hash=code_hash(), plan_hash=file_hash(plan_path('v5')),
                  base_hash='base', data_hash='data', amendment_hash='amendment')
    training, initial, candidate = [tmp_path / p for p in ('training', 'initial', 'candidate')]
    def seal(path, binding):
        write_json(path / 'DONE', {'binding': binding,
            'files': {p.name: file_hash(p) for p in path.iterdir() if p.is_file() and p.name != 'DONE'}})
    cfg = dict(dtype='float32', objective='ce', seed=0)
    bound = {**common, 'config': cfg}
    write_json(training / 'budget.json', {'optimizer_steps': 128})
    for step in (0,32):
        cp = training / 'checkpoints' / f'step_{step:06d}'
        write_json(cp / 'budget.json', {'optimizer_steps': step})
        seal(cp, bound)
        monitor = training / 'monitors' / cp.name
        write_json(monitor / 'monitor.json', {'ordinary': [{'family': 'TRAIN', 'correct_mass': .01 + step*.001}]})
        seal(monitor, {**bound, 'training_receipt_hash': file_hash(cp / 'DONE')})
    seal(training, bound)
    def atomic(plan_success=20):
        return {mode: [{'operation': op, 'witness': [op], 'parse_ok': True,
                       'correct': i < plan_success, 'semantic_correct': op != 'SH1'}
                      for op in core.OPS for i in range(20)] for mode in ('plan', 'apply')}
    for path, mass in ((initial,.01),(candidate,.015)):
        write_json(path / 'atomic_retention.json', atomic())
        write_json(path / 'summary.json', {'metrics': [{'family': 'A', 'depth': 3, 'kind': 'ordinary',
            'policy': 'local', 'temperature': 1, 'correct_mass': mass}]})
    ib = {**common, 'config': {'phase': 'dev', 'mode': 'full'}}
    cb = {**ib, 'checkpoint_step': 32, 'training_receipt_hash': file_hash(training / 'checkpoints/step_000032/DONE')}
    seal(initial, ib); seal(candidate, cb)
    queue = tmp_path / 'queue.json'
    write_json(queue, {'schema': 'iclr.research.queue.v5', 'phase': 'dev', 'experiments': ['stability'],
        'source_code_hash': code_hash(), 'research_plan_sha256': common['plan_hash'],
        'research_plan_path': 'plans/research_v5.json', 'smoke': False,
        'jobs': [{'id':'q06_mask1_R2_seed0_train','result':str(training)},
                 {'id':'q06_mask1_initial','result':str(initial)}]})
    selected = tmp_path / 'selected.json'
    assert admit(queue,'R2',32,selected,candidate)
    assert verify_selection(selected,common['plan_hash'])['settings']['max_steps'] == 32
    assert not next(r['required'] for r in json.loads(selected.read_text())['checks'] if r['operation']=='SH1' and r['mode']=='apply')
    with pytest.raises(ValueError,match='Freeze'):
        freeze(argparse.Namespace(queues=[str(queue)],out=str(tmp_path/'lock.json')))
    # A monitor cannot masquerade as full dev; nor can a step-32 receipt become step 64.
    with pytest.raises(ValueError,match='exact step'):
        admit(queue,'R2',64,tmp_path/'wrong.json',candidate)
    write_json(candidate / 'atomic_retention.json', atomic(plan_success=18))
    seal(candidate,cb)
    with pytest.raises(ValueError,match='evidence changed'):
        verify_selection(selected,common['plan_hash'])
    assert not admit(queue,'R2',32,tmp_path/'failed.json',candidate)
    with pytest.raises(ValueError,match='failed'):
        verify_selection(tmp_path/'failed.json',common['plan_hash'])


def test_source_archive_runs_without_git_metadata(tmp_path, monkeypatch):
    from iclr import common
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    result = common.environment()
    assert result['git_commit'] is None and 'without Git metadata' in result['git_status']
