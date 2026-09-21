"""Small real-model plumbing checks; GPU is opt-in via ICLR_SMOKE_DEVICE=cuda."""

from collections import Counter
import json
import os
from pathlib import Path

import pytest
import torch

import iclr
import composition_core as core
from iclr.common import file_hash, read_jsonl, verify_receipt, write_json
from iclr.data import _atomic_split
from iclr.modeling import load, program_scores
from iclr.research_data import generate_rows
from iclr.research_dsl import DOMAIN, trajectory, verify
from iclr.research_evaluate import run as evaluate, state_access
from iclr.research_gradient import run as gradient_diagnostic
from iclr.research_model import deterministic_training, score_task, state_prompt_ids
from iclr.research_train import run as train
from iclr.smoke import create_tiny_model


def tiny_setup(tmp_path):
    torch.set_num_threads(1)
    registry = core.FingerprintRegistry()
    registry.shape_counts = Counter()
    spec = {'heldout_motifs': [list(p) for p in core.HELDOUT_MOTIFS]}
    data, files = tmp_path / 'data', {}
    data.mkdir()
    def save(name, rows):
        path = data / (name + '.jsonl')
        path.write_bytes(core.canonical_jsonl_bytes(rows))
        files[path.name] = {'sha256': file_hash(path), 'rows': len(rows)}
    save('train_four', generate_rows('train', 4, spec, registry, 0, family='TRAIN', panel_targets=4, correct_count=1))
    save('train', read_jsonl(data / 'train_four.jsonl'))
    for index, family in enumerate('ABCD'):
        for phase, offset in (('dev', 10), ('final', 20)):
            save(f'{phase}_panel_{family}', generate_rows(f'{phase}_panel_{family}', 4, spec, registry,
                index + offset, family=family, panel_targets=4, correct_count=1))
    save('dev_A', generate_rows('dev_A', 1, spec, registry, 40))
    save('dev_atomic', _atomic_split('dev_atomic', 1, 50, registry))
    save('final_atomic', _atomic_split('final_atomic', 1, 60, registry))
    write_json(data / 'manifest.json', {'schema': 'iclr.research.data.v3', 'status': 'smoke',
        'files': files, 'training_panels': 'train_four.jsonl', 'constraints': spec})
    source = create_tiny_model(tmp_path / 'source')
    model, tokenizer, ids = load(source, initialize=True, device='cpu')
    base = tmp_path / 'base'
    model.save_pretrained(base); tokenizer.save_pretrained(base)
    return data, base


def test_real_forward_backward_save_reload_and_projection_branches(tmp_path, monkeypatch):
    data, base = tiny_setup(tmp_path)
    device = os.environ.get('ICLR_SMOKE_DEVICE', 'cpu')
    dtype = 'bfloat16' if device == 'cuda' else 'float32'
    cfg = dict(base=str(base), data=str(data), device=device, dtype=dtype, prefix_batch=8, epochs=1, reward_steps=1)
    evaluate({**cfg, 'seed': -1, 'arm': 'initial', 'output': str(tmp_path / 'initial'), 'solve_per_family': 1})
    # Production gates are tested below. This test exercises post-gate code even
    # when an untrained tiny model has no scientific state-access advantage.
    admitted = []
    monkeypatch.setattr('iclr.research_train.require_gate', lambda path, kind, binding: admitted.append(kind))
    for objective in ('ce', 'ce_cf_entropy', 'sampled_entropy', 'exact_entropy', 'state_ce', 'state', 'sigreg', 'state_sigreg'):
        out = tmp_path / objective
        train({**cfg, 'output': str(out), 'objective': objective})
        receipt = verify_receipt(out)
        assert receipt['payload_hash']
        assert json.loads((out / 'budget.json').read_text())['example_exposures'] == 4
        assert json.loads((out / 'reload_check.json').read_text())['maximum_error'] <= (.1 if device == 'cuda' else 1e-4)
    assert admitted == ['reward', 'reward', 'state', 'state', 'state', 'state']
    result = tmp_path / 'evaluation'
    evaluate({**cfg, 'output': str(result), 'adapter': str(tmp_path / 'state_sigreg/adapter'), 'solve_per_family': 1})
    assert (result / 'projection_ablation.json').exists()
    assert verify_receipt(result)['binding']['kind'] == 'evaluate'
    state_access({**cfg, 'output': str(tmp_path / 'state_access'), 'panels_per_family': 1})
    assert json.loads((tmp_path / 'state_access/gate.json').read_text())['pass'] is False


def test_exact_sampled_actual_optimizer_diagnostic_and_gate(tmp_path):
    data, base = tiny_setup(tmp_path)
    device = os.environ.get('ICLR_SMOKE_DEVICE', 'cpu')
    cfg = dict(base=str(base), data=str(data), output=str(tmp_path / 'gradient'), device=device,
               dtype='bfloat16' if device == 'cuda' else 'float32', prefix_batch=8,
               train_tasks=1, dev_panels_per_family=1, mc_groups=16, mc_blocks=4, learning_rates=[1e-5, 1e-6])
    gradient_diagnostic(cfg)
    receipt = verify_receipt(tmp_path / 'gradient')
    gate = json.loads((tmp_path / 'gradient/gate.json').read_text())
    assert gate['pass'] is False  # One task is never the scientific admission gate.
    result = json.loads((tmp_path / 'gradient/gradient_diagnostic.json').read_text())
    assert not set(result['train_ids']) & set(result['dev_ids'])
    assert {s['method'] for r in result['reports'] for s in r['isolated_steps']} == {'exact', 'sampled_one_group', 'sampled_mc_mean'}
    if device == 'cuda':
        assert {r['dtype'] for r in result['reports']} == {'float32', 'bfloat16'}
    from iclr.research_io import require_gate
    with pytest.raises(ValueError, match='did not pass'):
        require_gate(tmp_path / 'gradient', 'reward', receipt['binding'])


def test_selected_program_scorer_and_state_token_control(tmp_path):
    data, base = tiny_setup(tmp_path)
    model, tokenizer, ids = load(base, device='cpu')
    row = read_jsonl(data / 'train.jsonl')[0]
    programs = list(core.enumerate_programs(3))
    chosen = [programs[0], programs[12], programs[-1]]
    full, local, _ = score_task(model, tokenizer, ids, row)
    partial_full, partial_local, _ = score_task(model, tokenizer, ids, row, programs=chosen)
    torch.testing.assert_close(partial_full, full[[0, 12, 124]])
    torch.testing.assert_close(partial_local, local[[0, 12, 124]])
    torch.testing.assert_close(full, program_scores(model, tokenizer, ids, row))
    for prefix in ([], row['witness'][:1], row['witness'][:2]):
        truth, n = state_prompt_ids(row, prefix, tokenizer, 'state')
        control, m = state_prompt_ids(row, prefix, tokenizer, 'token_control')
        assert len(truth) == len(control) and n == m
    nonlinear = {'domain': DOMAIN, 'p': 7, 'start': [2, 3, 1, 4], 'target': [4, 3, 1, 4]}
    assert verify(nonlinear, ['SC2'])
    assert trajectory(nonlinear, ['AC1'])[-1] == (1, 2, 3, 4)
    assert trajectory(nonlinear, ['AX1'])[-1] == (3, 1, 4, 2)
    trained, _, _ = load(base, train=True, device='cpu', lora_dropout=.2)
    deterministic_training(trained)
    assert trained.training and any(getattr(m, 'gradient_checkpointing', False) for m in trained.modules())
    assert all(m.p == 0 for m in trained.modules() if isinstance(m, torch.nn.Dropout))


def test_crossed_training_uses_four_independent_cells(tmp_path):
    data, base = tiny_setup(tmp_path)
    fixture = json.loads(Path('plans/source/22sept_iclr_feedback/crossed_panel_fixture.json').read_text())
    rows = []
    for i, start in enumerate(fixture['starts']):
        for j, target in enumerate(fixture['targets']):
            program = fixture['program_A' if i == j else 'program_B']
            rows.append(dict(p=5, degree=2, start=start, target=target, depth=3, family='TRAIN',
                heldout_motifs=[], witness=program, correct_programs=[program], correct_count=1,
                task_id=f'fixture-{i}-{j}', panel_id='fixture', panel_kind='crossed', cell=2*i+j,
                program_A=fixture['program_A'], program_B=fixture['program_B']))
    path = data / 'train_crossed.jsonl'
    path.write_bytes(core.canonical_jsonl_bytes(rows))
    manifest = json.loads((data / 'manifest.json').read_text())
    manifest['files'][path.name] = {'rows': 4, 'sha256': file_hash(path)}
    manifest['training_panels'] = path.name
    write_json(data / 'manifest.json', manifest)
    device = os.environ.get('ICLR_SMOKE_DEVICE', 'cpu')
    output = tmp_path / 'crossed'
    train(dict(base=str(base), data=str(data), output=str(output), objective='ce_cf_entropy',
               epochs=1, device=device, dtype='bfloat16' if device == 'cuda' else 'float32', prefix_batch=8))
    stream = read_jsonl(output / 'training_stream.jsonl')
    assert len(stream) == 4 and len({r['task_id'] for r in stream}) == 4
    assert {r['kind'] for r in stream} == {'crossed'}
