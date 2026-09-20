import argparse
import json

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from iclr.common import file_hash, verify_receipt
from iclr.data import generate
from iclr.modeling import load
from iclr.smoke import create_tiny_model
from iclr.upgrade_data import prepare
from iclr.upgrade_evaluate import run
from iclr.upgrade_audit import audit
from iclr.train import run as train
from iclr.upgrade import import_data


def test_closed_prefix_evaluation_execution_and_receipt(tmp_path):
    torch.set_num_threads(1)
    reference = tmp_path / 'reference'
    generate(reference, size=20, eval_size=2, atomic_per_op=5, smoke=True)
    data = tmp_path / 'upgrade'
    prepare(data, reference, smoke=True)
    copied = tmp_path / 'copied_from_another_computer'
    transfer = argparse.Namespace(source=str(data), out=str(copied), reference=str(reference),
                                  expected_protocol_hash=file_hash(data / 'protocol.json'))
    import_data(transfer)
    import_data(transfer)
    data = copied
    assert audit(data, reference, tmp_path / 'audit')['status'] == 'PASS'
    source = create_tiny_model(tmp_path / 'source')
    model, tokenizer, _ = load(source, initialize=True, device='cpu')
    base = tmp_path / 'base'
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    out = tmp_path / 'eval'
    args = argparse.Namespace(base=str(base), adapter=None, data=str(data / 'original'), out=str(out),
        model='tiny', seed=0, arm='baseline', device='cpu', dtype='float32', prefix_batch=16,
        expected_base_hash=None, expected_adapter_hash=None, execution_tasks_per_family=1,
        reliable_ops=['AC1', 'AX1', 'REV'])
    run(args)
    receipt = verify_receipt(out)
    assert receipt['binding']['data_status'] == 'smoke'
    summaries = json.loads((out / 'hitk.json').read_text())
    assert {r['score_key'] for r in summaries} == {'score', 'local_score', 'format_score'}
    assert any(r['target_control'] == 'wrong_target' for r in summaries)
    assert any(r['depth'] == 4 and r['method'] == 'bigram' for r in summaries)
    assert not any(r['depth'] == 4 and r['method'] == 'full' for r in summaries)
    cells = json.loads((out / 'execution_summary.json').read_text())['cells']
    assert {c['kind'] for c in cells} == {'atomic', 'program', 'true_intermediate'}
    run(args)
    assert verify_receipt(out) == receipt
    with (out / 'prefixes.jsonl').open('a') as stream:
        stream.write('{}\n')
    with pytest.raises(ValueError, match='missing or changed'):
        run(args)


def test_llama_family_atomic_initialization_and_continuation(tmp_path):
    torch.set_num_threads(1)
    source = create_tiny_model(tmp_path / 'source')
    config = json.loads((source / 'config.json').read_text())
    config['model_type'] = 'llama'
    LlamaForCausalLM(LlamaConfig(**config)).save_pretrained(source)
    data = tmp_path / 'data'
    generate(data, size=20, eval_size=2, atomic_per_op=5, smoke=True)
    base = tmp_path / 'atomic'
    common = {'model': str(source), 'data': str(data), 'epochs': 1, 'num_examples': 2,
        'effective_batch': 2, 'micro_batch': 1, 'prefix_batch': 8, 'device': 'cpu', 'dtype': 'float32'}
    train({**common, 'initialize': True, 'replay_fraction': 1., 'output': str(base)})
    out = tmp_path / 'continuation'
    train({**common, 'base': str(base / 'checkpoint'), 'output': str(out)})
    for directory in (base, out):
        verify_receipt(directory)
        assert json.loads((directory / 'reload_check.json').read_text())['max_absolute_difference'] < 1e-4
