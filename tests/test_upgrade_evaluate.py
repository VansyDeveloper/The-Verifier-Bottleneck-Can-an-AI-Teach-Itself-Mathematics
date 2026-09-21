import argparse
import json
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from iclr.common import code_hash, file_hash, read_jsonl, verify_receipt, write_json
from iclr.data import generate
from iclr.modeling import load
from iclr.smoke import create_tiny_model
from iclr.upgrade_data import prepare
from iclr.upgrade_evaluate import run
from iclr.upgrade_audit import audit
from iclr.train import run as train
from iclr.upgrade import import_data
from iclr.upgrade_analysis import compare, reanalyze


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
    assert receipt['binding']['phase'] == 'dev'
    assert all(r['split'].startswith('dev') for r in read_jsonl(out / 'rankings.jsonl'))
    summaries = json.loads((out / 'hitk.json').read_text())
    assert {r['score_key'] for r in summaries} == {'score', 'local_score', 'format_score'}
    assert any(r['target_control'] == 'wrong_target' for r in summaries)
    assert any(r['depth'] == 4 and r['method'] == 'bigram' for r in summaries)
    assert not any(r['depth'] == 4 and r['method'] == 'full' for r in summaries)
    cells = json.loads((out / 'execution_summary.json').read_text())['cells']
    assert {c['kind'] for c in cells} == {'atomic', 'program', 'true_intermediate'}
    align = json.loads((out / 'target_alignment.json').read_text())
    assert all(v['eligible'] == 8 and not v['skipped'] for v in align['scores'].values())
    run(args)
    assert verify_receipt(out) == receipt
    with (out / 'prefixes.jsonl').open('a') as stream:
        stream.write('{}\n')
    with pytest.raises(ValueError, match='missing or changed'):
        run(args)
    args.out, args.phase = str(tmp_path / 'final_eval'), 'final'
    with pytest.raises(ValueError, match='analysis-lock'):
        run(args)
    lock = tmp_path / 'lock.json'
    write_json(lock, {'code_hash': code_hash(), 'dataset_hashes': [receipt['binding']['data_hash']],
                      'analysis_plan_sha256': file_hash('plans/upgrade_analysis_plan.json')})
    args.analysis_lock = str(lock)
    run(args)
    final = verify_receipt(args.out)
    assert final['binding']['phase'] == 'final'
    assert all(r['split'].startswith('final') for r in read_jsonl(Path(args.out) / 'rankings.jsonl'))
    expected_ids = {r['task_id'] for r in read_jsonl(data / 'original/final_atomic.jsonl')}
    actual_ids = {r['task_id'] for r in read_jsonl(Path(args.out) / 'execution.jsonl') if r['kind'] == 'atomic'}
    assert actual_ids == expected_ids
    rescored = tmp_path / 'offline_analysis'
    reanalyze(argparse.Namespace(source=args.out, data=str(data / 'original'), out=str(rescored)))
    assert json.loads((rescored / 'DONE').read_text())['model_inference_repeated'] is False
    result = json.loads((rescored / 'target_alignment.json').read_text())
    files = []
    for arm in ('composition', 'atomic_control'):
        result['binding']['arm'] = arm
        path = tmp_path / (arm + '.json')
        write_json(path, result)
        files.append(str(path))
    contrast = tmp_path / 'contrast.json'
    compare(argparse.Namespace(alignment=files, treatment='composition', control='atomic_control', out=str(contrast)))
    assert json.loads(contrast.read_text())['estimate'] == 0
    assert json.loads(contrast.read_text())['ci95_start_bootstrap'] == [0, 0]


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
