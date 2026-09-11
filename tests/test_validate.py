import json
import shutil
from types import SimpleNamespace

import pytest

from iclr import evaluate
from iclr.common import file_hash, read_jsonl, tree_hash, write_json
from iclr.data import generate
from iclr.validate import validate
import composition_core as core
import composition_eval as legacy


def make_saved_evaluation(root, monkeypatch, trained):
    data, run = root / 'data', root / 'run'
    generate(data, size=20, eval_size=2, atomic_per_op=5, smoke=True)
    binding = dict(base_hash='base', data_hash=file_hash(data / 'manifest.json'), code_hash='old-code',
                   dtype='float32', device='cpu', prefix_batch=8)
    if trained:
        payload = run / 'adapter'
        payload.mkdir(parents=True)
        (payload / 'weights').write_bytes(b'fixture payload; no model is loaded')
        payload_hash = tree_hash(payload)
        config = dict(seed=0, method='ce', device='cpu', dtype='float32', prefix_batch=8,
                      micro_batch=1, epochs=1, effective_batch=1, num_examples=1,
                      data='/different/machine/data', output='/different/machine/run')
        done_binding = {'config': config, **{key: binding[key] for key in ('base_hash', 'data_hash', 'code_hash')}}
        write_json(run / 'config.resolved.json', done_binding)
        task = read_jsonl(data / 'train.jsonl')[0]
        legacy.write_jsonl(run / 'training_stream.jsonl', [dict(step=1, task_id=task['task_id'], kind='composition',
            input_ids=[1, 2], labels=[-100, 2], set_objective=None)])
        legacy.write_jsonl(run / 'training_metrics.jsonl', [dict(step=1, loss=.5, gradient_norm=.1,
            target_tokens=1, examples=1, seconds=1.)])
        write_json(run / 'budget.json', dict(optimizer_steps=1, example_exposures=1, all_target_tokens=1,
            composition_exposures=1, composition_target_tokens=1, unique_tasks=1, prompt_tokens=1,
            masked_target_tokens=0, total_forward_tokens=2))
        for name in ('reload_probe.json', 'resolved_model.json', 'environment.json', 'reload_check.json'):
            write_json(run / name, {'fixture': True})
        write_json(run / 'TRAINED', {'payload_hash': payload_hash,
            'files': {name: file_hash(run / name) for name in
                      ('reload_probe.json', 'resolved_model.json', 'environment.json',
                       'training_stream.jsonl', 'training_metrics.jsonl', 'budget.json')}})
        binding.update(training_seed=0, method='ce', model_hash=payload_hash)
        out = run / 'eval'
    else:
        binding.update(adapter_hash=None, split='dev')
        done_binding = binding
        out = run
        write_json(out / 'binding.json', binding)

    def score_task(model, tokenizer, token_ids, row, prefix_batch, accounting):
        programs = sorted(core.enumerate_programs(row['depth']))
        candidates = [dict(program=list(program), score=-float(rank), rank=rank,
                           correct=core.verify_program(row['start'], row['target'], program, row['p']))
                      for rank, program in enumerate(programs, 1)]
        common = {key: row[key] for key in ('task_id', 'task_fingerprint', 'split', 'p', 'depth')}
        metric = dict(schema='stage4.composition.v5.task-metrics.v1', **common, **legacy.ranking_metrics(candidates))
        raw = dict(schema='stage4.composition.v5.ranking.v1', **common, start=row['start'],
                   target=row['target'], ranking=candidates)
        return metric, raw

    def atomic_plan(model, tokenizer, token_ids, rows, prefix_batch):
        raw = [dict(task_id=row['task_id'], operation=row['operation'],
                    plan_prediction=row['operation'], plan_correct=True) for row in rows]
        return {'overall': 1., 'by_operation': dict.fromkeys(core.OPS, 1.)}, raw

    def atomic_apply(model, tokenizer, rows, prefix_batch):
        raw = [dict(task_id=row['task_id'], operation=row['operation'], target=core.format_state(row['target']),
                    raw_completion=core.format_state(row['target']), correct=True,
                    match_mode='stripped_exact_equality') for row in rows]
        return {'overall': 1., 'by_operation': dict.fromkeys(core.OPS, 1.)}, raw

    monkeypatch.setattr(evaluate, 'environment', lambda: {'test': True})
    monkeypatch.setattr(legacy, 'score_task', score_task)
    monkeypatch.setattr(legacy, 'atomic_plan_metrics', atomic_plan)
    monkeypatch.setattr(legacy, 'atomic_apply_metrics', atomic_apply)
    model = SimpleNamespace(eval=lambda: None)
    tokenizer = SimpleNamespace(backend_tokenizer=SimpleNamespace(to_str=lambda: 'test tokenizer'))
    evaluate.evaluate(model, tokenizer, {}, data, out, binding)
    receipt = {'binding': done_binding, 'files': {str(path.relative_to(run)): file_hash(path)
               for path in run.rglob('*.json*') if 'adapter' not in path.parts}}
    if trained:
        receipt['payload_hash'] = payload_hash
    write_json(run / 'DONE', receipt)
    return data, run


@pytest.mark.parametrize('trained', [False, True])
def test_copied_output_recomputes_and_rejects_corruption(tmp_path, monkeypatch, trained):
    data, run = make_saved_evaluation(tmp_path / 'original', monkeypatch, trained)
    copied = tmp_path / 'copied'
    shutil.copytree(data.parent, copied)
    shutil.rmtree(data.parent)
    data, run = copied / 'data', copied / 'run'
    result = validate(run, data)
    assert result['raw_results_recomputed'] and result['composition_tasks'] == 8
    assert result['atomic_tasks'] == 10 and result['payload_hash_checked'] == trained
    assert not result['model_inference_repeated']
    out = run / 'eval' if trained else run
    filename = 'eval/metrics.jsonl' if trained else 'metrics.jsonl'
    metrics = out / 'metrics.jsonl'
    original = metrics.read_text()
    rows = [json.loads(line) for line in original.splitlines()]
    rows[0]['correct_mass'] = .123456789
    metrics.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='missing or changed'):
        validate(run, data)
    # Updating the checksum cannot hide disagreement with saved raw rankings.
    receipt = json.loads((run / 'DONE').read_text())
    receipt['files'][filename] = file_hash(metrics)
    write_json(run / 'DONE', receipt)
    with pytest.raises(ValueError, match='correct_mass'):
        validate(run, data)
    metrics.write_text('\n'.join(original.splitlines()[1:]) + '\n')
    receipt['files'][filename] = file_hash(metrics)
    write_json(run / 'DONE', receipt)
    with pytest.raises(ValueError, match='coverage'):
        validate(run, data)
    metrics.write_text(original)
    receipt['files'][filename] = file_hash(metrics)
    write_json(run / 'DONE', receipt)
    assert validate(run, data)['status'] == 'PASS'
    ranking_path = out / 'rankings.jsonl'
    rankings = [json.loads(line) for line in ranking_path.read_text().splitlines()]
    original_rankings = ranking_path.read_text()
    rankings[0]['ranking'][0]['correct'] = not rankings[0]['ranking'][0]['correct']
    ranking_path.write_text(''.join(json.dumps(row) + '\n' for row in rankings))
    rank_key = 'eval/rankings.jsonl' if trained else 'rankings.jsonl'
    receipt['files'][rank_key] = file_hash(ranking_path)
    write_json(run / 'DONE', receipt)
    with pytest.raises(RuntimeError, match='exact-correct flag'):
        validate(run, data)
    ranking_path.write_text(original_rankings)
    receipt['files'][rank_key] = file_hash(ranking_path)
    write_json(run / 'DONE', receipt)
    if trained:
        training = json.loads((run / 'TRAINED').read_text())
        budget = json.loads((run / 'budget.json').read_text())
        for key in ('all_target_tokens', 'prompt_tokens', 'masked_target_tokens',
                    'total_forward_tokens', 'atomic_plan_exposures', 'SH1_exposures'):
            write_json(run / 'budget.json', {**budget, key: budget.get(key, 0) + 1})
            for record in (receipt, training):
                record['files']['budget.json'] = file_hash(run / 'budget.json')
            write_json(run / 'TRAINED', training)
            write_json(run / 'DONE', receipt)
            with pytest.raises(ValueError, match='training budget'):
                validate(run, data)
        write_json(run / 'budget.json', budget)
        for record in (receipt, training):
            record['files']['budget.json'] = file_hash(run / 'budget.json')
        write_json(run / 'TRAINED', training)
        write_json(run / 'DONE', receipt)
        assert validate(run, data)['training_budget_checked']
        logs = read_jsonl(run / 'training_metrics.jsonl')
        for key in ('step', 'loss', 'gradient_norm', 'seconds'):
            legacy.write_jsonl(run / 'training_metrics.jsonl', [{**logs[0], key: True}])
            for record in (receipt, training):
                record['files']['training_metrics.jsonl'] = file_hash(run / 'training_metrics.jsonl')
            write_json(run / 'TRAINED', training)
            write_json(run / 'DONE', receipt)
            with pytest.raises(ValueError, match='updates|Nonfinite'):
                validate(run, data)
        legacy.write_jsonl(run / 'training_metrics.jsonl', logs)
        for record in (receipt, training):
            record['files']['training_metrics.jsonl'] = file_hash(run / 'training_metrics.jsonl')
        write_json(run / 'TRAINED', training)
        write_json(run / 'DONE', receipt)
        # Dropping a file and its declared checksum must not create a valid bundle.
        name = 'resolved_model.json'
        content = (run / name).read_text()
        (run / name).unlink()
        for record in (receipt, training):
            record['files'].pop(name)
        write_json(run / 'TRAINED', training)
        write_json(run / 'DONE', receipt)
        with pytest.raises(ValueError, match='receipt omits'):
            validate(run, data)
        (run / name).write_text(content)
        for record in (receipt, training):
            record['files'][name] = file_hash(run / name)
        write_json(run / 'TRAINED', training)
        write_json(run / 'DONE', receipt)
        (run / 'adapter' / 'weights').write_bytes(b'changed')
        with pytest.raises(ValueError, match='saved model'):
            validate(run, data)
