import argparse
import json
import zipfile
from pathlib import Path

import pytest

from iclr.common import code_hash, file_hash, tree_hash, write_json
from iclr.upgrade import EXPERIMENTS, bundle, plan, run_queue, send_results, verify_bundle


def plan_args(out, prepared, **extra):
    return argparse.Namespace(out=str(out), models='plans/upgrade_models.json', reference='outputs/data',
        prepared_data=str(prepared), smoke=False, include_grpo=False, include_external=False,
        device='cuda', dtype='bfloat16', prefix_batch=32, **extra)


@pytest.fixture
def prepared(tmp_path):
    directory = tmp_path / 'prepared'
    directory.mkdir()
    write_json(directory / 'protocol.json', {'schema': 'iclr.upgrade.protocol.v2', 'smoke': False,
        'analysis_plan_sha256': file_hash('plans/upgrade_analysis_plan.json'),
        'reference_manifest_sha256': json.loads(Path('evidence/upgrade/previous_q06.json').read_text())['runs'][0]['data_hash']})
    return directory


def test_default_is_old_q06_evaluation_and_deferred_blocks_cannot_start(tmp_path, prepared):
    plan(plan_args(tmp_path / 'default', prepared))
    queue = json.loads((tmp_path / 'default/queue.json').read_text())
    assert len(queue['jobs']) == 9 and all(j['kind'] not in ('train', 'init') for j in queue['jobs'])
    assert [m['name'] for m in queue['models']] == ['q06']
    assert all('--phase' not in j['argv'] or j['argv'][j['argv'].index('--phase') + 1] == 'dev' for j in queue['jobs'])
    for experiment in (list(EXPERIMENTS)[2], list(EXPERIMENTS)[4]):
        with pytest.raises(ValueError, match='retired|deferred'):
            plan(plan_args(tmp_path / experiment, prepared, experiments=[experiment]))
    with pytest.raises(ValueError, match='analysis-lock'):
        plan(plan_args(tmp_path / 'final', prepared, phase='final'))


def test_disjoint_pilot_remaining_and_multi_server_mask_shards(tmp_path, prepared):
    memberships = []
    for stage, count in (('pilot', 4), ('remaining', 12)):
        out = tmp_path / stage
        plan(plan_args(out, prepared, experiments=['02_new_pair_masks'], stage=stage))
        queue = json.loads((out / 'queue.json').read_text())
        trains = [j for j in queue['jobs'] if j['kind'] == 'train']
        assert len(trains) == count
        assert queue['models'][0]['required_adapters'] == []
        assert all(len(j['depends_on']) == 1 and '_baseline_' in j['depends_on'][0] for j in trains)
        for job in trains:
            config = json.loads(Path(job['argv'][-1]).read_text())
            assert config['num_examples'] == 6250
            other = job['id'].replace('atomic_control', 'composition') if 'atomic_control' in job['id'] else job['id'].replace('composition', 'atomic_control')
            assert other in {j['id'] for j in trains}
        memberships.append({j['id'] for j in trains})
    assert not memberships[0] & memberships[1] and len(set.union(*memberships)) == 16
    shards = []
    for mask in ('mask1', 'mask2'):
        out = tmp_path / mask
        plan(plan_args(out, prepared, experiments=['02_new_pair_masks'], masks=[mask]))
        jobs = json.loads((out / 'queue.json').read_text())['jobs']
        assert sum(j['kind'] == 'train' for j in jobs) == 2
        shards.append({j['id'] for j in jobs if j['resource'] == 'gpu'})
    assert not shards[0] & shards[1]
    plan(plan_args(tmp_path / 'witness', prepared, experiments=['04_correct_program_choice']))
    jobs = json.loads((tmp_path / 'witness/queue.json').read_text())['jobs']
    assert {j['id'] for j in jobs if j['kind'] == 'train'} == {'q06_witness_fixed_seed0', 'q06_witness_balanced_seed0'}
    plan(plan_args(tmp_path / 'external', prepared, experiments=['06_other_model_family'], stage='complete'))
    jobs = json.loads((tmp_path / 'external/queue.json').read_text())['jobs']
    assert sum(j['kind'] in ('train', 'init') for j in jobs) == 9
    assert all('_seed1' not in j['id'] and '_seed2' not in j['id'] for j in jobs)


def test_queue_dependencies_resume_and_verified_export(tmp_path):
    script = tmp_path / 'worker.py'
    script.write_text('''import json, sys, time
from pathlib import Path
from iclr.common import file_hash, write_json
out = Path(sys.argv[1]); out.mkdir()
write_json(out / "start.json", {"time": time.time()})
time.sleep(.2)
write_json(out / "result.json", {"value": 42, "time": time.time()})
write_json(out / "DONE", {"files": {p.name: file_hash(p) for p in out.glob("*.json")}})
''')
    jobs = [{'id': name, 'argv': ['-c', script.read_text(), str(tmp_path / name)], 'result': str(tmp_path / name),
             'kind': 'receipt', 'resource': 'gpu', 'depends_on': dependencies}
            for name, dependencies in (('a', []), ('b', ['a']), ('c', []))]
    path = tmp_path / 'queue.json'
    reference = tmp_path.parent / (tmp_path.name + '_reference')
    reference.mkdir()
    (reference / 'train.jsonl').write_text('{"old_task": 1}\n')
    write_json(reference / 'manifest.json', {'files': {'train.jsonl': {'sha256': file_hash(reference / 'train.jsonl')}}})
    write_json(path, {'models': [], 'source_code_hash': code_hash(), 'jobs': jobs,
                      'reference': str(reference), 'reference_hash': file_hash(reference / 'manifest.json')})
    write_json(tmp_path / 'status.json', {'queue_sha256': file_hash(path),
               'jobs': {j['id']: {'status': 'planned'} for j in jobs}})
    args = argparse.Namespace(queue=str(path), gpus=['0', '1'], cpu_threads=1, retry_failed=False, send_on_complete=True)
    run_queue(args)
    times = {name: [json.loads((tmp_path / name / f).read_text())['time'] for f in ('start.json', 'result.json')]
             for name in ('a', 'b', 'c')}
    assert times['b'][0] >= times['a'][1]
    assert max(times['a'][0], times['c'][0]) < min(times['a'][1], times['c'][1])
    hashes = {name: file_hash(tmp_path / name / 'DONE') for name in times}
    run_queue(args)
    assert hashes == {name: file_hash(tmp_path / name / 'DONE') for name in times}
    assert len(list(tmp_path.parent.glob('send_to_artem_exp_' + tmp_path.name + '_*.zip'))) == 2
    archive = tmp_path.parent / (tmp_path.name + '.zip')
    bundle(argparse.Namespace(out=str(tmp_path), archive=str(archive), include_models=False, allow_incomplete=False))
    with zipfile.ZipFile(archive) as saved:
        assert saved.testzip() is None
        assert json.loads(saved.read('EXPORT_MANIFEST.json'))['status'] == 'complete'
        assert json.loads(saved.read('results/b/result.json'))['value'] == 42
        assert saved.read('results/reference_data/train.jsonl') == (reference / 'train.jsonl').read_bytes()
    (tmp_path / 'a/model.safetensors').write_bytes(b'weights must not be sent')
    destination = tmp_path.parent / ('send_to_artem_exp_' + tmp_path.name)
    send_results(argparse.Namespace(out=str(tmp_path), destination=str(destination), allow_incomplete=False))
    assert (destination / 'START_HERE_RU.md').is_file()
    assert (destination / 'results/analysis/runs.csv').is_file()
    assert not list(destination.rglob('*.safetensors'))
    verify_bundle(argparse.Namespace(archive=str(destination.with_suffix('.zip'))))
    # Verification of the downloaded bundle must not need source output paths.
    for directory in (tmp_path / name for name in times):
        directory.rename(directory.with_name(directory.name + '_moved'))
    verify_bundle(argparse.Namespace(archive=str(archive)))
    changed = tmp_path.parent / (tmp_path.name + '_changed.zip')
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(changed, 'w') as target:
        for entry in source.infolist():
            target.writestr(entry.filename, b'changed' if entry.filename == 'results/b/result.json' else source.read(entry))
    with pytest.raises(ValueError, match='failed verification'):
        verify_bundle(argparse.Namespace(archive=str(changed)))


def test_v3_export_includes_external_decisions_and_training_without_weights(tmp_path):
    prior, final = tmp_path / 'dev', tmp_path / 'final'
    prior.mkdir(); final.mkdir()
    (prior / 'rankings.jsonl').write_text('{"scores": [1, 2]}\n')
    write_json(prior / 'queue.json', {'jobs': [], 'models': [], 'source_code_hash': code_hash()})
    write_json(prior / 'status.json', {'jobs': {}, 'queue_sha256': file_hash(prior / 'queue.json')})
    selection, lock = tmp_path / 'selection.json', tmp_path / 'lock.json'
    write_json(selection, {'source_queue': str(prior / 'queue.json'),
                          'source_queue_sha256': file_hash(prior / 'queue.json')})
    write_json(lock, {'queues': {str(prior / 'queue.json'): file_hash(prior / 'queue.json')}})
    train, gate = prior / 'training', tmp_path / 'gate'
    for directory in (train, gate):
        directory.mkdir()
        write_json(directory / 'budget.json', {'tokens': 8})
        write_json(directory / 'DONE', {'files': {'budget.json': file_hash(directory / 'budget.json')}})
        for suffix in ('.safetensors', '.bin', '.pt', '.pth', '.ckpt'):
            (directory / ('weights' + suffix)).write_bytes(b'omit')
    result, config = final / 'evaluation', final / 'config.json'
    result.mkdir()
    write_json(config, {'training_receipt': str(train), 'gate': str(gate), 'analysis_lock': str(lock)})
    binding = {'training_receipt_hash': file_hash(train / 'DONE'), 'gate_receipt_hash': file_hash(gate / 'DONE'),
               'analysis_lock_hash': file_hash(lock)}
    write_json(result / 'binding.json', binding)
    write_json(result / 'DONE', {'binding': binding, 'files': {'binding.json': file_hash(result / 'binding.json')}})
    job = {'id': 'eval', 'kind': 'receipt', 'result': str(result), 'config_path': str(config),
           'config_sha256': file_hash(config)}
    write_json(final / 'queue.json', {'schema': 'iclr.research.queue.v3', 'source_code_hash': code_hash(),
        'jobs': [job], 'selection_path': str(selection), 'selection_hash': file_hash(selection),
        'analysis_lock_path': str(lock), 'analysis_lock_sha256': file_hash(lock)})
    write_json(final / 'status.json', {'queue_sha256': file_hash(final / 'queue.json'),
        'jobs': {'eval': {'status': 'done', 'receipt_sha256': file_hash(result / 'DONE')}}})
    destination = tmp_path / 'send_to_artem_exp_final'
    send_results(argparse.Namespace(out=str(final), destination=str(destination), allow_incomplete=False))
    references = json.loads((destination / 'DEPENDENCIES.json').read_text())
    for source in (selection, lock, train, gate, prior / 'queue.json'):
        exported = destination / references[str(source)]['archive_path']
        assert exported.exists()
        if source.is_file():
            assert exported.read_bytes() == source.read_bytes()
    archived_dev = destination / references[str(prior / 'queue.json')]['archive_path']
    assert archived_dev.with_name('rankings.jsonl').read_bytes() == (prior / 'rankings.jsonl').read_bytes()
    assert not any(p.suffix in ('.safetensors', '.bin', '.pt', '.pth', '.ckpt') for p in destination.rglob('*'))
    verify_bundle(argparse.Namespace(archive=str(destination.with_suffix('.zip'))))
    lock.write_text('{}')
    with pytest.raises(ValueError, match='Export dependency changed'):
        bundle(argparse.Namespace(out=str(final), archive=str(tmp_path / 'changed.zip'),
                                  include_models=False, allow_incomplete=False))


def test_final_reuses_exact_completed_cells_and_rejects_wrong_training(tmp_path, prepared):
    history = json.loads(Path('evidence/upgrade/previous_q06.json').read_text())
    proto = json.loads((prepared / 'protocol.json').read_text())
    proto['datasets'] = {'mask1': {'manifest_sha256': 'mask-one'}}
    write_json(prepared / 'protocol.json', proto)
    lock = tmp_path / 'analysis.lock.json'
    write_json(lock, {'code_hash': code_hash(), 'protocol_sha256': file_hash(prepared / 'protocol.json'),
        'history_receipts': {'q06': file_hash('evidence/upgrade/previous_q06.json')}})
    source = tmp_path / 'completed_dev'
    for arm in ('atomic_control', 'composition'):
        root = source / f'experiments/02_new_pair_masks/training/q06_mask1_{arm}_seed0'
        adapter = root / 'adapter'
        adapter.mkdir(parents=True)
        (adapter / 'model.safetensors').write_bytes(b'test payload')
        write_json(root / 'DONE', {'payload_hash': tree_hash(adapter),
            'files': {'adapter/model.safetensors': file_hash(adapter / 'model.safetensors')}, 'binding': {
            'code_hash': code_hash(), 'data_hash': 'mask-one', 'base_hash': history['runs'][0]['base_hash'],
            'config': {'model': 'Qwen/Qwen3-0.6B', 'seed': 0, 'epochs': 2, 'num_examples': 6250,
                       'supervision': 'trace', 'witness_policy': 'fixed', 'replay_fraction': 1. if arm == 'atomic_control' else .2}}})
    def args(out):
        return plan_args(out, prepared, experiments=['02_new_pair_masks'], masks=['mask1'],
                         phase='final', analysis_lock=str(lock), trained_from=str(source))
    plan(args(tmp_path / 'final'))
    jobs = json.loads((tmp_path / 'final/queue.json').read_text())['jobs']
    assert all(j['kind'] not in ('train', 'init') for j in jobs)
    evaluations = [j for j in jobs if j['id'].endswith('_closed')]
    assert len(evaluations) == 2 and all(str(source) in ' '.join(j['argv']) for j in evaluations)
    wrong = source / 'experiments/02_new_pair_masks/training/q06_mask1_composition_seed0/DONE'
    receipt = json.loads(wrong.read_text()); receipt['binding']['config']['seed'] = 1
    write_json(wrong, receipt)
    with pytest.raises(ValueError, match='training cell'):
        plan(args(tmp_path / 'wrong'))
