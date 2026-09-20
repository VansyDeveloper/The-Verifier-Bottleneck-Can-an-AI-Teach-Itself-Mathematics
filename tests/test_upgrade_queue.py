import argparse
import json
import zipfile
from pathlib import Path

import pytest

from iclr.common import code_hash, file_hash, write_json
from iclr.upgrade import EXPERIMENTS, bundle, plan, run_queue, send_results, verify_bundle


def test_full_plan_has_paired_cells_and_external_initialization(tmp_path):
    plan(argparse.Namespace(out=str(tmp_path), models='plans/upgrade_models.json', reference='outputs/data',
        smoke=False, include_grpo=True, include_external=True, device='cuda', dtype='bfloat16', prefix_batch=32))
    jobs = json.loads((tmp_path / 'queue.json').read_text())['jobs']
    by_id = {job['id']: job for job in jobs}
    assert len(by_id) == len(jobs) == 137
    assert sum(j['kind'] in ('train', 'init') for j in jobs) == 60
    assert sum(j['kind'] == 'train' and j['id'].startswith('q06_mask') for j in jobs) == 16
    for job in jobs:
        assert set(job['depends_on']) <= by_id.keys()
    assert by_id['smol17_baseline']['depends_on'] == ['smol17_atomic_init']
    for method in ('sampled', 'exact'):
        assert len(by_id[f'q06_reward_full_{method}_seed0']['depends_on']) == 7
    for model in ('q06', 'smol17'):
        assert sum(j['kind'] == 'train' and j['id'].startswith(model + '_original_') for j in jobs) == 6
    config = json.loads(Path(by_id['smol17_atomic_init']['argv'][-1]).read_text())
    assert config['initialize'] and len(config['revision']) == 40
    for job in jobs:
        if job['kind'] == 'train' and job['argv'][1] == 'iclr.train':
            config = json.loads(Path(job['argv'][-1]).read_text())
            assert config['num_examples'] == (5000 if '_witness_' in job['id'] else 6250)


def test_experiment_and_seed_shards_keep_pairs_and_require_only_used_weights(tmp_path):
    memberships = []
    for seed in (0, 1):
        out = tmp_path / f'computer_{seed}'
        plan(argparse.Namespace(out=str(out), models='plans/upgrade_models.json', reference='outputs/data',
            smoke=False, include_grpo=False, include_external=False, device='cuda', dtype='bfloat16', prefix_batch=32,
            experiments=['02_new_pair_masks'], seeds=[seed]))
        queue = json.loads((out / 'queue.json').read_text())
        assert len(queue['jobs']) == 18  # eight paired trainings, eight evaluations, preparation/audit
        assert [m['name'] for m in queue['models']] == ['q06']
        assert queue['models'][0]['required_adapters'] == []
        trains = [job for job in queue['jobs'] if job['kind'] == 'train']
        assert len(trains) == 8 and all(job['seed'] == seed for job in trains)
        assert len(list((out / 'configs').glob('*.json'))) == 8
        memberships.append({j['id'] for j in queue['jobs'] if j['resource'] == 'gpu'})
        assert all('/experiments/02_new_pair_masks/' in j['result'] for j in trains)
    assert not memberships[0] & memberships[1]
    for experiment in list(EXPERIMENTS)[4:]:
        with pytest.raises(ValueError, match='Keep experiments'):
            plan(argparse.Namespace(experiments=[experiment], seeds=[0], include_grpo=False, include_external=False))


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
