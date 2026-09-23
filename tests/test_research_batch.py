"""The unattended grid must cover each declared training cell exactly once."""

import argparse
import json
from pathlib import Path
import tarfile

import pytest

from iclr.common import ROOT, file_hash
from iclr.research import plan


def test_fixed_batch_inventory_and_plan_boundaries(tmp_path):
    with tarfile.open(ROOT / 'evidence/research_v4/Q1_Q3_20260923/shared_inputs_v4.tar.gz') as archive:
        archive.extractall(tmp_path, filter='data')
    args = argparse.Namespace(inputs=str(tmp_path / 'shared_inputs_v4'), smoke=False,
        queue_name='batch', stage='screen', phase='dev', analysis_lock=None, trained_from=None,
        selection=None, gate=None, mask=None, models=str(ROOT / 'plans/research_v5_models.example.json'),
        model_names=['q06'], device='cuda', dtype='bfloat16', prefix_batch=16,
        amendments=str(ROOT / 'evidence/research_v6/amendments'))
    cells = set()
    for number in range(1, 9):
        args.batch, args.out = number, str(tmp_path / f'batch_{number:02d}')
        plan(args)
        queue = json.loads((Path(args.out) / 'queue.json').read_text())
        assert queue['schema'] == 'iclr.research.queue.v6' and queue['phase'] == 'dev'
        assert queue['research_plan_sha256'] == file_hash(ROOT / 'plans/research_v6.json')
        assert queue['batch_priority'] == (1 if number <= 4 else 2)
        assert not queue['selection_path'] and not queue['stability_selection_path']
        assert len(queue['jobs']) == 27
        trains = [j for j in queue['jobs'] if j['kind'] == 'train']
        assert len(trains) == 12
        initial = [j for j in queue['jobs'] if j['id'].endswith('_initial')]
        assert len(initial) == 2
        for job in trains:
            cfg = json.loads(Path(job['config_path']).read_text())
            assert job['config_sha256'] == file_hash(job['config_path'])
            assert cfg['plan'] == 'v6' and cfg['monitor'] and cfg['resume_from'] == 'latest'
            assert cfg['epochs'] == 2 and cfg['max_steps'] == 128
            assert cfg['checkpoint_steps'] == [0, 8, 16, 32, 64, 128]
            assert job['depends_on'] == [f'q06_{Path(cfg["data"]).name}_initial']
            evaluation = next(j for j in queue['jobs'] if j['id'] == job['id'].removesuffix('_train') + '_eval')
            assert evaluation['depends_on'] == [job['id']]
            key = (Path(cfg['data']).name, cfg['seed'], cfg['objective'], cfg['learning_rate'], cfg['replay_weight'])
            assert key not in cells
            cells.add(key)
    assert cells == {(mask, seed, arm, lr, replay)
        for mask in ('mask1', 'mask2') for seed in (0, 1, 2)
        for arm in ('ce', 'ce_cf', 'ce_entropy', 'ce_cf_entropy')
        for lr in (1e-4, 3e-5) for replay in (0., .25)}
    for change in ({'phase': 'final'}, {'mask': 'mask1'}, {'plan': 'v5'}, {'batch': None}, {'with_entropy': True}):
        with pytest.raises(ValueError, match='Batch requires'):
            plan(argparse.Namespace(**{**vars(args), **change}))
    with pytest.raises(ValueError, match='Fixed queues'):
        plan(argparse.Namespace(**{**vars(args), 'learning_rate': 2e-4}))
