"""Recovery checks for committed log prefixes and published checkpoints."""

import json

import pytest

from iclr import research_train as train


def test_explicit_resume_and_config_resume_do_not_conflict(monkeypatch):
    monkeypatch.setattr(train, 'start_run', lambda cfg, kind: ({}, {}, True))
    train.run({'resume_from': 'latest'}, resume_from='latest')


def test_recovery_discards_only_uncommitted_torn_tail(tmp_path):
    committed = {}
    for index, name in enumerate(train.LOG_NAMES):
        committed[name] = json.dumps({'step': 1, 'index': index}).encode() + b'\n'
        (tmp_path / name).write_bytes(committed[name])
    snapshot = train.snapshot_logs(tmp_path)
    for name in train.LOG_NAMES:
        with (tmp_path / name).open('ab') as stream:
            stream.write(b'{"step": 2, "loss":')
    train.restore_logs(tmp_path, snapshot)
    assert all((tmp_path / name).read_bytes() == value for name, value in committed.items())


@pytest.mark.parametrize('damage', ['corrupt', 'shorten', 'remove'])
def test_recovery_rejects_committed_damage_before_truncating(tmp_path, damage):
    for name in train.LOG_NAMES:
        (tmp_path / name).write_bytes(b'{"step": 1}\n')
    snapshot = train.snapshot_logs(tmp_path)
    damaged = tmp_path / 'training_stream.jsonl'
    if damage == 'remove':
        damaged.unlink()
    else:
        damaged.write_bytes(b'X"step": 1}\n' if damage == 'corrupt' else b'{"step":')
    with (tmp_path / 'training_metrics.jsonl').open('ab') as stream:
        stream.write(b'uncommitted tail')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match='Committed log prefix'):
        train.restore_logs(tmp_path, snapshot)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_step0_missing_logs_are_valid(tmp_path):
    snapshot = train.snapshot_logs(tmp_path)
    (tmp_path / 'training_metrics.jsonl').write_bytes(b'{"step": 1,')
    train.restore_logs(tmp_path, snapshot)
    assert (tmp_path / 'training_metrics.jsonl').read_bytes() == b''


def test_recovery_requires_committed_boundaries(tmp_path):
    path = tmp_path / 'training_metrics.jsonl'
    path.write_bytes(b'{"step": 1,')
    with pytest.raises(ValueError, match='committed-log snapshot'):
        train.restore_logs(tmp_path, None)
    assert path.read_bytes() == b'{"step": 1,'


def test_latest_ignores_unpublished_partial_checkpoints(tmp_path):
    for name in ('step_000008', 'step_000016.partial'):
        path = tmp_path / 'checkpoints' / name
        path.mkdir(parents=True)
        (path / 'DONE').write_text('{}')
    assert [p.parent.name for p in train.complete_checkpoint_receipts(tmp_path)] == ['step_000008']
