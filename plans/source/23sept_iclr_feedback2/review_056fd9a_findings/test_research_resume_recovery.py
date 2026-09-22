"""Recovery regressions to run in the project's locked environment after the patch."""
import json
from pathlib import Path

import pytest

from iclr import research_train as train


def test_explicit_resume_and_config_resume_do_not_conflict(monkeypatch):
    # Exercise actual configuration handling before model loading. The existing
    # test_research_v5 suite separately checks real Adam/model continuation.
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


def test_recovery_rejects_committed_corruption_before_truncating(tmp_path):
    for name in train.LOG_NAMES:
        (tmp_path / name).write_bytes(b'{"step": 1}\n')
    snapshot = train.snapshot_logs(tmp_path)
    (tmp_path / 'training_stream.jsonl').write_bytes(b'X"step": 1}\n')
    with (tmp_path / 'training_metrics.jsonl').open('ab') as stream:
        stream.write(b'uncommitted tail')
    before = {name: (tmp_path / name).read_bytes() for name in train.LOG_NAMES}
    with pytest.raises(ValueError, match='Committed log prefix'):
        train.restore_logs(tmp_path, snapshot)
    assert before == {name: (tmp_path / name).read_bytes() for name in train.LOG_NAMES}


def test_step0_missing_logs_are_valid(tmp_path):
    snapshot = train.snapshot_logs(tmp_path)
    (tmp_path / 'training_metrics.jsonl').write_bytes(b'{"step": 1,')
    train.restore_logs(tmp_path, snapshot)
    assert (tmp_path / 'training_metrics.jsonl').read_bytes() == b''


def test_latest_ignores_unpublished_partial_checkpoints(tmp_path):
    for name in ('step_000008', 'step_000016.partial'):
        path = tmp_path / 'checkpoints' / name
        path.mkdir(parents=True)
        (path / 'DONE').write_text('{}')
    assert [p.parent.name for p in train.complete_checkpoint_receipts(tmp_path)] == ['step_000008']
