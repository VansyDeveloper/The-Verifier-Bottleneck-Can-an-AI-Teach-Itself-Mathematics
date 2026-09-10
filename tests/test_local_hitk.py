import json
import tempfile
from pathlib import Path

from iclr.local_hitk import analyze_local


def test_local_saved_ranks_and_paired_seed_curves():
    source = Path(__file__).resolve().parents[1] / 'evidence/local_series/manifest.json'
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / 'results'
        curves = analyze_local(source, out)
        assert len(curves) == 250
        for split, expected in [('capacity_d3_train', (43.4444444444, 91.5555555556)),
                                ('capacity_d3_heldout', (36.3333333333, 93.7777777778))]:
            group = [row for row in curves if row['split'] == split]
            assert group[-1]['atomic_control'] == group[-1]['composition'] == 1.
            assert group[-1]['delta'] == 0.
            for arm in ('atomic_control', 'composition'):
                assert [row[arm] for row in group] == sorted(row[arm] for row in group)
            assert abs(group[31]['atomic_control'] * 100 - expected[0]) < 1e-8
            assert abs(group[31]['composition'] * 100 - expected[1]) < 1e-8
            assert group[31]['sign_flip_p'] == .25
        summary = json.loads((out / 'summary.json').read_text())
        assert summary['main_six_seed_series'] is False
        assert summary['historical_tokenizer_hash_and_revision'] is None
        assert len((out / 'paired_task_deltas.csv').read_text().splitlines()) == 1801


if __name__ == '__main__':
    test_local_saved_ranks_and_paired_seed_curves()
