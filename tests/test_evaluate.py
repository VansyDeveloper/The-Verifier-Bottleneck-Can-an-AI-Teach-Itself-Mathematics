import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from iclr.evaluate import family_summaries, main


def test_cli_loads_and_records_requested_dtype():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        base, data, output = root / 'base', root / 'data', root / 'eval'
        base.mkdir()
        data.mkdir()
        (base / 'config.json').write_text('{}')
        (data / 'manifest.json').write_text('{"files": {}}')
        command = ['--base', str(base), '--data', str(data), '--out', str(output),
                   '--device', 'cuda', '--dtype', 'bfloat16']
        with patch('iclr.evaluate.load', return_value=(None, None, None)) as load, \
                patch('iclr.evaluate.evaluate'):
            main(command)
            main(command)
            load.assert_called_once_with(str(base), adapter=None, device='cuda', dtype='bfloat16')
        with pytest.raises(ValueError, match='different inputs'):
            main(command + ['--dtype', 'float32'])
        with pytest.raises(SystemExit):
            main(command + ['--device', 'cpu'])
        assert json.loads((output / 'DONE').read_text())['binding']['dtype'] == 'bfloat16'


def test_diversity_summary_only_counts_multiple_correct_programs():
    rows = [dict(family=family, **{'hit@32': 1., 'correct_mass': .5, 'mrr': .5}, correct_count=1,
                 correct_conditional_entropy=0., correct_effective_count=1., max_correct_conditional_probability=1.)
            for family in 'ABCD']
    rows += [{**rows[0], 'correct_count': 2, 'correct_conditional_entropy': .4,
              'correct_effective_count': 1.4, 'max_correct_conditional_probability': .8},
             {**rows[0], 'correct_count': 3, 'correct_conditional_entropy': .8,
              'correct_effective_count': 2.4, 'max_correct_conditional_probability': .6}]
    result = family_summaries(rows)
    assert result['A']['n_multisolution'] == 2
    assert abs(result['A']['multisolution_correct_conditional_entropy'] - .6) < 1e-12
    assert result['A']['multisolution_correct_effective_count'] == 1.9
    assert result['A']['multisolution_max_correct_conditional_probability'] == .7
    assert result['B']['n_multisolution'] == 0
    assert result['B']['multisolution_correct_conditional_entropy'] is None
    assert result['B']['multisolution_correct_effective_count'] is None
    assert result['B']['multisolution_max_correct_conditional_probability'] is None
