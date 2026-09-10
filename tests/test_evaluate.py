from iclr.evaluate import family_summaries


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
