import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "code" / "stage4.py"
SPEC = importlib.util.spec_from_file_location("stage4_atomic_negative_report", PATH)
STAGE4 = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(STAGE4)


def test_paired_binary_statistics_are_exact_and_deterministic():
    before = [True, True, False, False]
    after = [True, False, True, True]
    first = STAGE4.paired_binary_statistics(before, after, repetitions=200, seed=17)
    second = STAGE4.paired_binary_statistics(before, after, repetitions=200, seed=17)
    assert first == second
    assert first["before_only"] == 1 and first["after_only"] == 2
    assert first["delta"] == 0.25
    assert first["exact_mcnemar_p_two_sided"] == 1.0
    assert first["paired_bootstrap"]["ci95"][0] <= first["delta"] <= first["paired_bootstrap"]["ci95"][1]


@pytest.mark.parametrize(("before", "after"), (([], []), ([True], [])))
def test_paired_binary_statistics_reject_unaligned_inputs(before, after):
    with pytest.raises(RuntimeError, match="non-empty and aligned"):
        STAGE4.paired_binary_statistics(before, after, repetitions=10)
