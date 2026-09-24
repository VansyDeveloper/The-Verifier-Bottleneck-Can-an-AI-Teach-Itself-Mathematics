import sys
import unittest
from pathlib import Path

from holdout import classify_programs, generate_conditioned


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/code"))
import composition_core as core


class HoldoutTests(unittest.TestCase):
    def test_classification_distinguishes_all_some_and_no_pair_solutions(self):
        pairs = {("AX1", "REV")}
        self.assertEqual(classify_programs((("AX1", "REV", "SC2"), ("SH1", "AX1", "REV")), pairs), "withheld")
        self.assertEqual(classify_programs((("AX1", "REV", "SC2"), ("REV", "SC2", "SH1")), pairs), "mixed")
        self.assertEqual(classify_programs((("REV", "SC2", "SH1"),), pairs), "control")

    def test_generated_tasks_are_unique_and_pair_classified(self):
        registry = core.FingerprintRegistry()
        pairs = {("AX1", "REV")}
        withheld = generate_conditioned(core, registry, pairs, kind="withheld", count=2, seed=86001)
        control = generate_conditioned(core, registry, pairs, kind="control", count=2, seed=86002)
        self.assertEqual(len({row["task_id"] for row in withheld + control}), 4)
        for row in withheld:
            search = core.exact_shortest_solutions(row["start"], row["target"], row["p"], 3, limit=125)
            self.assertIn(classify_programs((solution.program for solution in search.solutions), pairs), ("withheld", "mixed"))
            self.assertEqual(core._validate_task_row(row, row["split"], verify_shortest=True)[0], [])
        for row in control:
            search = core.exact_shortest_solutions(row["start"], row["target"], row["p"], 3, limit=125)
            self.assertEqual(classify_programs((solution.program for solution in search.solutions), pairs), "control")
            self.assertEqual(core._validate_task_row(row, row["split"], verify_shortest=True)[0], [])

    def test_withheld_generation_accepts_task_with_alternative_correct_solution(self):
        rows = generate_conditioned(core, core.FingerprintRegistry(), {("SH1", "AC1")},
                                    kind="withheld", count=2, seed=86108, max_attempts=100)
        self.assertEqual(len(rows), 2)
        for row in rows:
            search = core.exact_shortest_solutions(row["start"], row["target"], row["p"], 3, limit=125)
            self.assertIn(classify_programs((solution.program for solution in search.solutions),
                                            {("SH1", "AC1")}), ("withheld", "mixed"))


if __name__ == "__main__":
    unittest.main()
