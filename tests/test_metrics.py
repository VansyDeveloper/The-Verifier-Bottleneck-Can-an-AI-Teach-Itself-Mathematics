import unittest

from modcomp.metrics import base_solved_summary, summarize_checker_counts


class ForgettingMetricsTest(unittest.TestCase):
    def test_base_solved_forgetting(self):
        selection = {
            0: {"correct": 1, "n": 2},
            1: {"correct": 2, "n": 2},
            2: {"correct": 0, "n": 2},
        }
        baseline = {
            0: {"correct": 2, "n": 2},
            1: {"correct": 1, "n": 2},
            2: {"correct": 0, "n": 2},
        }
        final = {
            0: {"correct": 0, "n": 2},
            1: {"correct": 1, "n": 2},
            2: {"correct": 1, "n": 2},
        }
        result = base_solved_summary(selection, baseline, final)
        self.assertEqual(result["base_solved_count"], 2)
        self.assertEqual(result["base_solved_nonrediscovery_count"], 1)
        self.assertEqual(result["base_solved_nonrediscovery_rate"], 0.5)
        self.assertEqual(result["base_pass@1"], 0.5)
        self.assertEqual(result["final_pass@1"], 1 / 3)
        self.assertEqual(result["base_pass@1_on_base_solved"], 0.75)
        self.assertEqual(result["final_pass@1_on_base_solved"], 0.25)
        self.assertEqual(result["pass@1_change_on_base_solved"], -0.5)

    def test_checker_diagnostics_preserve_signal_sign(self):
        result = summarize_checker_counts(
            {"tp": 80, "fn": 20, "fp": 30, "tn": 70, "groups": 10, "zero_variance_groups": 2}
        )
        self.assertEqual(result["realized_tpr"], 0.8)
        self.assertEqual(result["realized_fpr"], 0.3)
        self.assertEqual(result["realized_signed_alignment"], 0.5)
        self.assertEqual(result["acceptance_rate"], 0.55)
        self.assertEqual(result["zero_variance_group_rate"], 0.2)
        self.assertGreater(result["mutual_information_bits"], 0)


if __name__ == "__main__":
    unittest.main()
