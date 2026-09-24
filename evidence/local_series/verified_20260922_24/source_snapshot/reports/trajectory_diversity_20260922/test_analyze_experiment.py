import unittest

from analyze_experiment import paired_stats


class PairedStatisticsTests(unittest.TestCase):
    def test_all_positive_six_differences_have_exact_signflip_p_003125(self):
        got = paired_stats([1, 1, 1, 1, 1, 1])
        self.assertEqual(got["n"], 6)
        self.assertEqual(got["mean_difference"], 1.0)
        self.assertEqual(got["exact_signflip_two_sided_p"], 0.03125)

    def test_zero_differences_cannot_be_significant(self):
        got = paired_stats([0, 0, 0, 0, 0, 0])
        self.assertEqual(got["exact_signflip_two_sided_p"], 1.0)
        self.assertEqual(got["ci95_low"], 0.0)
        self.assertEqual(got["ci95_high"], 0.0)


if __name__ == "__main__":
    unittest.main()
