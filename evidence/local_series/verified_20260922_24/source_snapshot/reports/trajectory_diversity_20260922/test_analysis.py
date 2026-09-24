import math
import unittest

from analysis import analyze_ranking, hit_curve


class RankingAnalysisTests(unittest.TestCase):
    def test_hit_curve(self):
        self.assertEqual(hit_curve([1, 3, 3], 3), [1 / 3, 1 / 3, 1.0])

    def test_entropy_and_correct_mass(self):
        ranking = [
            {"program": ["A"], "rank": 1, "score": 0.0, "correct": True},
            {"program": ["B"], "rank": 2, "score": 0.0, "correct": False},
        ]
        got = analyze_ranking(ranking, 2)
        self.assertEqual(got["best_rank"], 1)
        self.assertAlmostEqual(got["entropy_nats"], math.log(2))
        self.assertAlmostEqual(got["effective_programs"], 2.0)
        self.assertAlmostEqual(got["correct_mass"], 0.5)
        self.assertAlmostEqual(got["correct_entropy_nats"], 0.0)

    def test_rejects_duplicate_program(self):
        ranking = [
            {"program": ["A"], "rank": 1, "score": 0.0, "correct": True},
            {"program": ["A"], "rank": 2, "score": -1.0, "correct": True},
        ]
        with self.assertRaises(ValueError):
            analyze_ranking(ranking, 2)

    def test_rejects_wrong_order(self):
        ranking = [
            {"program": ["A"], "rank": 1, "score": -1.0, "correct": True},
            {"program": ["B"], "rank": 2, "score": 0.0, "correct": False},
        ]
        with self.assertRaises(ValueError):
            analyze_ranking(ranking, 2)


if __name__ == "__main__":
    unittest.main()
