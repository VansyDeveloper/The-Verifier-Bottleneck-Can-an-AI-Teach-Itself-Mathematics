import unittest

from eval.eval_exploration import aggregate, mix_candidate_rows
from modcomp.exploration import novelty_metrics, outcome_class


class ExplorationTest(unittest.TestCase):
    def test_canonical_outcome_and_novelty(self):
        self.assertEqual(outcome_class("Answer: A=8, B=-1", 7), (1, 6))
        self.assertIsNone(outcome_class("no final answer", 7))

        metrics = novelty_metrics(
            reference=[(1, 1), (1, 1), (2, 2), None],
            candidates=[(1, 1), (3, 3), None, (2, 2)],
            gold=(3, 3),
            resolution_m=2,
        )
        self.assertEqual(metrics["effective_support_size"], 1)
        self.assertEqual(metrics["epsilon_raw"], 0.75)
        self.assertEqual(metrics["epsilon_parseable"], 0.5)
        self.assertEqual(metrics["epsilon_correct"], 0.25)
        self.assertEqual(metrics["novel_precision"], 1 / 3)
        self.assertEqual(metrics["novel_parseable_precision"], 0.5)
        self.assertEqual(metrics["novel_count"], 3)
        self.assertEqual(metrics["novel_parseable_count"], 2)
        self.assertEqual(metrics["novel_correct_count"], 1)
        self.assertEqual(metrics["pass@1"], 0.25)
        self.assertEqual(metrics["pass@k"], 1.0)

    def test_pooled_novel_precision(self):
        rows = [
            novelty_metrics([(0, 0)] * 4, [(1, 1)], (1, 1), 2),
            novelty_metrics([(0, 0)] * 4, [(1, 1)] * 3, (9, 9), 2),
        ]
        self.assertEqual(aggregate(rows)["novel_precision"], 0.25)
        self.assertEqual(aggregate(rows)["novel_parseable_precision"], 0.25)

    def test_iid_mixture_is_reproducible_and_has_exact_endpoints(self):
        base = [["b0", "b1", "b2"]]
        proposal = [["p0", "p1", "p2"]]
        self.assertEqual(mix_candidate_rows(base, proposal, 0, 7), (base, 0.0))
        self.assertEqual(mix_candidate_rows(base, proposal, 1, 7), (proposal, 1.0))
        self.assertEqual(
            mix_candidate_rows(base, proposal, 0.5, 7),
            mix_candidate_rows(base, proposal, 0.5, 7),
        )


if __name__ == "__main__":
    unittest.main()
