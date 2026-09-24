import unittest

from plot_first import aggregate_curve


class PlotFirstTests(unittest.TestCase):
    def test_curve_requires_all_three_seeds_at_each_k(self):
        rows = [{"seed": str(seed), "arm": "random", "split": "withheld",
                 "k": str(k), "hit_k": str(value)}
                for k, value in ((1, 0.1), (2, 0.2)) for seed in (85000, 85001, 85002)]
        self.assertEqual(aggregate_curve(rows, "random", "withheld", maximum=2),
                         [(1, 0.1, 0.1, 0.1), (2, 0.2, 0.2, 0.2)])
        with self.assertRaises(ValueError):
            aggregate_curve(rows[:-1], "random", "withheld", maximum=2)


if __name__ == "__main__":
    unittest.main()
