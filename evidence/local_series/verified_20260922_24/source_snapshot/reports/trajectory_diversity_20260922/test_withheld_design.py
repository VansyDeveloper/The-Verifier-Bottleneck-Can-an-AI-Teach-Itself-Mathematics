import unittest

from withheld_design import build_design


class WithheldDesignTests(unittest.TestCase):
    def test_deterministic_feasible_sets(self):
        rows = []
        operations = ("A", "B", "C", "D", "E")
        for depth in (2, 3, 4):
            for i in range(300):
                program = [operations[(i // (5 ** j)) % 5] for j in range(depth)]
                rows.append({"depth": depth, "program": program})
        design = build_design(rows, ks=(1, 2), subsets_per_k=2,
                              minimum_per_depth=10, seed=4)
        self.assertEqual(design, build_design(rows, ks=(1, 2),
                                               subsets_per_k=2,
                                               minimum_per_depth=10, seed=4))
        self.assertEqual([len(x["sets"]) for x in design["k_groups"]], [2, 2])
        for group in design["k_groups"]:
            for item in group["sets"]:
                self.assertEqual(len(item["pairs"]), group["k"])
                self.assertGreaterEqual(min(item["remaining_by_depth"].values()), 10)


if __name__ == "__main__":
    unittest.main()
