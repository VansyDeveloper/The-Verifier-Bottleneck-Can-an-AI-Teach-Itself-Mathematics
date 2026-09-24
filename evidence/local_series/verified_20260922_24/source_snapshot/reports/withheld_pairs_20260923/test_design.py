import unittest

from design import build_all_selections, build_selection


class MatchedExclusionTests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        self.lengths = {}
        for depth in (2, 3, 4):
            for p in (101, 103):
                for index in range(30):
                    identity = f"{depth}-{p}-{index}"
                    program = ["AX1", "REV"] + ["SC2"] * (depth - 2) if index < 5 else ["REV", "SC2"] + ["AC1"] * (depth - 2)
                    self.rows.append({"trajectory_id": identity, "depth": depth, "p": p, "program": program})
                    self.lengths[identity] = 12 + index % 2

    def test_random_removal_matches_withheld_counts_and_training_budget(self):
        result = build_selection(self.rows, self.lengths, {("AX1", "REV")}, seed=85000, per_depth=10)
        self.assertEqual(result["removed_by_depth_p"], {"2:101": 5, "2:103": 5, "3:101": 5, "3:103": 5, "4:101": 5, "4:103": 5})
        self.assertEqual(len(result["withheld_ids"]), 30)
        self.assertEqual(len(result["random_ids"]), 30)
        self.assertTrue(set(result["withheld_ids"]).isdisjoint(result["withheld_removed_ids"]))
        self.assertTrue(set(result["random_ids"]).isdisjoint(result["random_removed_ids"]))
        self.assertEqual(result["budget"]["records_each"], 30)
        self.assertEqual(result["budget"]["target_tokens_per_epoch_each"], sum(self.lengths[x] for x in result["withheld_ids"]))
        self.assertEqual(result, build_selection(self.rows, self.lengths, {("AX1", "REV")}, seed=85000, per_depth=10))

    def test_insufficient_matched_source_fails_without_partial_selection(self):
        with self.assertRaisesRegex(ValueError, "insufficient matched pool"):
            build_selection(self.rows, self.lengths, {("AX1", "REV")}, seed=85000, per_depth=51)

    def test_infeasible_preselected_set_is_recorded_not_replaced(self):
        design = {"k_groups": [{"k": 1, "sets": [{"subset": 1, "pairs": [["AX1", "REV"]]}]}]}
        result = build_all_selections(self.rows, self.lengths, design, seeds=(85000, 85001), per_depth=51)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["status"], "INFEASIBLE")
        self.assertEqual(result[0]["k"], 1)
        self.assertEqual(result[0]["subset"], 1)
        self.assertIn("insufficient matched pool", result[0]["reason"])


if __name__ == "__main__":
    unittest.main()
