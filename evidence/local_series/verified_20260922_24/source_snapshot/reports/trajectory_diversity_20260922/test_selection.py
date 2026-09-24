import unittest

from selection import select_pair, state_class, validate_pair_budget


class SelectionTests(unittest.TestCase):
    def test_state_class_uses_quarters_with_modulus(self):
        self.assertEqual(state_class(8, [0, 1, 2, 5, 7]), (8, 0, 0, 1, 2, 3))

    def test_state_class_accepts_all_observed_polynomial_degrees(self):
        self.assertEqual(state_class(8, [0, 2, 7]), (8, 0, 1, 3))

    def test_pair_has_same_depth_modulus_and_target_token_quota(self):
        rows = []
        for depth in (2, 3, 4):
            for index in range(6):
                rows.append({"trajectory_id": f"{depth}-{index}", "depth": depth,
                             "p": 23 if index < 3 else 29,
                             "program": ["A"] * depth,
                             "states": [[0] * 5] + [[index] * 5 for _ in range(depth)]})
        lengths = {row["trajectory_id"]: 5 + (int(row["trajectory_id"].split("-")[1]) % 2) for row in rows}
        one = select_pair(rows, lengths, seed=31, per_depth=2)
        two = select_pair(rows, lengths, seed=31, per_depth=2)
        self.assertEqual(one, two)
        self.assertEqual(len(one["random_ids"]), 6)
        self.assertEqual(len(one["diverse_ids"]), 6)
        by_id = {row["trajectory_id"]: row for row in rows}
        def strata(ids):
            return sorted((by_id[item]["depth"], by_id[item]["p"], lengths[item]) for item in ids)
        self.assertEqual(strata(one["random_ids"]), strata(one["diverse_ids"]))

    def test_diverse_arm_improves_feature_score_in_simple_pool(self):
        rows = []
        for index, op in enumerate(("A", "A", "A", "B")):
            rows.append({"trajectory_id": str(index), "depth": 2, "p": 23,
                         "program": [op, op], "states": [[0] * 5, [index + 1] * 5, [0] * 5]})
        result = select_pair(rows, {str(i): 7 for i in range(4)}, seed=0, per_depth=2, depths=(2,))
        self.assertIn("3", result["diverse_ids"])

    def test_budget_validation_rejects_different_target_tokens(self):
        rows = [{"trajectory_id": "a", "depth": 2, "p": 23},
                {"trajectory_id": "b", "depth": 2, "p": 23}]
        with self.assertRaises(ValueError):
            validate_pair_budget(rows, {"a": 5, "b": 6}, ["a"], ["b"])


if __name__ == "__main__":
    unittest.main()
