import math
import unittest

from grpo_entropy import group_entropy


class GroupEntropyTests(unittest.TestCase):
    def test_all_identical_answers_have_zero_entropy_and_full_collision(self):
        rows = [{"parsed_program": ["A"], "text": "A", "is_correct": False} for _ in range(8)]
        got = group_entropy(rows)
        self.assertEqual(got["unique_answers"], 1)
        self.assertAlmostEqual(got["sample_entropy_nats"], 0.0)
        self.assertAlmostEqual(got["collision_fraction"], 1.0)
        self.assertIsNone(got["correct_sample_entropy_nats"])

    def test_eight_distinct_answers_have_log_eight_entropy(self):
        rows = [{"parsed_program": [str(index)], "text": str(index), "is_correct": index < 2} for index in range(8)]
        got = group_entropy(rows)
        self.assertEqual(got["unique_answers"], 8)
        self.assertAlmostEqual(got["sample_entropy_nats"], math.log(8))
        self.assertAlmostEqual(got["collision_fraction"], 0.0)
        self.assertEqual(got["correct_count"], 2)
        self.assertAlmostEqual(got["correct_sample_entropy_nats"], math.log(2))

    def test_invalid_programs_remain_distinct_by_text(self):
        rows = [{"parsed_program": None, "text": f"bad-{index}", "is_correct": False} for index in range(8)]
        self.assertEqual(group_entropy(rows)["unique_answers"], 8)


if __name__ == "__main__":
    unittest.main()
