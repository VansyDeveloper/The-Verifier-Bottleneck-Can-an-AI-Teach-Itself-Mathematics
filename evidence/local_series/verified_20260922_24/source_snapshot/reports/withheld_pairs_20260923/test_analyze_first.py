import unittest
import hashlib
import tempfile
from pathlib import Path

from analyze_first import branch_prefix, derive_atomic_rows, hit_curve, paired_stats, same_hash, selected_pair_count, tree_digest, validate_ranking


class FirstBlockAnalysisTests(unittest.TestCase):
    def test_dynamic_branch_prefix_keeps_set_identity(self):
        self.assertEqual(branch_prefix(1, 2, 85000, "withheld"),
                         "k1_s2_seed85000_withheld")

    def test_three_positive_pairs_have_discrete_signflip_limit(self):
        stats = paired_stats([0.1, 0.2, 0.3])
        self.assertEqual(stats["n"], 3)
        self.assertAlmostEqual(stats["mean_difference"], 0.2)
        self.assertEqual(stats["exact_signflip_two_sided_p"], 0.25)

    def test_hit_curve_uses_best_rank_for_all_k(self):
        curve = hit_curve([1, 32, 125], 125)
        self.assertEqual(len(curve), 125)
        self.assertEqual(curve[0], 1 / 3)
        self.assertEqual(curve[31], 2 / 3)
        self.assertEqual(curve[-1], 1)

    def test_ranking_rejects_duplicate_program_and_nonfinite_score(self):
        rows = [{"program": ["A"], "rank": 1, "score": 0.0, "correct": True},
                {"program": ["B"], "rank": 2, "score": -1.0, "correct": False}]
        self.assertEqual(validate_ranking(rows, expected_programs=2), 1)
        for bad in ([rows[0], {**rows[1], "program": ["A"]}],
                    [rows[0], {**rows[1], "score": float("nan")}],
                    [rows[0], {**rows[1], "rank": 1}]):
            with self.assertRaises(ValueError):
                validate_ranking(bad, expected_programs=2)

    def test_full_ranking_requires_exact_program_universe(self):
        from itertools import product

        ops = ("SH1", "SC2", "REV", "AC1", "AX1")
        rows = [{"program": list(program), "rank": rank, "score": -float(rank),
                 "correct": rank == 1}
                for rank, program in enumerate(product(ops, repeat=3), 1)]
        self.assertEqual(validate_ranking(rows), 1)
        rows[-1]["program"] = ["BOGUS", "SC2", "REV"]
        with self.assertRaises(ValueError):
            validate_ranking(rows)

    def test_tree_hash_uses_linux_lexical_order_on_windows(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "README.md").write_bytes(b"upper")
            (root / "adapter.json").write_bytes(b"lower")
            expected = hashlib.sha256()
            for name, payload in (("README.md", b"upper"), ("adapter.json", b"lower")):
                expected.update(name.encode())
                expected.update(b"\0")
                expected.update(hashlib.sha256(payload).digest())
            self.assertEqual(tree_digest(root), expected.hexdigest())

    def test_selected_pair_count_uses_adjacent_ordered_pairs(self):
        source = {"a": {"program": ["AX1", "REV", "SH1"]},
                  "b": {"program": ["REV", "AX1", "SH1"]}}
        self.assertEqual(selected_pair_count(source, ["a", "b"], ("AX1", "REV")), 1)

    def test_sha_comparison_accepts_hex_letter_case_only(self):
        self.assertTrue(same_hash("AB12", "ab12"))
        self.assertFalse(same_hash("AB12", "AB13"))

    def test_atomic_rederivation_rejects_false_correctness(self):
        source = [{"task_id": "one", "operation": "SH1"}]
        raw = [{"task_id": "one", "operation": "SH1", "target": "[1]",
                "plan_prediction": "SH1", "plan_correct": True,
                "raw_completion": "[1]", "correct": True}]
        self.assertEqual(derive_atomic_rows(source, raw)["plan"]["SH1"], 1.0)
        with self.assertRaises(ValueError):
            derive_atomic_rows(source, [{**raw[0], "correct": False}])


if __name__ == "__main__":
    unittest.main()
