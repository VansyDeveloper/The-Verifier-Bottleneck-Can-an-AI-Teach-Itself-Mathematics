import unittest

from run_first import validate_pair_receipts


class PairAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.left = {"status": "DONE", "optimizer_steps": 30, "effective_batch": 64,
                     "epochs": 2, "records": 938, "loss_bearing_target_tokens_seen": 87500,
                     "mean_loss": 0.4, "runtime_compute_dtype": "torch.float16",
                     "atomic_export_sha256": "abc"}
        self.right = dict(self.left)

    def test_complete_matched_pair_is_accepted(self):
        self.assertEqual(validate_pair_receipts(self.left, self.right)["optimizer_steps"], 30)

    def test_partial_pair_is_rejected(self):
        self.right["status"] = "FAILED"
        with self.assertRaisesRegex(ValueError, "DONE"):
            validate_pair_receipts(self.left, self.right)

    def test_token_budget_mismatch_is_rejected(self):
        self.right["loss_bearing_target_tokens_seen"] += 1
        with self.assertRaisesRegex(ValueError, "budget"):
            validate_pair_receipts(self.left, self.right)

    def test_missing_export_binding_is_rejected(self):
        del self.right["atomic_export_sha256"]
        with self.assertRaisesRegex(ValueError, "export"):
            validate_pair_receipts(self.left, self.right)


if __name__ == "__main__":
    unittest.main()
