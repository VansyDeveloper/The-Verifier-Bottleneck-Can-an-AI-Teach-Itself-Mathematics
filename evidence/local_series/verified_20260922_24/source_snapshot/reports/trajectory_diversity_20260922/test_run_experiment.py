import unittest

from run_experiment import validate_training_pair


class TrainingPairTests(unittest.TestCase):
    def test_rejects_different_seen_target_tokens(self):
        common = {"status": "DONE", "optimizer_steps": 30, "effective_batch": 64,
                  "epochs": 2, "records": 938, "loss_bearing_target_tokens_seen": 1000,
                  "mean_loss": 0.5}
        different = {**common, "loss_bearing_target_tokens_seen": 999}
        with self.assertRaises(ValueError):
            validate_training_pair(common, different)

    def test_accepts_equal_receipts(self):
        common = {"status": "DONE", "optimizer_steps": 30, "effective_batch": 64,
                  "epochs": 2, "records": 938, "loss_bearing_target_tokens_seen": 1000,
                  "mean_loss": 0.5}
        self.assertEqual(validate_training_pair(common, dict(common))["loss_bearing_target_tokens_seen"], 1000)


if __name__ == "__main__":
    unittest.main()
