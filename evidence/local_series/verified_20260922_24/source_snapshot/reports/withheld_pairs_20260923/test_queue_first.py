import unittest

from queue_first import JOBS, validate_training_receipt, gpu0_available


class QueueFirstTests(unittest.TestCase):
    def test_fixed_order_has_three_complete_pairs_before_evaluations(self):
        self.assertEqual(len(JOBS), 12)
        self.assertEqual([job[0] for job in JOBS], ["train"] * 6 + ["evaluate"] * 6)
        for offset, seed in enumerate((85000, 85001, 85002)):
            self.assertEqual(JOBS[2 * offset:2 * offset + 2],
                             [("train", seed, "random"), ("train", seed, "withheld")])

    def test_receipt_requires_complete_finite_fp16_training(self):
        receipt = {"status": "DONE", "optimizer_steps": 30,
                   "optimizer_steps_expected": 30, "mean_loss": 0.8,
                   "runtime_compute_dtype": "torch.float16"}
        validate_training_receipt(receipt)
        for change in ({"status": "FAILED"}, {"optimizer_steps": 29},
                       {"mean_loss": float("nan")}, {"runtime_compute_dtype": "torch.bfloat16"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_training_receipt({**receipt, **change})

    def test_gpu0_must_be_idle(self):
        self.assertTrue(gpu0_available("0, 3 MiB\n1, 8845 MiB\n"))
        self.assertFalse(gpu0_available("0, 6000 MiB\n1, 8845 MiB\n"))
        with self.assertRaises(ValueError):
            gpu0_available("1, 3 MiB\n")


if __name__ == "__main__":
    unittest.main()
