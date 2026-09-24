import unittest
from unittest.mock import patch

import torch

from fp16_training import fp16_runtime


class FP16RuntimeTests(unittest.TestCase):
    def test_rejects_non_v100_gpu(self):
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "get_device_name", return_value="RTX 5060 Ti"):
            with self.assertRaisesRegex(RuntimeError, "V100"):
                fp16_runtime({"dtype": "float16"})

    def test_v100_uses_fp16_compute_and_fp32_master(self):
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "get_device_name", return_value="Tesla V100-SXM2-32GB"):
            receipt = fp16_runtime({"dtype": "float16"})
        self.assertEqual(receipt["runtime_compute_dtype"], "torch.float16")
        self.assertEqual(receipt["runtime_master_dtype"], "torch.float32")

    def test_bfloat16_protocol_is_not_silently_accepted(self):
        with self.assertRaisesRegex(RuntimeError, "float16"):
            fp16_runtime({"dtype": "bfloat16"})


if __name__ == "__main__":
    unittest.main()
