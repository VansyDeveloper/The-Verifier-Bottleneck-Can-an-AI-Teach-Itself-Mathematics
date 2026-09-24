import tempfile
import unittest
from pathlib import Path

from bundle_first import audit_file_candidates, require_docker_evidence, require_queue_evidence


class BundleFirstTests(unittest.TestCase):
    def test_bundle_includes_frozen_model_code_dependencies(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            expected = (
                "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/code/composition_core.py",
                "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/code/composition_eval.py",
                "artifacts/stage4_distill_v5_0p6b_exploratory_sh1_72p8/code/composition_model.py",
                "artifacts/stage4_sh1_v5_0p6b/code/v5_train.py",
                "artifacts/stage4_distill_v3/code/v3_train.py",
                "artifacts/stage4_distill_v2/code/v2_atomic.py",
                "artifacts/stage4_distill_v1/code/stage4_core.py",
            )
            for relative in expected:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("source")
            candidates = {path.relative_to(root).as_posix() for path in audit_file_candidates(root)}
            for relative in expected:
                self.assertIn(relative, candidates)

    def test_queue_logs_are_required_and_included(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runs = root / "artifacts/withheld_pairs_20260923/runs"
            runs.mkdir(parents=True)
            for name in ("queue_first.jsonl", "queue.stdout.log", "queue.stderr.log"):
                (runs / name).write_text("evidence")
            require_queue_evidence(root / "artifacts/withheld_pairs_20260923")
            candidates = {path.relative_to(root).as_posix() for path in audit_file_candidates(root)}
            for name in ("queue_first.jsonl", "queue.stdout.log", "queue.stderr.log"):
                self.assertIn(f"artifacts/withheld_pairs_20260923/runs/{name}", candidates)
            (runs / "queue.stderr.log").unlink()
            with self.assertRaisesRegex(ValueError, "Queue evidence missing"):
                require_queue_evidence(root / "artifacts/withheld_pairs_20260923")

    def test_all_first_block_docker_logs_are_required(self):
        with tempfile.TemporaryDirectory() as folder:
            artifact = Path(folder)
            logs = artifact / "runs/docker"
            logs.mkdir(parents=True)
            for mode in ("train", "eval"):
                for seed in (85000, 85001, 85002):
                    for arm in ("random", "withheld"):
                        name = f"verifier_withheld_{mode}_20260923_k1s1_{seed}_{arm}"
                        for suffix in ("inspect.json", "stdout.log", "stderr.log"):
                            (logs / f"{name}.{suffix}").write_text("evidence")
            require_docker_evidence(artifact)
            (logs / "verifier_withheld_eval_20260923_k1s1_85002_withheld.stderr.log").unlink()
            with self.assertRaisesRegex(ValueError, "Docker evidence missing"):
                require_docker_evidence(artifact)

    def test_compact_bundle_keeps_receipts_and_excludes_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            adapter = root / "artifacts/withheld_pairs_20260923/adapters/k1_s1_seed85000_random"
            adapter.mkdir(parents=True)
            (adapter / "training_receipt.json").write_text("{}")
            (adapter / "adapter_model.safetensors").write_bytes(b"large")
            later = root / "artifacts/withheld_pairs_20260923/adapters/k1_s2_seed85000_random"
            later.mkdir(parents=True)
            (later / "training_receipt.json").write_text("{}")
            report = root / "reports/withheld_pairs_20260923"
            report.mkdir(parents=True)
            (report / "README.md").write_text("readme")
            candidates = {path.relative_to(root).as_posix() for path in audit_file_candidates(root)}
            self.assertIn("reports/withheld_pairs_20260923/README.md", candidates)
            self.assertIn("artifacts/withheld_pairs_20260923/adapters/k1_s1_seed85000_random/training_receipt.json",
                          candidates)
            self.assertNotIn("artifacts/withheld_pairs_20260923/adapters/k1_s1_seed85000_random/adapter_model.safetensors",
                             candidates)
            self.assertNotIn("artifacts/withheld_pairs_20260923/adapters/k1_s2_seed85000_random/training_receipt.json",
                             candidates)


if __name__ == "__main__":
    unittest.main()
