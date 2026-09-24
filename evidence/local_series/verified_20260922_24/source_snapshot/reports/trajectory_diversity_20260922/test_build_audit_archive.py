import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from build_audit_archive import build_archive, selected_files


class AuditArchiveTests(unittest.TestCase):
    def test_selection_keeps_evidence_and_excludes_model_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "reports" / "study"
            artifact = root / "artifacts" / "study"
            (report / "data").mkdir(parents=True)
            (report / "__pycache__").mkdir()
            (artifact / "rankings").mkdir(parents=True)
            (artifact / "adapters" / "seed1_random").mkdir(parents=True)
            (report / "REPORT.md").write_text("result", encoding="utf-8")
            (report / "data" / "summary.json").write_text("{}", encoding="utf-8")
            (report / "__pycache__" / "cache.pyc").write_bytes(b"cache")
            (report / "old.zip").write_bytes(b"zip")
            (report / "trajectory_diversity_audit.receipt.json").write_text(
                '{"sha256":"stale"}', encoding="utf-8"
            )
            (artifact / "rankings" / "part.jsonl").write_text("{}\n", encoding="utf-8")
            (artifact / "adapters" / "seed1_random" / "training_receipt.json").write_text("{}", encoding="utf-8")
            (artifact / "adapters" / "seed1_random" / "adapter_model.safetensors").write_bytes(b"weights")

            paths = {item.archive_path for item in selected_files(root, report, artifact)}

            self.assertIn("reports/study/REPORT.md", paths)
            self.assertIn("artifacts/study/rankings/part.jsonl", paths)
            self.assertIn("artifacts/study/adapters/seed1_random/training_receipt.json", paths)
            self.assertNotIn("artifacts/study/adapters/seed1_random/adapter_model.safetensors", paths)
            self.assertFalse(any("__pycache__" in path for path in paths))
            self.assertFalse(any(path.endswith(".zip") for path in paths))
            self.assertNotIn(
                "reports/study/trajectory_diversity_audit.receipt.json", paths
            )

    def test_archive_manifest_hashes_every_included_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "reports" / "study"
            artifact = root / "artifacts" / "study"
            report.mkdir(parents=True)
            artifact.mkdir(parents=True)
            (report / "REPORT.md").write_text("result", encoding="utf-8")
            (artifact / "SELECTION_FROZEN.json").write_text("{}", encoding="utf-8")
            output = root / "audit.zip"

            receipt = build_archive(root, report, artifact, output)

            self.assertEqual(receipt["status"], "DONE")
            with zipfile.ZipFile(output) as archive:
                names = set(archive.namelist())
                self.assertIn("ARCHIVE_MANIFEST.json", names)
                manifest = json.loads(archive.read("ARCHIVE_MANIFEST.json"))
                self.assertEqual(manifest["status"], "DONE")
                self.assertEqual({row["path"] for row in manifest["files"]}, names - {"ARCHIVE_MANIFEST.json"})
                self.assertTrue(all(len(row["sha256"]) == 64 for row in manifest["files"]))


if __name__ == "__main__":
    unittest.main()
