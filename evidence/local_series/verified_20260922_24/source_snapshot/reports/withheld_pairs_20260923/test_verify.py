import hashlib
import tempfile
import unittest
from pathlib import Path

from verify import validate_holdout_file


class HoldoutIntegrityTests(unittest.TestCase):
    def test_missing_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "missing"):
                validate_holdout_file(Path(directory), {"path": "missing.jsonl", "sha256": "X", "rows": 2})

    def test_changed_sha_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "holdout.jsonl"
            target.write_bytes(b'{"task_id":"a"}\n')
            with self.assertRaisesRegex(ValueError, "SHA"):
                validate_holdout_file(Path(directory), {"path": "holdout.jsonl", "sha256": "X", "rows": 1})

    def test_duplicate_task_is_rejected_even_with_matching_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "holdout.jsonl"
            payload = b'{"task_id":"a"}\n{"task_id":"a"}\n'
            target.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                validate_holdout_file(Path(directory), {"path": "holdout.jsonl",
                                                        "sha256": hashlib.sha256(payload).hexdigest().upper(), "rows": 2})


if __name__ == "__main__":
    unittest.main()
