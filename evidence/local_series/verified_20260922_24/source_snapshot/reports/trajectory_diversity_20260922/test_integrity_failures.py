import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from analysis import audit_arm, analyze_ranking
from selection import validate_pair_budget


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_row(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write(json.dumps(value) + "\n")
    else:
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class IntegrityFailureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.branch = "replicate0_atomic_control_final_a"
        self.base = self.root / "rankings/replicate0"
        _write_json(self.base / "status" / f"{self.branch}.json",
                    {"status": "DONE", "tasks": 1000})
        for shard in range(40):
            self._shard(shard, shard)

    def _shard(self, shard, task_number, *, bad_hit32=False):
        stem = f"part-{shard:05d}"
        task = f"task{task_number}"
        metrics_dir = self.base / "metrics" / self.branch
        ranking_dir = self.base / "rankings" / self.branch
        metrics_path = metrics_dir / f"{stem}.jsonl"
        ranking_path = ranking_dir / f"{stem}.jsonl.gz"
        ranking = [{"program": [str(i)], "rank": i + 1,
                    "score": 0.0, "correct": i == 0} for i in range(125)]
        metric = {"task_id": task, "task_fingerprint": task, "split": "final_a",
                  "depth": 3, "best_rank": 1, "correct_mass": 1 / 125,
                  **{f"hit@{k}": 1.0 for k in (1, 8, 16, 32, 64)}}
        if bad_hit32:
            metric["hit@32"] = 0.0
        raw = {"task_id": task, "task_fingerprint": task,
               "split": "final_a", "depth": 3, "ranking": ranking}
        _write_row(metrics_path, metric)
        _write_row(ranking_path, raw)
        _write_json(metrics_dir / f"{stem}.receipt.json",
                    {"status": "DONE", "metrics_sha256": _sha(metrics_path),
                     "ranking_sha256": _sha(ranking_path), "metrics_rows": 1,
                     "ranking_rows": 1, "task_ids": [task]})

    def test_missing_shard_is_rejected(self):
        (self.base / "metrics" / self.branch / "part-00039.receipt.json").unlink()
        with self.assertRaisesRegex(ValueError, "expected 40 receipts"):
            audit_arm(self.root, 0, "atomic_control")

    def test_wrong_sha_is_rejected(self):
        path = self.base / "metrics" / self.branch / "part-00000.receipt.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["ranking_sha256"] = "0" * 64
        _write_json(path, value)
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            audit_arm(self.root, 0, "atomic_control")

    def test_duplicate_task_is_rejected(self):
        self._shard(1, 0)
        with self.assertRaisesRegex(ValueError, "duplicate task"):
            audit_arm(self.root, 0, "atomic_control")

    def test_wrong_hit32_is_rejected(self):
        self._shard(0, 0, bad_hit32=True)
        with self.assertRaisesRegex(ValueError, "saved hit@32 mismatch"):
            audit_arm(self.root, 0, "atomic_control")

    def test_nonnumeric_score_is_rejected(self):
        ranking = [{"program": ["A"], "rank": 1, "score": "not-a-number",
                    "correct": True}]
        with self.assertRaises(ValueError):
            analyze_ranking(ranking, expected_count=1)

    def test_unequal_training_budget_is_rejected(self):
        rows = [{"trajectory_id": "a", "depth": 2, "p": 7},
                {"trajectory_id": "b", "depth": 2, "p": 7}]
        with self.assertRaisesRegex(ValueError, "different depth, modulus, or target-token strata"):
            validate_pair_budget(rows, {"a": 10, "b": 11}, ["a"], ["b"])


if __name__ == "__main__":
    unittest.main()
