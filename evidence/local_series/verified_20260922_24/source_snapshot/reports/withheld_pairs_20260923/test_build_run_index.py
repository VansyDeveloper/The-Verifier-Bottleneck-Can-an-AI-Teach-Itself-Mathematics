import csv
import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import build_run_index as index
from build_run_index import build_index, load_audited_sets


class RunIndexTests(unittest.TestCase):
    def test_frozen_design_has_120_rows_and_six_infeasible(self):
        root = Path(__file__).resolve().parents[2] / "artifacts/withheld_pairs_20260923"
        freeze = json.loads((root / "FREEZE.json").read_text(encoding="utf-8"))
        rows = build_index(freeze, root)
        self.assertEqual(len(rows), 120)
        self.assertEqual(sum(row["training_status"] == "DESIGN_INFEASIBLE" for row in rows), 6)
        self.assertEqual({row["training_status"] for row in rows if row["k"] == 5 and row["subset"] == 5},
                         {"DESIGN_INFEASIBLE"})

    def test_independently_audited_set_counts_all_six_rows_without_raw_local_copy(self):
        root = Path(__file__).resolve().parents[2] / "artifacts/withheld_pairs_20260923"
        freeze = json.loads((root / "FREEZE.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            empty_artifacts = Path(directory)
            rows = build_index(freeze, empty_artifacts, {(1, 2): {
                "status": "VALID", "k": 1, "subset": 2,
                "completed_trainings": 6, "completed_evaluations": 6,
            }})
        selected = [row for row in rows if row["k"] == 1 and row["subset"] == 2]
        self.assertEqual(len(selected), 6)
        self.assertTrue(all(row[key] == "DONE" for row in selected for key in
                            ("training_status", "withheld_evaluation_status",
                             "control_evaluation_status", "atomic_evaluation_status")))

    def test_remote_queue_snapshot_keeps_unarchived_work_visible_without_accepting_it(self):
        root = Path(__file__).resolve().parents[2] / "artifacts/withheld_pairs_20260923"
        freeze = json.loads((root / "FREEZE.json").read_text(encoding="utf-8"))
        events = [
            {"job": "verifier_withheld_train_20260923_k2s2_85000_random", "status": "STARTED"},
            {"job": "verifier_withheld_train_20260923_k2s2_85000_random", "status": "DONE"},
            {"job": "verifier_withheld_eval_20260923_k2s2_85000_random", "status": "STARTED"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            rows = build_index(freeze, Path(directory), remote_events=events)
        row = next(item for item in rows if (item["k"], item["subset"], item["seed"], item["arm"]) ==
                   (2, 2, 85000, "random"))
        self.assertEqual(row["training_status"], "NOT_STARTED")
        self.assertEqual(row["withheld_evaluation_status"], "NOT_STARTED")
        self.assertEqual(row["remote_training_status"], "DONE")
        self.assertEqual(row["remote_evaluation_status"], "STARTED")
        self.assertEqual(len(rows), 120)

    def test_unknown_remote_job_cannot_contaminate_frozen_matrix(self):
        root = Path(__file__).resolve().parents[2] / "artifacts/withheld_pairs_20260923"
        freeze = json.loads((root / "FREEZE.json").read_text(encoding="utf-8"))
        event = {"job": "verifier_withheld_train_20260923_k9s9_85000_random", "status": "DONE"}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "outside frozen design"):
                build_index(freeze, Path(directory), remote_events=[event])

    def test_queue_snapshot_loader_reads_complete_jsonl_without_dropping_events(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "queue_k2s2.jsonl"
            first.write_text(
                '{"job":"verifier_withheld_train_20260923_k2s2_85000_random","status":"STARTED"}\n'
                '{"status":"SET_DONE","k":2,"subset":2}\n', encoding="utf-8")
            events = index.load_queue_logs([first])
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["status"], "STARTED")
        self.assertEqual(events[1]["status"], "SET_DONE")

    def test_queue_snapshot_loader_rejects_two_hosts_for_one_set(self):
        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "ccmplanner_k2s2.jsonl"
            stale = Path(directory) / "cdsserver_k2s2.jsonl"
            current.write_text(
                '{"job":"verifier_withheld_train_20260923_k2s2_85000_random","status":"DONE"}\n',
                encoding="utf-8")
            stale.write_text(
                '{"job":"verifier_withheld_train_20260923_k2s2_85000_random","status":"STARTED"}\n',
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate remote set"):
                index.load_queue_logs([current, stale])

    def test_cli_writes_snapshot_matrix_to_requested_file(self):
        with tempfile.TemporaryDirectory() as directory:
            isolated = Path(directory)
            script = isolated / "reports/withheld_pairs_20260923/build_run_index.py"
            script.parent.mkdir(parents=True)
            script.write_bytes(Path(index.__file__).read_bytes())
            freeze = isolated / "artifacts/withheld_pairs_20260923/FREEZE.json"
            freeze.parent.mkdir(parents=True)
            freeze.write_bytes((index.ROOT / "FREEZE.json").read_bytes())
            queue = Path(directory) / "queue_k2s2.jsonl"
            output = Path(directory) / "run_index.csv"
            queue.write_text(
                '{"job":"verifier_withheld_train_20260923_k2s2_85000_random","status":"DONE"}\n',
                encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(script), "--queue-log", str(queue),
                 "--out", str(output)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 120)
        row = next(item for item in rows if (item["k"], item["subset"], item["seed"], item["arm"]) ==
                   ("2", "2", "85000", "random"))
        self.assertEqual(row["remote_training_status"], "DONE")
        self.assertEqual(row["training_status"], "NOT_STARTED")

    def test_archive_summary_and_sidecar_are_required_before_index_override(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "reports/withheld_pairs_20260923"
            data = report / "data/k1_s2"
            data.mkdir(parents=True)
            summary = {"status": "VALID", "k": 1, "subset": 2,
                       "freeze_sha256": "FROZEN", "completed_trainings": 6,
                       "completed_evaluations": 6}
            summary_path = data / "analysis_summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            archive = data / "k1_s2_audit.zip"
            member = "reports/withheld_pairs_20260923/data/k1_s2/analysis_summary.json"
            payload = summary_path.read_bytes()
            manifest = [{"path": member, "sha256": hashlib.sha256(payload).hexdigest().upper(),
                         "bytes": len(payload)}]
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr(member, payload)
                handle.writestr("AUDIT_MANIFEST.json", json.dumps(manifest))
                handle.writestr("CHECKPOINT_INVENTORY.json", json.dumps([
                    {"weights_in_audit_zip": False} for _ in range(6)]))
            sidecar = data / "k1_s2_audit.zip.sha256"
            sidecar.write_text(f"{hashlib.sha256(archive.read_bytes()).hexdigest().upper()}  {archive.name}\n")
            self.assertEqual(load_audited_sets(report, "FROZEN"), {(1, 2): summary})
            sidecar.write_text("BAD  k1_s2_audit.zip\n")
            with self.assertRaisesRegex(ValueError, "archive SHA"):
                load_audited_sets(report, "FROZEN")


if __name__ == "__main__":
    unittest.main()
