import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from collections import Counter

from iclr.data import correct_programs, generate, mixture, render_target
import composition_core as core


class DataChecks(unittest.TestCase):
    def test_generation_and_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "data"
            manifest = generate(out, size=20, eval_size=2, atomic_per_op=5, smoke=True)
            splits, seen_states, seen_tasks = {}, set(), set()
            for filename, entry in manifest["files"].items():
                payload = (out / filename).read_bytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), entry["sha256"])
                rows = [json.loads(line) for line in payload.splitlines()]
                splits[filename] = rows
                self.assertEqual(len(rows), entry["rows"])
                for row in rows:
                    programs = correct_programs(row)
                    self.assertEqual(len(programs), row["correct_count"])
                    self.assertTrue(all(core.program_matches_family(p, row["family"]) for p in programs))
                    states = {core.state_fingerprint(row["p"], s) for p in programs
                              for s in core.trajectory(row["start"], p, row["p"])}
                    self.assertFalse(states & seen_states)
                    self.assertNotIn(row["task_id"], seen_tasks)
                    seen_states.update(states)
                    seen_tasks.add(row["task_id"])
            train, atomic = splits["train.jsonl"], splits["atomic_train.jsonl"]
            self.assertEqual(Counter(r["depth"] for r in train), {2: 5, 3: 10, 4: 5})
            last_ids = {r["task_id"] for r in train}
            for replay in (0, .1, .2, .4, 1):
                specs = mixture(train, atomic, 20, replay, 7, "program_trace")
                composition = [s for s in specs if s["kind"] == "composition"]
                self.assertEqual(len(specs), 20)
                self.assertEqual(len(composition), 20 - round(20 * replay))
                if len(composition) % 4 == 0:
                    depths = Counter(s["row"]["depth"] for s in composition)
                    self.assertEqual([depths[2], depths[3], depths[4]],
                                     [len(composition) // 4, len(composition) // 2, len(composition) // 4])
                ids = {s["task_id"] for s in composition}
                self.assertTrue(ids <= last_ids)
                last_ids = ids
                counts = Counter((s["operation"], s["kind"]) for s in specs if s["kind"] != "composition")
                all_counts = [counts[(op, kind)] for op in core.OPS for kind in ("atomic_plan", "atomic_apply")]
                self.assertLessEqual(max(all_counts) - min(all_counts), 1)
                only = mixture(train, atomic, 20, replay, 7, "program_only")
                self.assertEqual([s["task_id"] for s in specs], [s["task_id"] for s in only])
                for full, short in zip(specs, only):
                    self.assertEqual(full["prompt"], short["prompt"])
                    self.assertEqual(full["answer"].split("\nTRACE:")[0], short["answer"])
            self.assertTrue(core.audit_splits({name[:-6]: rows for name, rows in splits.items()})["ok"])
            repeat = Path(tmp) / "repeat"
            self.assertEqual(manifest, generate(repeat, 20, 2, 5, smoke=True))
            with self.assertRaises(FileExistsError):
                generate(out, 20, 2, 5, smoke=True)
            for bad in (-.1, 1.1, float("nan")):
                with self.assertRaises(ValueError):
                    mixture(train, atomic, 20, bad, 7, "program_only")
            repeated = mixture(train, atomic, 40, 0, 7, "program_only")
            self.assertEqual(Counter(s["task_id"] for s in repeated), {r["task_id"]: 2 for r in train})
            self.assertEqual([s["exposure_index"] for s in repeated], list(range(40)))
            with self.assertRaises(ValueError):
                mixture(train, [], 20, .2, 7, "program_only")
            with self.assertRaises(ValueError):
                mixture([], atomic, 20, .2, 7, "program_only")
            with self.assertRaises(ValueError):
                render_target({**train[0], "target": [999]}, "program_trace")

    def test_full_correct_set_is_not_limited_to_two(self):
        row = {"p": 5, "depth": 3, "start": [1, 2, 3], "target": [2, 3, 2]}
        self.assertGreater(len(correct_programs(row)), 2)
        with self.assertRaises(ValueError):
            correct_programs({**row, "family": "TRAIN"})


if __name__ == "__main__":
    unittest.main()
