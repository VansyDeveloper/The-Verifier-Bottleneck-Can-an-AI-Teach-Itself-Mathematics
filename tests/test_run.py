import json
import tempfile
import unittest
from pathlib import Path

from iclr.run import build_jobs, comparison_entries, parser, recipe_cells, write_config


class QueueTest(unittest.TestCase):
    def test_recipes_pair_and_reuse_controls(self):
        args = parser().parse_args(["--recipe", "replay", "trace", "set", "size",
                                   "--data", "bundle", "--base", "atomic", "--model", "Qwen/Qwen3-0.6B",
                                   "--output", "runs", "--seeds", "0", "1", "2"])
        jobs = build_jobs(args)
        self.assertEqual(len(jobs), 30)
        self.assertEqual(len({job["output"] for job in jobs}), 30)
        cells = recipe_cells(["replay", "trace", "set", "size"])
        self.assertEqual([c["replay_fraction"] for c in recipe_cells(["replay"])[:4]], [0, .1, .2, .4])
        self.assertEqual(len([c for c in cells if c["replay_fraction"] == 1.0]), 1)
        set_cells = [c for c in cells if c["depth3_only"]]
        self.assertEqual({c["method"] for c in set_cells}, {"ce", "single_norm", "set_mass"})
        self.assertTrue(all(c["supervision"] == "program" for c in set_cells))
        self.assertTrue(all(c["replay_fraction"] == 0 for c in set_cells))
        entries = comparison_entries(jobs, Path(args.output).resolve())
        self.assertEqual(len(entries), 6)
        self.assertEqual({entry['arm'] for entry in entries}, {'atomic_control', 'composition'})
        self.assertTrue(all(entry['metrics'].startswith('runs/') and
                            entry['metrics'].endswith('/eval/metrics.jsonl') for entry in entries))
        self.assertEqual(comparison_entries([j for j in jobs if j['depth3_only']], Path(args.output).resolve()), [])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            write_config(path, jobs[0])
            write_config(path, jobs[0])
            with self.assertRaises(ValueError):
                write_config(path, {**jobs[0], "epochs": 3})
            self.assertEqual(json.loads(path.read_text()), jobs[0])
        args.seeds = [0, 0]
        with self.assertRaises(ValueError):
            build_jobs(args)


if __name__ == "__main__":
    unittest.main()
