import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from iclr.run import ablation_comparisons, build_jobs, comparison_entries, completed_jobs, main, parser, recipe_cells, update_comparison, write_config


class QueueTest(unittest.TestCase):
    def test_dtype_reaches_training_and_baseline(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()) as stdout:
            command = ['--recipe', 'size', '--data', 'data', '--base', 'base', '--model', 'large-model',
                       '--output', directory, '--seeds', '0', '--dtype', 'bfloat16', '--device', 'cuda']
            main(command)
            main(command)
            configs = [json.loads(path.read_text()) for path in (Path(directory) / 'configs').glob('*.json')]
            self.assertTrue(configs and all(config['dtype'] == 'bfloat16' for config in configs))
            baseline = next(line for line in stdout.getvalue().splitlines() if 'iclr.evaluate' in line)
            self.assertIn('--dtype bfloat16', baseline)
            self.assertIn('After training: ', stdout.getvalue())
            self.assertIn('comparison_large-model.json', stdout.getvalue())
            with self.assertRaises(SystemExit):
                main(command + ['--device', 'cpu'])
            with self.assertRaises(SystemExit):
                main(command + ['--dtype', 'float32'])

    def test_extend_seed_queue_without_changing_existing_jobs(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            output = Path(directory)
            command = ['--recipe', 'size', '--data', 'data', '--base', 'base', '--model', 'Qwen/Qwen3-0.6B',
                       '--output', directory, '--seeds']
            main(command + ['0'])
            original = {path: path.read_bytes() for path in (output / 'configs').glob('*.json')}
            main(command + ['0', '1', '2'])
            comparison = output / 'comparison_Qwen_Qwen3-0.6B.json'
            entries = json.loads(comparison.read_text())
            self.assertEqual(len(entries), 6)
            self.assertTrue(all(path.read_bytes() == content for path, content in original.items()))
            main(command + ['0'])
            self.assertEqual(json.loads(comparison.read_text()), entries)
            with self.assertRaises(ValueError):
                update_comparison(comparison, [{**entries[0], 'metrics': 'different.jsonl'}])
            args = parser().parse_args(command + ['3', '--epochs', '3'])
            for job in build_jobs(args):
                write_config(output / 'configs' / f"{Path(job['output']).name}.json", job)
            with self.assertRaises(ValueError):
                update_comparison(comparison, comparison_entries(build_jobs(args), output))
            self.assertEqual(json.loads(comparison.read_text()), entries)

    def test_completed_summary_keeps_previous_recipes_and_excludes_other_bases(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            reference = {'model': 'Qwen', 'base': 'base', 'data': 'data'}
            for index, fields in enumerate(({}, {'method': 'set_mass'}, {'base': 'other'}, {'unfinished': True})):
                job = {**reference, **fields, 'output': str(output / 'runs' / str(index))}
                write_config(output / 'configs' / f'{index}.json', job)
                Path(job['output']).mkdir(parents=True)
                if not fields.get('unfinished'):
                    (Path(job['output']) / 'DONE').write_text('{}')
            jobs = completed_jobs(output, reference)
            self.assertEqual([Path(job['output']).name for job in jobs], ['0', '1'])

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
        contrasts = list(ablation_comparisons(jobs, Path(args.output).resolve()))
        self.assertEqual(len(contrasts), 8)
        for _, entries, arms, varying in contrasts:
            self.assertEqual(len(entries), 6)
            self.assertEqual({entry['arm'] for entry in entries}, set(arms))
            paired_jobs = [next(job for job in jobs if Path(job['output']).name ==
                               Path(entry['metrics']).parent.parent.name) for entry in entries]
            settings = [{k: v for k, v in job.items() if k not in ('seed', 'output', *varying)}
                        for job in paired_jobs]
            self.assertTrue(all(setting == settings[0] for setting in settings))
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

    def test_ablation_manifests_preserve_seeds_and_reject_changed_settings(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            command = ['--recipe', 'trace', 'set', '--data', 'data', '--base', 'base', '--model', 'Qwen',
                       '--output', directory, '--seeds']
            main(command + ['0'])
            main(command + ['0', '1', '2'])
            path = Path(directory) / 'comparison_Qwen_set_mass.json'
            original = path.read_bytes()
            entries = json.loads(original)
            self.assertEqual(len(entries), 6)
            self.assertEqual({entry['arm'] for entry in entries}, {'single_norm', 'set_mass'})
            main(command + ['0'])
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(SystemExit):
                main(command + ['3', '--learning-rate', '0.002'])
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
