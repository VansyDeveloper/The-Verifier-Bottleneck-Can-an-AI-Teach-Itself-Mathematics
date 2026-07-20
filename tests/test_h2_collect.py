import unittest

from analysis.h2_checker_noise.collect_results import (
    first_hitting_step,
    parse_config_id,
    parse_run_name,
    summarize_cell,
)


class H2CollectorTest(unittest.TestCase):
    def test_hitting_time_is_per_curve(self):
        self.assertEqual(first_hitting_step([0, 25, 50], [0.4, 0.91, 0.8], 0.9), 25)
        self.assertIsNone(first_hitting_step([0, 25], [0.4, 0.89], 0.9))

    def test_parses_fingerprinted_run_name(self):
        name = "h2_cabc123_n600_a0.8_b0.2_t1.0_g16_s2"
        self.assertEqual(parse_config_id(name), "abc123")
        self.assertEqual(
            parse_run_name(name),
            {"n": 600.0, "a": 0.8, "b": 0.2, "t": 1.0, "s": 2.0},
        )

    def test_cell_requires_all_seeds_and_uses_common_steps(self):
        runs = [
            {
                "seed": seed,
                "eval_steps": [0, 25] + ([50] if seed == 0 else []),
                "eval_true_acc": [0.5, 0.5 + 0.1 * seed]
                + ([0.9] if seed == 0 else []),
                "base_solved_pass1_change": -0.1 * seed,
            }
            for seed in range(3)
        ]
        self.assertIsNone(summarize_cell(1.0, 0.0, runs[:2], 3))
        cell = summarize_cell(1.0, 0.0, runs, 3, "abc123")
        self.assertEqual(cell["eval_steps"], [0, 25])
        self.assertEqual(cell["seeds"], [0, 1, 2])
        self.assertAlmostEqual(cell["mean_eval_true_acc"][1], 0.6)
        self.assertAlmostEqual(cell["mean_base_solved_pass1_change"], -0.1)
        self.assertEqual(cell["config_id"], "abc123")

        no_step_zero = [{**run, "eval_steps": [25]} for run in runs]
        self.assertIsNone(summarize_cell(1.0, 0.0, no_step_zero, 3))


if __name__ == "__main__":
    unittest.main()
