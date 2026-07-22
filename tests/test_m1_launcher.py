import os
import subprocess
import unittest


class M1LauncherTest(unittest.TestCase):
    def dry_run(self, session_tag, **overrides):
        env = {**os.environ, "DRY_RUN": "1", "SESSION_TAG": session_tag, **overrides}
        return subprocess.check_output(
            ["bash", "scripts/run_5h_m1.sh"], env=env, text=True
        )

    def test_checkpointed_runs_are_stable_across_sessions(self):
        first = self.dry_run("first")
        second = self.dry_run("second")
        self.assertEqual(first, second)
        self.assertIn("--checkpointing --save-steps 1", first)

    def test_generation_profile_is_configurable(self):
        output = self.dry_run(
            "profile",
            GENERATIONS="8",
            PER_DEVICE_BATCH="8",
            MAX_COMPLETION_LENGTH="128",
            PRIME_MAX="13",
            K_MIN="2",
            K_MAX="2",
            EVAL_STEPS="25",
            SAVE_FINAL="1",
        )
        self.assertIn("--num-generations 8", output)
        self.assertIn("--per-device-batch 8", output)
        self.assertIn("--max-completion-length 128", output)
        self.assertIn("--prime-max 13 --k-min 2 --k-max 2", output)
        self.assertIn("--eval-steps 25", output)
        self.assertNotIn("--no-save", output)

    def test_diagnostic_mode_has_only_three_declared_cells(self):
        output = self.dry_run("diagnostic", PILOT_MODE="three-cell-diagnostic")
        self.assertEqual(output.count("START h2_"), 3)
        for pair in ("a1.0_b0.0", "a0.6_b0.6", "a0.2_b0.8"):
            self.assertIn(pair, output)

    def test_multiple_seeds_expand_each_diagnostic_cell(self):
        output = self.dry_run(
            "seeds", PILOT_MODE="three-cell-diagnostic", SEEDS="0 1 2"
        )
        self.assertEqual(output.count("START h2_"), 9)
        for seed in ("s0", "s1", "s2"):
            self.assertIn(seed, output)

    def test_long_run_is_three_seed_and_keeps_final_adapters(self):
        env = {**os.environ, "DRY_RUN": "1", "SESSION_TAG": "long"}
        output = subprocess.check_output(
            ["bash", "scripts/run_long_m1.sh"], env=env, text=True
        )
        self.assertEqual(output.count("START h2_"), 9)
        self.assertIn("--max-steps 60 --eval-steps 15", output)
        self.assertIn("--checkpointing --save-steps 5", output)
        self.assertNotIn("--no-save", output)


if __name__ == "__main__":
    unittest.main()
