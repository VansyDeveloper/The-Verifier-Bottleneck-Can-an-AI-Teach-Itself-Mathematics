import tempfile
import unittest
from pathlib import Path

from plot_analysis import plot_hitk


class PlotTests(unittest.TestCase):
    def test_hitk_plot_writes_nonempty_png(self):
        rows = [{"arm": arm, "k": k, "mean_hit_fraction": value}
                for arm, values in (("atomic_control", (0.2, 1.0)),
                                    ("composition_distill", (0.5, 1.0)))
                for k, value in enumerate(values, 1)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "hitk.png"
            plot_hitk(rows, output, title="Test")
            self.assertTrue(output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertGreater(output.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
