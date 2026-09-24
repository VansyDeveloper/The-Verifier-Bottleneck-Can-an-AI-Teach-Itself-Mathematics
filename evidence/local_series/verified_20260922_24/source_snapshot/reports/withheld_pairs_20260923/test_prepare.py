import tempfile
import unittest
from pathlib import Path

from prepare import reserved_registry, write_exclusive_or_verify


class ImmutableArtifactTests(unittest.TestCase):
    def test_prior_registry_includes_previous_holdout(self):
        import composition_core as core
        registry = reserved_registry(core)
        self.assertGreater(len(registry.tasks), 1000)
        self.assertGreater(len(registry.states), len(registry.tasks))

    def test_existing_different_artifact_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "manifest.json"
            write_exclusive_or_verify(target, b"first\n")
            write_exclusive_or_verify(target, b"first\n")
            with self.assertRaisesRegex(RuntimeError, "existing artifact differs"):
                write_exclusive_or_verify(target, b"second\n")
            self.assertEqual(target.read_bytes(), b"first\n")


if __name__ == "__main__":
    unittest.main()
