import unittest

from prepare_experiment import EXPORT_TREE_SHA, LEGACY, verify_export


class PreparationTests(unittest.TestCase):
    def test_frozen_atomic_export_payload_matches_confirmation_hash(self):
        export = LEGACY / "atomic_export/frozen_atomic_0p6b"
        self.assertEqual(verify_export(export, EXPORT_TREE_SHA), EXPORT_TREE_SHA)


if __name__ == "__main__":
    unittest.main()
