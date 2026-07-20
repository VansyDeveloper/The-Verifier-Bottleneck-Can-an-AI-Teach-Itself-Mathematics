import unittest

from modcomp.checker import noisy_verdict_from_event


class CheckerNoiseTest(unittest.TestCase):
    def test_event_noise_is_idempotent_and_answer_independent(self):
        first = noisy_verdict_from_event(True, "0|10|0|2|17|0", 0.7, 0.2)
        self.assertEqual(first, noisy_verdict_from_event(True, "0|10|0|2|17|0", 0.7, 0.2))
        second = noisy_verdict_from_event(True, "0|10|0|2|17|1", 0.7, 0.2)
        self.assertNotEqual(first[1], second[1])

    def test_event_noise_validates_parameters(self):
        with self.assertRaises(ValueError):
            noisy_verdict_from_event(True, "", 0.7, 0.2)
        with self.assertRaises(ValueError):
            noisy_verdict_from_event(True, "rollout", 1.1, 0.2)


if __name__ == "__main__":
    unittest.main()
