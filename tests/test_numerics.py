"""Testes do núcleo numérico: estabilidade, simplex e determinismo."""

from __future__ import annotations

import math
import unittest

from specter_decision.numerics import (
    argmax,
    entropy_normalized,
    expectation,
    margin,
    renormalize,
    softmax,
    validate_simplex,
)


class NumericsTests(unittest.TestCase):
    def test_softmax_matches_reference(self):
        probabilities = softmax([0.0, 1.0])
        self.assertAlmostEqual(probabilities[1], 1 / (1 + math.exp(-1)), places=12)

    def test_softmax_sums_to_one_exactly(self):
        for logits in ([0.0, 1.0, 2.0], [-5.0] * 7, list(range(10))):
            probabilities = softmax(logits)
            self.assertEqual(math.fsum(probabilities), 1.0)
            self.assertTrue(validate_simplex(probabilities, size=len(logits)))

    def test_softmax_is_stable_with_huge_logits(self):
        probabilities = softmax([1e308, 1e308 - 1.0, -1e308])
        self.assertTrue(all(math.isfinite(value) for value in probabilities))
        self.assertAlmostEqual(math.fsum(probabilities), 1.0, places=12)

    def test_softmax_is_shift_invariant(self):
        base = softmax([0.3, -1.2, 4.0])
        shifted = softmax([100.3, 98.8, 104.0])
        for left, right in zip(base, shifted):
            self.assertAlmostEqual(left, right, places=12)

    def test_temperature_controls_sharpness(self):
        cold = softmax([0.0, 2.0], 0.5)
        warm = softmax([0.0, 2.0], 4.0)
        self.assertGreater(cold[1], warm[1])
        self.assertGreater(entropy_normalized(warm), entropy_normalized(cold))

    def test_invalid_inputs_raise(self):
        for logits, temperature in (
            ([], 1.0),
            ([float("nan"), 0.0], 1.0),
            ([float("inf"), 0.0], 1.0),
            ([0.0, 1.0], 0.0),
            ([0.0, 1.0], -1.0),
            ([0.0, 1.0], float("nan")),
        ):
            with self.assertRaises(ValueError):
                softmax(logits, temperature)

    def test_argmax_breaks_ties_by_first_index(self):
        self.assertEqual(argmax([0.5, 0.5, 0.5]), 0)
        self.assertEqual(argmax([0.1, 0.9, 0.9]), 1)

    def test_expectation_and_margin(self):
        self.assertAlmostEqual(expectation([0.5, 0.0, 0.5]), 1.0)
        self.assertAlmostEqual(expectation([1.0, 0.0]), 0.0)
        self.assertAlmostEqual(margin([0.6, 0.3, 0.1]), 0.3)

    def test_entropy_bounds(self):
        self.assertAlmostEqual(entropy_normalized([0.25] * 4), 1.0, places=12)
        self.assertAlmostEqual(entropy_normalized([1.0, 0.0]), 0.0, places=12)

    def test_renormalize_fixes_residual(self):
        values = [0.3333333333, 0.3333333333, 0.3333333333]
        self.assertEqual(math.fsum(renormalize(values)), 1.0)

    def test_validate_simplex_rejects_bad_distributions(self):
        self.assertFalse(validate_simplex([0.5, 0.4]))
        self.assertFalse(validate_simplex([0.5, 0.6]))
        self.assertFalse(validate_simplex([1.2, -0.2]))
        self.assertFalse(validate_simplex([0.5, 0.5], size=3))
        self.assertFalse(validate_simplex([float("nan"), 1.0]))

    def test_underflow_degenerates_to_uniform(self):
        probabilities = softmax([-1e6, -1e6 - 1000.0])
        self.assertAlmostEqual(math.fsum(probabilities), 1.0, places=12)
        self.assertTrue(all(value >= 0.0 for value in probabilities))


if __name__ == "__main__":
    unittest.main()
