"""Testes da maquinaria de calibração.

Os dados são sintéticos e gerados com semente fixa: servem para verificar
propriedades do ajuste (monotonicidade, redução de NLL, vínculo do
artefato), nunca para afirmar desempenho de modelo.
"""

from __future__ import annotations

import math
import random
import unittest

from specter_decision import Question
from specter_decision.calibration import (
    CalibrationArtifact,
    IsotonicCalibrator,
    SimpleCalibration,
    TemperatureScaler,
    accuracy_event,
    brier_score,
    bootstrap_interval,
    confidence_statistic,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_table,
)
from specter_decision.numerics import softmax


def _overconfident_dataset(size: int = 400, sharpness: float = 3.0):
    """Gera (pergunta, logits, verdade) com logits exagerados de propósito."""
    generator = random.Random(7)
    question = Question("route", "choice", "Qual equipe?", ("a", "b", "c"))
    records = []
    for _ in range(size):
        truth = generator.randrange(3)
        base = [generator.gauss(0.0, 0.6) for _ in range(3)]
        base[truth] += 1.1  # sinal real, modesto
        records.append((question, [value * sharpness for value in base], truth))
    return records


class TemperatureTests(unittest.TestCase):
    def test_fit_reduces_nll_on_overconfident_logits(self):
        records = _overconfident_dataset()
        scaler = TemperatureScaler().fit(
            [row for _, row, _ in records], [truth for _, _, truth in records]
        )
        self.assertGreater(scaler.temperature, 1.0)
        self.assertLess(scaler.nll_after, scaler.nll_before)

    def test_fit_keeps_temperature_when_already_calibrated(self):
        records = _overconfident_dataset(sharpness=1.0)
        logits = [row for _, row, _ in records]
        truths = [truth for _, _, truth in records]
        scaler = TemperatureScaler().fit(logits, truths)
        self.assertLessEqual(scaler.nll_after, scaler.nll_before + 1e-9)

    def test_fit_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            TemperatureScaler().fit([], [])
        with self.assertRaises(ValueError):
            TemperatureScaler().fit([[0.0, 1.0]], [0, 1])


class IsotonicTests(unittest.TestCase):
    def test_output_is_monotone(self):
        generator = random.Random(3)
        scores = [generator.random() for _ in range(300)]
        outcomes = [1 if generator.random() < score else 0 for score in scores]
        mapping = IsotonicCalibrator().fit(scores, outcomes)
        predictions = [mapping.predict(value / 50) for value in range(51)]
        self.assertEqual(predictions, sorted(predictions))

    def test_recovers_identity_on_well_calibrated_data(self):
        generator = random.Random(11)
        scores, outcomes = [], []
        for _ in range(4000):
            score = generator.random()
            scores.append(score)
            outcomes.append(1 if generator.random() < score else 0)
        mapping = IsotonicCalibrator().fit(scores, outcomes)
        for probe in (0.2, 0.5, 0.8):
            self.assertAlmostEqual(mapping.predict(probe), probe, delta=0.12)

    def test_serialization_roundtrip(self):
        mapping = IsotonicCalibrator().fit([0.1, 0.4, 0.9], [0, 1, 1])
        clone = IsotonicCalibrator.from_dict(mapping.to_dict())
        self.assertEqual(clone.knots_x, mapping.knots_x)
        self.assertEqual(clone.predict(0.5), mapping.predict(0.5))

    def test_empty_fit_raises(self):
        with self.assertRaises(ValueError):
            IsotonicCalibrator().fit([], [])


class MetricsTests(unittest.TestCase):
    def test_nll_and_brier_reward_correct_confidence(self):
        good = [[0.9, 0.1], [0.05, 0.95]]
        bad = [[0.1, 0.9], [0.95, 0.05]]
        truths = [0, 1]
        self.assertLess(negative_log_likelihood(good, truths), negative_log_likelihood(bad, truths))
        self.assertLess(brier_score(good, truths), brier_score(bad, truths))

    def test_ece_is_zero_for_perfect_calibration(self):
        confidences = [0.5] * 100
        outcomes = [1] * 50 + [0] * 50
        self.assertAlmostEqual(expected_calibration_error(confidences, outcomes), 0.0, places=9)

    def test_ece_detects_overconfidence(self):
        confidences = [0.99] * 100
        outcomes = [1] * 60 + [0] * 40
        self.assertAlmostEqual(expected_calibration_error(confidences, outcomes), 0.39, places=2)

    def test_reliability_table_covers_every_sample(self):
        confidences = [i / 100 for i in range(100)]
        outcomes = [i % 2 for i in range(100)]
        table = reliability_table(confidences, outcomes, bins=10)
        self.assertEqual(sum(row["n"] for row in table), 100)
        self.assertTrue(all(0.0 <= row["accuracy"] <= 1.0 for row in table))

    def test_bootstrap_is_reproducible(self):
        sample = [1.0] * 70 + [0.0] * 30
        first = bootstrap_interval(sample)
        second = bootstrap_interval(sample)
        self.assertEqual(first, second)
        self.assertLess(first[0], 0.7)
        self.assertGreater(first[1], 0.7)


class ConfidenceEventTests(unittest.TestCase):
    def test_choice_statistic_is_top_mass(self):
        question = Question("q", "choice", "?", ("a", "b", "c"))
        self.assertAlmostEqual(confidence_statistic(question, [0.7, 0.2, 0.1], "a"), 0.7)

    def test_score_statistic_uses_tolerance_window(self):
        question = Question("q", "score", "?", ("0", "1", "2"), tolerance=0.5)
        probabilities = [0.1, 0.8, 0.1]
        value = 1.0
        self.assertAlmostEqual(confidence_statistic(question, probabilities, value), 0.8)
        wide = Question("q", "score", "?", ("0", "1", "2"), tolerance=1.0)
        self.assertAlmostEqual(confidence_statistic(wide, probabilities, value), 1.0)

    def test_accuracy_event_matches_definition(self):
        choice = Question("q", "choice", "?", ("a", "b"))
        self.assertEqual(accuracy_event(choice, [0.2, 0.8], 1), 1)
        self.assertEqual(accuracy_event(choice, [0.8, 0.2], 1), 0)
        score = Question("q", "score", "?", ("0", "1", "2"), tolerance=0.5)
        self.assertEqual(accuracy_event(score, [0.0, 1.0, 0.0], 1), 1)
        self.assertEqual(accuracy_event(score, [1.0, 0.0, 0.0], 2), 0)


class ArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = _overconfident_dataset()
        self.artifact = CalibrationArtifact.fit(
            self.records, domain="synthetic", model_id="fixture", notes="teste"
        )

    def test_fit_produces_metrics_and_dataset_hash(self):
        metrics = self.artifact.metrics
        self.assertGreater(metrics["n_train"], metrics["n_test"])
        self.assertTrue(math.isfinite(metrics["ece"]))
        self.assertTrue(math.isfinite(metrics["brier"]))
        self.assertEqual(len(self.artifact.dataset_hash), 32)

    def test_calibration_improves_or_keeps_nll(self):
        self.assertLessEqual(
            self.artifact.metrics["nll_after"], self.artifact.metrics["nll_before"] + 1e-9
        )

    def test_confidence_is_bounded_and_uses_mapping(self):
        question = self.records[0][0]
        probabilities = softmax(self.records[0][1], self.artifact.temperature_value)
        confidence = self.artifact.confidence(question, probabilities, "a")
        self.assertTrue(0.0 <= confidence <= 1.0)

    def test_supports_refuses_other_domain_model_or_question(self):
        question = self.records[0][0]
        self.assertTrue(self.artifact.supports("synthetic", "fixture", question))
        self.assertFalse(self.artifact.supports("outro", "fixture", question))
        self.assertFalse(self.artifact.supports("synthetic", "outro-modelo", question))
        bound = CalibrationArtifact.fit(
            self.records, domain="synthetic", model_id="fixture", bind_signatures=True
        )
        other = Question("q", "choice", "Outra pergunta?", ("a", "b", "c"))
        self.assertFalse(bound.supports("synthetic", "fixture", other))

    def test_json_roundtrip_preserves_behaviour(self):
        clone = CalibrationArtifact.from_json(self.artifact.to_json())
        question = self.records[0][0]
        probabilities = softmax(self.records[0][1], clone.temperature_value)
        self.assertAlmostEqual(clone.temperature_value, self.artifact.temperature_value)
        self.assertAlmostEqual(
            clone.confidence(question, probabilities, "a"),
            self.artifact.confidence(question, probabilities, "a"),
        )

    def test_version_mismatch_is_rejected(self):
        payload = self.artifact.to_json().replace("specter-calibration/1", "outra/9")
        with self.assertRaises(ValueError):
            CalibrationArtifact.from_json(payload)

    def test_fit_requires_records(self):
        with self.assertRaises(ValueError):
            CalibrationArtifact.fit([], domain="d", model_id="m")


class SimpleCalibrationTests(unittest.TestCase):
    def test_ceiling_caps_confidence(self):
        calibration = SimpleCalibration(ceiling=0.5)
        question = Question("q", "choice", "?", ("a", "b"))
        self.assertAlmostEqual(calibration.confidence(question, [0.9, 0.1], "a"), 0.45)

    def test_domain_and_model_binding(self):
        calibration = SimpleCalibration(domain="d", model_id="m")
        self.assertTrue(calibration.supports("d", "m", None))
        self.assertFalse(calibration.supports("outro", "m", None))
        self.assertFalse(calibration.supports("d", "outro", None))

    def test_invalid_configuration_raises(self):
        with self.assertRaises(ValueError):
            SimpleCalibration(0.0)
        with self.assertRaises(ValueError):
            SimpleCalibration(1.0, ceiling=1.5)


if __name__ == "__main__":
    unittest.main()
