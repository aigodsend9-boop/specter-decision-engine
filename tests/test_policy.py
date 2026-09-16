"""Testes das políticas de consumo das decisões."""

from __future__ import annotations

import unittest

from specter_decision import (
    ConfidenceGate,
    Question,
    Route,
    SpecterDecisionEngine,
    SystemOneCalibration,
    SystemOneLocalBackend,
    composite_score,
    is_ok,
    rank_then_choose,
    require,
)


def _decision(identifier: str, confidence: float | None, *, status: str = "ok", **extra):
    payload = {
        "id": identifier,
        "type": extra.get("type", "choice"),
        "status": status,
        "value": extra.get("value", "a"),
        "probabilities": extra.get("probabilities", [0.9, 0.1]),
        "confidence": confidence,
        "legend": extra.get("legend", ["a", "b"]),
        "error": None if status == "ok" else "BACKEND_ERROR",
    }
    return payload


class GateTests(unittest.TestCase):
    def test_routes_by_threshold(self):
        gate = ConfidenceGate(act=0.9, verify=0.6)
        self.assertEqual(gate.route(_decision("x", 0.95)), Route.ACT)
        self.assertEqual(gate.route(_decision("x", 0.7)), Route.VERIFY)
        self.assertEqual(gate.route(_decision("x", 0.2)), Route.HUMAN)

    def test_failed_decision_is_unavailable(self):
        gate = ConfidenceGate()
        self.assertEqual(gate.route(_decision("x", None, status="error")), Route.UNAVAILABLE)
        self.assertEqual(gate.route(_decision("x", None)), Route.HUMAN)

    def test_thresholds_are_validated(self):
        with self.assertRaises(ValueError):
            ConfidenceGate(act=0.5, verify=0.9)
        with self.assertRaises(ValueError):
            ConfidenceGate(act=1.5)

    def test_partition_preserves_order(self):
        gate = ConfidenceGate(act=0.9, verify=0.5)
        decisions = [
            _decision("a", 0.95),
            _decision("b", 0.6),
            _decision("c", 0.95),
            _decision("d", 0.1),
        ]
        buckets = gate.partition(decisions)
        self.assertEqual([item["id"] for item in buckets[Route.ACT]], ["a", "c"])
        self.assertEqual([item["id"] for item in buckets[Route.VERIFY]], ["b"])
        self.assertEqual([item["id"] for item in buckets[Route.HUMAN]], ["d"])

    def test_require_enforces_success_and_confidence(self):
        self.assertEqual(require(_decision("x", 0.8)), "a")
        with self.assertRaises(ValueError):
            require(_decision("x", 0.8, status="error"))
        with self.assertRaises(ValueError):
            require(_decision("x", 0.4), minimum=0.7)

    def test_is_ok_matches_status(self):
        self.assertTrue(is_ok(_decision("x", 0.5)))
        self.assertFalse(is_ok(_decision("x", None, status="error")))


class CompositeTests(unittest.TestCase):
    def test_weighted_average_of_scores(self):
        decisions = [
            _decision("risco", 0.8, type="score", value=2.0, legend=["a", "b", "c"],
                      probabilities=[0.0, 0.0, 1.0]),
            _decision("impacto", 0.6, type="score", value=0.0, legend=["a", "b", "c"],
                      probabilities=[1.0, 0.0, 0.0]),
        ]
        result = composite_score(decisions, {"risco": 1.0, "impacto": 1.0})
        self.assertAlmostEqual(result["value"], 0.5)
        self.assertAlmostEqual(result["confidence"], 0.6)
        self.assertAlmostEqual(result["coverage"], 1.0)

    def test_noul_counts_as_unit_interval(self):
        decisions = [_decision("urgente", 0.9, type="noul", value=0.75, legend=["false", "true"])]
        result = composite_score(decisions, {"urgente": 1.0})
        self.assertAlmostEqual(result["value"], 0.75)

    def test_missing_components_reduce_coverage(self):
        decisions = [
            _decision("a", 0.9, type="score", value=1.0, legend=["x", "y", "z"]),
            _decision("b", None, type="score", status="error", legend=["x", "y", "z"]),
        ]
        result = composite_score(decisions, {"a": 1.0, "b": 1.0})
        self.assertEqual(result["missing"], ["b"])
        self.assertAlmostEqual(result["coverage"], 0.5)

    def test_unknown_ids_are_ignored(self):
        decisions = [_decision("a", 0.9, type="score", value=1.0, legend=["x", "y", "z"])]
        result = composite_score(decisions, {"a": 1.0, "inexistente": 2.0})
        self.assertAlmostEqual(result["coverage"], 1 / 3)


class RankThenChooseTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.engine = SpecterDecisionEngine(
            SystemOneLocalBackend(),
            SystemOneCalibration(domain="high-cardinality"),
            domain="high-cardinality",
            concurrency=8,
            timeout=5.0,
        )

    async def test_handles_more_than_255_options(self):
        options = [f"opcao_{index}" for index in range(600)]
        options[123] = "reembolso da fatura cobrada em duplicidade"
        decision = await rank_then_choose(
            self.engine, {"ticket": "Quero reembolso da fatura"}, "Qual rótulo?", options
        )
        self.assertEqual(decision["type"], "choice")
        self.assertEqual(decision["id"], "final")
        self.assertLessEqual(len(decision["legend"]), 8)
        self.assertIn(decision["value"], options)

    async def test_small_universe_skips_first_stage(self):
        decision = await rank_then_choose(
            self.engine, {"ticket": "erro de integração"}, "Qual equipe?", ["billing", "technical"]
        )
        self.assertEqual(decision["legend"], ["billing", "technical"])

    async def test_duplicates_are_collapsed(self):
        decision = await rank_then_choose(
            self.engine, {"t": "x"}, "Qual?", ["a", "a", "b", "b"]
        )
        self.assertEqual(decision["legend"], ["a", "b"])

    async def test_requires_two_options(self):
        with self.assertRaisesRegex(ValueError, "INVALID_INPUT"):
            await rank_then_choose(self.engine, {"t": "x"}, "Qual?", ["unica"])

    async def test_criteria_are_forwarded(self):
        decision = await rank_then_choose(
            self.engine,
            {"ticket": "quero cancelar minha assinatura e pedir reembolso"},
            "Qual equipe?",
            ["alpha", "beta"],
            criteria={"alpha": "faturas, reembolso, cobrança", "beta": "erros de integração"},
        )
        self.assertEqual(decision["value"], "alpha")


if __name__ == "__main__":
    unittest.main()
