"""Testes de contrato do kernel.

Superconjunto da suíte 0.2: todo teste original está aqui (adaptado apenas
onde a validação passou a ser antecipada para a construção da pergunta) e
os novos cobrem lote, deduplicação, admissão por deadline, abstenção,
recibos e propagação de cancelamento.

As fixtures são sintéticas. Elas não são modelo nem calibração e suas
probabilidades não significam nada fora destes testes.
"""

from __future__ import annotations

import asyncio
import json
import math
import unittest

from specter_decision import ErrorCode, Question, SpecterDecisionEngine


class FixtureModel:
    """Backend sintético: logits = (0, 1, 2, ...) com concorrência observável."""

    model_id = "fixture-not-a-model"

    def __init__(self, delay: float = 0.005) -> None:
        self.active = self.peak = 0
        self.seen: list[tuple[str, Question]] = []
        self.delay = delay

    async def logits(self, state_json: str, question: Question):
        self.seen.append((state_json, question))
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delay)
            return tuple(float(index) for index in range(question.cardinality))
        finally:
            self.active -= 1


class FixtureBatchModel(FixtureModel):
    """Backend sintético que implementa o protocolo de lote."""

    model_id = "fixture-batch-not-a-model"

    def __init__(self, delay: float = 0.005) -> None:
        super().__init__(delay)
        self.batches = 0
        self.batch_sizes: list[int] = []

    async def logits_batch(self, state_json: str, questions):
        self.batches += 1
        self.batch_sizes.append(len(questions))
        self.seen.extend((state_json, question) for question in questions)
        await asyncio.sleep(self.delay)
        return [
            tuple(float(index) for index in range(question.cardinality))
            for question in questions
        ]


class FixtureCalibration:
    """Calibração sintética. NÃO é calibração empírica."""

    def supports(self, *args) -> bool:
        return True

    def temperature(self, signature) -> float:
        return 1.0

    def confidence(self, question, probabilities, value) -> float:
        return 0.75


def _questions() -> list[Question]:
    return [
        Question("a", "noul", "Is X true?"),
        Question("b", "choice", "Route?", ("billing", "support")),
        Question("c", "score", "Severity?", ("none", "partial", "blocked")),
    ]


class EngineContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.model = FixtureModel()
        self.engine = SpecterDecisionEngine(
            self.model, FixtureCalibration(), domain="test", concurrency=2
        )
        self.qs = _questions()

    # --- suíte 0.2 ------------------------------------------------------
    async def test_values_and_serialization(self):
        result = await self.engine.decide({"ticket": "example"}, self.qs)
        self.assertAlmostEqual(result[0]["value"], 1 / (1 + math.exp(-1)))
        self.assertEqual(result[1]["value"], "support")
        self.assertTrue(0 < result[2]["value"] < 2)
        self.assertEqual(result[2]["legend"], ["none", "partial", "blocked"])
        for item in result:
            self.assertEqual(item["status"], "ok")
            self.assertAlmostEqual(sum(item["probabilities"]), 1)
        json.dumps(result, allow_nan=False)

    async def test_parallel_isolated_order(self):
        result = await self.engine.decide({"nested": [1]}, self.qs)
        self.assertEqual(self.model.peak, 2)
        self.assertEqual([item["id"] for item in result], ["a", "b", "c"])
        self.assertEqual(len({state for state, _ in self.model.seen}), 1)
        self.assertTrue(all(question.id == "_" for _, question in self.model.seen))

    async def test_missing_calibration(self):
        engine = SpecterDecisionEngine(self.model, domain="test")
        result = await engine.decide("x", self.qs)
        self.assertTrue(all(item["error"] == ErrorCode.UNCALIBRATED for item in result))
        self.assertEqual(self.model.seen, [])

    async def test_calibration_mismatch(self):
        self.engine.calibration.supports = lambda *args: False
        result = await self.engine.decide("x", self.qs)
        self.assertTrue(all(item["error"] == ErrorCode.UNCALIBRATED for item in result))
        self.assertEqual(self.model.seen, [])

    async def test_timeout_cancels(self):
        self.engine.timeout = 0.0001
        result = await self.engine.decide("x", self.qs)
        self.assertTrue(all(item["error"] == ErrorCode.TIMEOUT for item in result))
        self.assertEqual(self.model.active, 0)

    async def test_bad_logits(self):
        async def invalid(*args):
            return (float("nan"), 1)

        self.model.logits = invalid
        result = await self.engine.decide("x", self.qs)
        self.assertTrue(all(item["error"] == ErrorCode.INVALID_OUTPUT for item in result))

    async def test_bad_confidence(self):
        self.engine.calibration.confidence = lambda *args: float("inf")
        result = await self.engine.decide("x", self.qs)
        self.assertTrue(all(item["error"] == ErrorCode.INVALID_OUTPUT for item in result))

    async def test_invalid_requests(self):
        cases = [
            (float("nan"), self.qs),
            ({"x": float("nan")}, self.qs),
            ("x", [self.qs[0], self.qs[0]]),
            ("x", []),
            ("x", [Question("d", "noul", "ok"), object()]),
        ]
        for state, questions in cases:
            with self.assertRaisesRegex(ValueError, "INVALID_INPUT"):
                await self.engine.decide(state, questions)

    async def test_invalid_questions_fail_on_construction(self):
        # Mudança deliberada em 0.3: a pergunta inválida é rejeitada na
        # construção, não na avaliação. O tipo de exceção e o prefixo da
        # mensagem continuam iguais.
        with self.assertRaisesRegex(ValueError, "INVALID_INPUT"):
            Question("x", "choice", "?", ("a", "a"))
        with self.assertRaisesRegex(ValueError, "INVALID_INPUT"):
            Question("x", "score", "?", ("only",))

    async def test_single_matches_batch(self):
        batch = await self.engine.decide("x", self.qs)
        for question, expected in zip(self.qs, batch):
            single = await self.engine.decide("x", [question])
            self.assertEqual(single[0], expected)

    async def test_backend_failure_isolated(self):
        original = self.model.logits

        async def failing(state, question):
            if question.type == "choice":
                raise RuntimeError("never emit provider text")
            return await original(state, question)

        self.model.logits = failing
        result = await self.engine.decide("x", self.qs)
        self.assertEqual([item["status"] for item in result], ["ok", "error", "ok"])
        self.assertEqual(result[1]["error"], ErrorCode.BACKEND_ERROR)

    # --- novidades 0.3 --------------------------------------------------
    async def test_batch_backend_uses_single_call(self):
        model = FixtureBatchModel()
        engine = SpecterDecisionEngine(model, FixtureCalibration(), domain="test")
        result = await engine.decide("x", self.qs)
        self.assertEqual(model.batches, 1)
        self.assertEqual(model.batch_sizes, [3])
        self.assertTrue(all(item["status"] == "ok" for item in result))

    async def test_identical_questions_are_deduplicated(self):
        questions = [Question(f"q{i}", "noul", "Is this urgent?") for i in range(24)]
        result = await self.engine.decide("x", questions)
        self.assertEqual(len(result), 24)
        self.assertEqual(len(self.model.seen), 1)
        values = {item["value"] for item in result}
        self.assertEqual(len(values), 1)
        self.assertEqual([item["id"] for item in result], [f"q{i}" for i in range(24)])

    async def test_batch_failure_is_typed(self):
        model = FixtureBatchModel()

        async def broken(*args):
            raise RuntimeError("backend indisponível")

        model.logits_batch = broken
        engine = SpecterDecisionEngine(model, FixtureCalibration(), domain="test")
        result = await engine.decide("x", self.qs)
        self.assertTrue(all(item["error"] == ErrorCode.BACKEND_ERROR for item in result))

    async def test_batch_wrong_cardinality_is_invalid_output(self):
        model = FixtureBatchModel()

        async def short(state_json, questions):
            return [(0.0, 1.0)]

        model.logits_batch = short
        engine = SpecterDecisionEngine(model, FixtureCalibration(), domain="test")
        result = await engine.decide("x", self.qs)
        self.assertTrue(all(item["error"] == ErrorCode.INVALID_OUTPUT for item in result))

    async def test_abstention_policy(self):
        engine = SpecterDecisionEngine(
            self.model, FixtureCalibration(), domain="test", abstain_below=0.9
        )
        result = await engine.decide("x", self.qs)
        self.assertTrue(all(item["error"] == ErrorCode.ABSTAINED for item in result))
        self.assertTrue(all(item["value"] is None for item in result))

    async def test_receipts_are_stable_and_state_bound(self):
        engine = SpecterDecisionEngine(
            self.model, FixtureCalibration(), domain="test", receipts=True
        )
        first = await engine.decide({"a": 1}, self.qs)
        again = await engine.decide({"a": 1}, self.qs)
        other = await engine.decide({"a": 2}, self.qs)
        self.assertEqual([item["receipt"] for item in first], [item["receipt"] for item in again])
        self.assertNotEqual(first[0]["receipt"], other[0]["receipt"])

    async def test_cancellation_propagates_and_cleans_up(self):
        slow = FixtureModel(delay=5.0)
        engine = SpecterDecisionEngine(slow, FixtureCalibration(), domain="test", timeout=None)
        task = asyncio.create_task(engine.decide("x", self.qs))
        await asyncio.sleep(0.02)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        self.assertEqual(slow.active, 0)

    async def test_state_larger_than_limit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "INVALID_INPUT"):
            await self.engine.decide({"blob": "x" * 300_000}, self.qs)

    async def test_capabilities_do_not_claim_readiness(self):
        capabilities = self.engine.capabilities()
        self.assertFalse(capabilities["production_ready"])
        self.assertEqual(capabilities["contract"], "specter-decision/0.3")
        self.assertIn("TIMEOUT", capabilities["error_codes"])

    async def test_decide_sync_outside_loop(self):
        def run() -> list:
            engine = SpecterDecisionEngine(
                FixtureModel(), FixtureCalibration(), domain="test"
            )
            return engine.decide_sync("x", _questions())

        result = await asyncio.to_thread(run)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(item["status"] == "ok" for item in result))


if __name__ == "__main__":
    unittest.main()
