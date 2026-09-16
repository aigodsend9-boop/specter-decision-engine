"""Testes do backend local heurístico.

Superconjunto da suíte 0.2. Os testes semânticos verificam o comportamento
DA HEURÍSTICA declarada (expansão por glossário), não inteligência de
modelo: eles falham se a heurística regredir, e é para isso que existem.
"""

from __future__ import annotations

import asyncio
import json
import unittest

from specter_decision import (
    Question,
    SpecterDecisionEngine,
    SystemOneCalibration,
    SystemOneLocalBackend,
)
from specter_decision.backends.system_one import tokenize

TICKET = (
    "Hi, I've been trying to connect my Stripe account for 3 days "
    "and it keeps failing. I'm losing sales. Please help ASAP."
)


class SystemOneTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.backend = SystemOneLocalBackend()
        self.engine = SpecterDecisionEngine(
            self.backend,
            SystemOneCalibration(domain="system-one-local"),
            domain="system-one-local",
            concurrency=8,
            timeout=2.0,
        )
        self.state = {"ticket": TICKET}
        self.qs = [
            Question(
                "department", "choice", "Which team should handle this",
                ("billing", "technical", "sales"),
            ),
            Question(
                "frustration", "score", "How frustrated the customer appears",
                ("Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"),
            ),
            Question("is_urgent", "noul", "The message conveys urgency or time-sensitivity"),
        ]

    # --- suíte 0.2 ------------------------------------------------------
    async def test_all_ok_and_serializable(self):
        result = await self.engine.decide(self.state, self.qs)
        self.assertEqual(len(result), 3)
        for item in result:
            self.assertEqual(item["status"], "ok")
            self.assertIsNone(item["error"])
            self.assertTrue(0 <= item["confidence"] <= 1)
            self.assertAlmostEqual(sum(item["probabilities"]), 1.0, places=5)
        json.dumps(result, allow_nan=False)

    async def test_prefill_reused(self):
        await self.engine.decide(self.state, self.qs)
        first = self.backend._last_prefill.fingerprint
        await self.engine.decide(self.state, self.qs[:1])
        second = self.backend._last_prefill.fingerprint
        self.assertEqual(first, second)
        self.assertGreaterEqual(self.backend.stats()["prefill_cache"]["hits"], 1)

    async def test_technical_preferred_on_stripe_fail(self):
        result = await self.engine.decide(self.state, self.qs)
        department = result[0]
        self.assertEqual(department["value"], "technical")
        self.assertGreater(department["probabilities"][1], department["probabilities"][0])

    async def test_urgency_high(self):
        result = await self.engine.decide(self.state, self.qs)
        self.assertGreater(result[2]["value"], 0.6)

    async def test_parallel_28(self):
        questions = [Question(f"q{i}", "noul", "Is this urgent?") for i in range(28)]
        result = await self.engine.decide(self.state, questions)
        self.assertEqual(len(result), 28)
        self.assertTrue(all(item["status"] == "ok" for item in result))

    async def test_exports(self):
        import specter_decision as sd

        self.assertTrue(hasattr(sd, "SystemOneLocalBackend"))
        self.assertTrue(hasattr(sd, "SimpleCalibration"))
        self.assertEqual(sd.__version__, "0.3.0")

    # --- novidades 0.3 --------------------------------------------------
    async def test_billing_wins_on_invoice_language(self):
        state = {"ticket": "Fui cobrado duas vezes na fatura deste mês e quero reembolso."}
        result = await self.engine.decide(state, [self.qs[0]])
        self.assertEqual(result[0]["value"], "billing")

    async def test_sales_wins_on_pricing_language(self):
        state = {"ticket": "Gostaria de um orçamento e uma demo antes de fechar o contrato."}
        result = await self.engine.decide(state, [self.qs[0]])
        self.assertEqual(result[0]["value"], "sales")

    async def test_non_urgent_message_scores_low(self):
        state = {"ticket": "Hello, I was just wondering how the export feature works."}
        result = await self.engine.decide(state, [self.qs[2]])
        self.assertLess(result[0]["value"], 0.5)

    async def test_criteria_sharpen_the_decision(self):
        plain = Question("d", "choice", "Qual equipe?", ("alpha", "beta"))
        described = Question.choice(
            "d",
            "Qual equipe?",
            {"alpha": "Cobrança, faturas e reembolso", "beta": "Preço, upgrade e contrato"},
        )
        state = {"ticket": "Quero reembolso da fatura cobrada em duplicidade."}
        flat = (await self.engine.decide(state, [plain]))[0]
        rich = (await self.engine.decide(state, [described]))[0]
        self.assertAlmostEqual(flat["probabilities"][0], 0.5, places=6)
        self.assertEqual(rich["value"], "alpha")
        self.assertGreater(rich["confidence"], flat["confidence"])

    async def test_batch_path_is_used_and_matches_single(self):
        batch = await self.engine.decide(self.state, self.qs)
        self.assertEqual(self.backend.batch_calls, 1)
        for question, expected in zip(self.qs, batch):
            single = await self.engine.decide(self.state, [question])
            self.assertEqual(single[0]["value"], expected["value"])
            self.assertEqual(single[0]["probabilities"], expected["probabilities"])

    async def test_deterministic_across_instances(self):
        other = SpecterDecisionEngine(
            SystemOneLocalBackend(),
            SystemOneCalibration(domain="system-one-local"),
            domain="system-one-local",
        )
        left = await self.engine.decide(self.state, self.qs)
        right = await other.decide(self.state, self.qs)
        self.assertEqual(left, right)

    async def test_empty_state_gives_flat_distribution(self):
        result = await self.engine.decide({"ticket": ""}, [self.qs[0]])
        probabilities = result[0]["probabilities"]
        self.assertAlmostEqual(max(probabilities), 1 / 3, places=6)
        self.assertLess(result[0]["confidence"], 0.5)

    async def test_sketch_cache_avoids_recomputation(self):
        await self.engine.decide(self.state, self.qs)
        await self.engine.decide({"ticket": "outro texto totalmente diferente"}, self.qs)
        self.assertGreaterEqual(self.backend.stats()["sketch_cache"]["hits"], 3)

    async def test_calibration_refuses_foreign_model(self):
        engine = SpecterDecisionEngine(
            self.backend,
            SystemOneCalibration(domain="outro-dominio"),
            domain="system-one-local",
        )
        result = await engine.decide(self.state, self.qs)
        self.assertTrue(all(item["error"] == "UNCALIBRATED" for item in result))

    def test_tokenizer_normalizes_accents_and_plural(self):
        self.assertEqual(tokenize("Ações URGENTES!"), ["acao", "urgente"])
        self.assertEqual(tokenize("integrações"), ["integracao"])
        self.assertEqual(tokenize("the and of"), [])
        self.assertEqual(tokenize("API 500 erro"), ["api", "500", "erro"])


if __name__ == "__main__":
    unittest.main()
