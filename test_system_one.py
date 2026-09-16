"""Integration tests for SystemOneLocalBackend (contract, not semantic model claims)."""
import asyncio
import json
import math
import unittest

from specter_decision import (
    Question,
    SpecterDecisionEngine,
    SystemOneLocalBackend,
    SystemOneCalibration,
)


class SystemOneTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.backend = SystemOneLocalBackend()
        self.engine = SpecterDecisionEngine(
            self.backend,
            SystemOneCalibration(domain="system-one-local"),
            domain="system-one-local",
            concurrency=8,
            timeout=2.0,
        )
        self.state = {
            "ticket": (
                "Hi, I've been trying to connect my Stripe account for 3 days "
                "and it keeps failing. I'm losing sales. Please help ASAP."
            )
        }
        self.qs = [
            Question("department", "choice", "Which team should handle this",
                     ("billing", "technical", "sales")),
            Question("frustration", "score", "How frustrated the customer appears",
                     ("Calm, just stating facts", "Frustrated but civil", "Very angry, strong language")),
            Question("is_urgent", "noul", "The message conveys urgency or time-sensitivity"),
        ]

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
        fp1 = self.backend._last_prefill.fingerprint
        await self.engine.decide(self.state, self.qs[:1])
        fp2 = self.backend._last_prefill.fingerprint
        self.assertEqual(fp1, fp2)

    async def test_technical_preferred_on_stripe_fail(self):
        result = await self.engine.decide(self.state, self.qs)
        dept = result[0]
        self.assertEqual(dept["value"], "technical")
        self.assertGreater(dept["probabilities"][1], dept["probabilities"][0])

    async def test_urgency_high(self):
        result = await self.engine.decide(self.state, self.qs)
        self.assertGreater(result[2]["value"], 0.6)

    async def test_parallel_28(self):
        qs = [Question(f"q{i}", "noul", "Is this urgent?") for i in range(28)]
        result = await self.engine.decide(self.state, qs)
        self.assertEqual(len(result), 28)
        self.assertTrue(all(r["status"] == "ok" for r in result))

    async def test_exports(self):
        import specter_decision as sd
        self.assertTrue(hasattr(sd, "SystemOneLocalBackend"))
        self.assertTrue(hasattr(sd, "SimpleCalibration"))
        self.assertEqual(sd.__version__, "0.2.0rc3")


if __name__ == "__main__":
    unittest.main()
