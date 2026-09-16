"""Synthetic contract tests only. Fixtures are NOT a semantic model or calibration."""
import asyncio
import json
import math
import unittest
from engine import Question, SpecterDecisionEngine


class FixtureModel:
    model_id = 'fixture-not-a-model'
    def __init__(self):
        self.active = self.peak = 0
        self.seen = []
    async def logits(self, state_json, question):
        self.seen.append((state_json, question))
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(.005)
            return tuple(float(i) for i in range(len(question.labels)))
        finally:
            self.active -= 1


class FixtureCalibration:
    def supports(self, *args): return True
    def temperature(self, signature): return 1.0
    def confidence(self, q, ps, value): return .75  # TEST ONLY; not calibrated.


class EngineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model = FixtureModel()
        self.engine = SpecterDecisionEngine(self.model, FixtureCalibration(), domain='test', concurrency=2)
        self.qs = [Question('a', 'noul', 'Is X true?'),
                   Question('b', 'choice', 'Route?', ('billing', 'support')),
                   Question('c', 'score', 'Severity?', ('none', 'partial', 'blocked'))]

    async def test_values_and_serialization(self):
        result = await self.engine.decide({'ticket': 'example'}, self.qs)
        self.assertAlmostEqual(result[0]['value'], 1 / (1 + math.exp(-1)))
        self.assertEqual(result[1]['value'], 'support')
        self.assertTrue(0 < result[2]['value'] < 2)
        self.assertEqual(result[2]['legend'], ['none', 'partial', 'blocked'])
        for item in result:
            self.assertEqual(item['status'], 'ok')
            self.assertAlmostEqual(sum(item['probabilities']), 1)
        json.dumps(result, allow_nan=False)

    async def test_parallel_isolated_order(self):
        result = await self.engine.decide({'nested': [1]}, self.qs)
        self.assertEqual(self.model.peak, 2)
        self.assertEqual([x['id'] for x in result], ['a', 'b', 'c'])
        self.assertEqual(len({s for s, q in self.model.seen}), 1)
        self.assertTrue(all(q.id == '_' for s, q in self.model.seen))

    async def test_missing_calibration(self):
        engine = SpecterDecisionEngine(self.model, domain='test')
        result = await engine.decide('x', self.qs)
        self.assertTrue(all(x['error'] == 'UNCALIBRATED' for x in result))
        self.assertEqual(self.model.seen, [])

    async def test_calibration_mismatch(self):
        self.engine.calibration.supports = lambda *args: False
        result = await self.engine.decide('x', self.qs)
        self.assertTrue(all(x['error'] == 'UNCALIBRATED' for x in result))
        self.assertEqual(self.model.seen, [])

    async def test_timeout_cancels(self):
        self.engine.timeout = .0001
        result = await self.engine.decide('x', self.qs)
        self.assertTrue(all(x['error'] == 'TIMEOUT' for x in result))
        self.assertEqual(self.model.active, 0)

    async def test_bad_logits(self):
        async def invalid(*args): return (float('nan'), 1)
        self.model.logits = invalid
        result = await self.engine.decide('x', self.qs)
        self.assertTrue(all(x['error'] == 'INVALID_OUTPUT' for x in result))

    async def test_bad_confidence(self):
        self.engine.calibration.confidence = lambda *args: float('inf')
        result = await self.engine.decide('x', self.qs)
        self.assertTrue(all(x['error'] == 'INVALID_OUTPUT' for x in result))

    async def test_invalid_requests(self):
        cases = [(float('nan'), self.qs), ({'x': float('nan')}, self.qs),
                 ('x', [self.qs[0], self.qs[0]]), ('x', []),
                 ('x', [Question('x', 'choice', '?', ('a', 'a'))]),
                 ('x', [Question('x', 'score', '?', ('only',))])]
        for state, qs in cases:
            with self.assertRaisesRegex(ValueError, 'INVALID_INPUT'):
                await self.engine.decide(state, qs)

    async def test_single_matches_batch(self):
        batch = await self.engine.decide('x', self.qs)
        for q, expected in zip(self.qs, batch):
            self.assertEqual((await self.engine.decide('x', [q]))[0], expected)

    async def test_backend_failure_isolated(self):
        original = self.model.logits
        async def failing(state, q):
            if q.type == 'choice': raise RuntimeError('never emit provider text')
            return await original(state, q)
        self.model.logits = failing
        result = await self.engine.decide('x', self.qs)
        self.assertEqual([x['status'] for x in result], ['ok', 'error', 'ok'])
        self.assertEqual(result[1]['error'], 'BACKEND_ERROR')


if __name__ == '__main__':
    unittest.main()
