"""Testes do serviço, do hook HTTP e do cliente remoto.

Nenhum teste abre porta ou faz rede: o hook é chamado diretamente, como o
Core faria. A validação do cliente é exercida sobre envelopes sintéticos.
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
from specter_decision.http import DecisionClient, configure, dispatch_decision, openapi_document
from specter_decision.service import DecisionService, ServiceError, parse_questions

TOKEN = "segredo-do-operador"


def _factory(domain: str) -> SpecterDecisionEngine:
    return SpecterDecisionEngine(
        SystemOneLocalBackend(),
        SystemOneCalibration(domain=domain),
        domain=domain,
        concurrency=8,
        timeout=2.0,
    )


def _document() -> dict:
    return {
        "domain": "suporte",
        "state": {"ticket": "A integração de pagamentos falhou e estou perdendo vendas."},
        "questions": [
            {
                "id": "equipe",
                "type": "choice",
                "instructions": "Qual equipe deve atender?",
                "criteria": {"billing": "Faturas", "technical": "Erros e integrações"},
            },
            {"id": "urgente", "type": "noul", "instructions": "O relato indica urgência?"},
        ],
    }


class ParseTests(unittest.TestCase):
    def test_accepts_list_and_map_forms(self):
        as_list = parse_questions(_document()["questions"])
        as_map = parse_questions(
            {
                "equipe": {"type": "choice", "instructions": "Qual equipe?", "criteria": {"a": None, "b": None}},
                "urgente": {"type": "noul", "instructions": "urgente?"},
            }
        )
        self.assertEqual([q.id for q in as_list], ["equipe", "urgente"])
        self.assertEqual([q.type for q in as_map], ["choice", "noul"])

    def test_rejects_unknown_type_and_missing_criteria(self):
        with self.assertRaises(ServiceError):
            parse_questions([{"id": "x", "type": "extract", "instructions": "?"}])
        with self.assertRaises(ServiceError):
            parse_questions([{"id": "x", "type": "choice", "instructions": "?"}])

    def test_rejects_empty_and_oversized(self):
        with self.assertRaises(ServiceError):
            parse_questions([])
        big = [{"id": f"q{i}", "type": "noul", "instructions": "?"} for i in range(129)]
        with self.assertRaises(ServiceError):
            parse_questions(big)

    def test_score_requires_level_list(self):
        with self.assertRaises(ServiceError):
            parse_questions([{"id": "s", "type": "score", "instructions": "?", "criteria": "abc"}])


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.service = DecisionService(token=TOKEN, engine_factory=_factory, max_requests=2)

    def test_authorization_is_constant_time_and_strict(self):
        self.service.authorize(f"Bearer {TOKEN}")
        for header in (None, "", "Token abc", "Bearer errado"):
            with self.assertRaises(ServiceError) as context:
                self.service.authorize(header)
            self.assertEqual(context.exception.code, "UNAUTHORIZED")

    async def test_evaluate_returns_typed_envelope(self):
        envelope = await self.service.evaluate(_document())
        self.assertEqual(envelope["status"], "ok")
        self.assertEqual(envelope["contract"], "specter-decision/0.3")
        self.assertEqual([item["id"] for item in envelope["answers"]], ["equipe", "urgente"])
        self.assertEqual(envelope["answers"][0]["value"], "technical")
        json.dumps(envelope, allow_nan=False)

    async def test_missing_factory_reports_not_ready(self):
        service = DecisionService(token=TOKEN)
        with self.assertRaises(ServiceError) as context:
            await service.evaluate(_document())
        self.assertEqual(context.exception.code, "NOT_READY")

    async def test_invalid_documents_are_rejected(self):
        for document in ({"questions": []}, {"state": 1, "questions": "x"}, "não é objeto"):
            with self.assertRaises(ServiceError):
                await self.service.evaluate(document)

    async def test_capabilities_do_not_promise_production(self):
        capabilities = self.service.capabilities()
        self.assertFalse(capabilities["production_ready"])
        self.assertTrue(capabilities["configured"])
        self.assertEqual(capabilities["max_requests"], 2)

    async def test_overload_is_typed_not_queued(self):
        service = DecisionService(token=TOKEN, engine_factory=_factory, max_requests=1)
        service._in_flight = 1
        with self.assertRaises(ServiceError) as context:
            await service.evaluate(_document())
        self.assertEqual(context.exception.code, "OVERLOADED")
        self.assertEqual(context.exception.status, 429)


class DispatchTests(unittest.TestCase):
    def tearDown(self) -> None:
        configure(None)

    def test_unknown_paths_are_not_ours(self):
        configure(DecisionService(token=TOKEN, engine_factory=_factory))
        self.assertIsNone(dispatch_decision("GET", "/status", {}))
        self.assertIsNone(dispatch_decision("POST", "/v1/other", {}))

    def test_without_service_returns_not_configured(self):
        configure(None)
        status, _, body = dispatch_decision("GET", "/v1/decision/capabilities", {})
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(body)["error"], "NOT_CONFIGURED")

    def test_requires_authentication(self):
        configure(DecisionService(token=TOKEN, engine_factory=_factory))
        status, _, body = dispatch_decision("GET", "/v1/decision/capabilities", {})
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"], "UNAUTHORIZED")

    def test_evaluate_roundtrip(self):
        configure(DecisionService(token=TOKEN, engine_factory=_factory))
        payload = json.dumps(_document()).encode("utf-8")
        status, headers, body = dispatch_decision(
            "POST",
            "/v1/decision/evaluate?trace=1",
            {"Authorization": f"Bearer {TOKEN}"},
            payload,
        )
        envelope = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(len(envelope["answers"]), 2)

    def test_bad_json_and_oversized_body(self):
        configure(DecisionService(token=TOKEN, engine_factory=_factory))
        headers = {"Authorization": f"Bearer {TOKEN}"}
        status, _, body = dispatch_decision(
            "POST", "/v1/decision/evaluate", headers, b"{nao-e-json"
        )
        self.assertEqual(status, 422)
        status, _, _ = dispatch_decision(
            "POST", "/v1/decision/evaluate", headers, b"x" * (300 * 1024)
        )
        self.assertEqual(status, 413)

    def test_method_not_allowed_and_options(self):
        configure(DecisionService(token=TOKEN, engine_factory=_factory))
        headers = {"Authorization": f"Bearer {TOKEN}"}
        status, _, _ = dispatch_decision("DELETE", "/v1/decision/evaluate", headers, b"{}")
        self.assertEqual(status, 405)
        status, _, _ = dispatch_decision("OPTIONS", "/v1/decision/evaluate", {})
        self.assertEqual(status, 204)

    def test_openapi_is_served_and_valid(self):
        configure(DecisionService(token=TOKEN, engine_factory=_factory))
        status, _, body = dispatch_decision(
            "GET", "/v1/decision/openapi.json", {"Authorization": f"Bearer {TOKEN}"}
        )
        document = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(document["openapi"], "3.1.1")
        self.assertIn("/v1/decision/evaluate", document["paths"])
        self.assertEqual(document, openapi_document())


class ClientTests(unittest.TestCase):
    def test_rejects_plain_http_outside_loopback(self):
        with self.assertRaises(ValueError):
            DecisionClient("http://exemplo.com", TOKEN)
        DecisionClient("http://127.0.0.1:8888", TOKEN)
        DecisionClient("https://exemplo.com", TOKEN)

    def test_requires_token(self):
        with self.assertRaises(ValueError):
            DecisionClient("https://exemplo.com", "")

    def test_envelope_validation_catches_drift(self):
        questions = [Question("a", "choice", "?", ("x", "y"))]
        good = {
            "contract": "specter-decision/0.3",
            "answers": [
                {
                    "id": "a",
                    "type": "choice",
                    "status": "ok",
                    "value": "x",
                    "probabilities": [0.6, 0.4],
                    "confidence": 0.6,
                    "legend": ["x", "y"],
                    "error": None,
                }
            ],
        }
        DecisionClient._validate(good, questions)
        for mutation in (
            {"contract": "outro"},
            {"answers": []},
        ):
            broken = dict(good)
            broken.update(mutation)
            with self.assertRaises(ValueError):
                DecisionClient._validate(broken, questions)
        drifted = json.loads(json.dumps(good))
        drifted["answers"][0]["probabilities"] = [0.6, 0.6]
        with self.assertRaises(ValueError):
            DecisionClient._validate(drifted, questions)


if __name__ == "__main__":
    unittest.main()
