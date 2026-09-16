"""Camada de serviço: envelope tipado, autenticação e controle de admissão.

Separada do transporte de propósito. `DecisionService` não sabe o que é
HTTP: recebe um documento já decodificado e devolve um envelope. Isso
permite montá-la sobre o Core legado, sobre asyncio.Server, sobre um
worker de fila ou dentro de testes, sem duplicar regra de negócio.

Regras inegociáveis:
    * token comparado em tempo constante;
    * sem fila ilimitada — excedeu a concorrência, responde OVERLOADED;
    * `status=ok` no envelope significa requisição processada, jamais que
      todas as perguntas decidiram;
    * nenhum conteúdo de estado é registrado em log.
"""

from __future__ import annotations

import hmac
import time
import uuid
from typing import Any, Callable, Mapping, Sequence

from .engine import SpecterDecisionEngine
from .types import DEFAULT_LIMITS, Limits, Question

__all__ = ["DecisionService", "ServiceError", "parse_questions"]

CONTRACT = "specter-decision/0.3"


class ServiceError(Exception):
    """Erro de aplicação com código estável e status sugerido."""

    def __init__(self, code: str, detail: str = "", status: int = 400) -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail
        self.status = status


def parse_questions(payload: Any, limits: Limits = DEFAULT_LIMITS) -> list[Question]:
    """Converte o documento de entrada em perguntas validadas.

    Aceita duas formas:
        * lista: [{"id": "...", "type": "...", "text": ..., "labels": [...]}]
        * mapa (estilo System One): {"id": {"type": ..., "instructions": ...,
          "criteria": ...}}

    Args:
        payload: valor do campo `questions`.
        limits: limites de validação.

    Returns:
        Lista de `Question` na ordem original.

    Raises:
        ServiceError: INVALID_INPUT em qualquer desvio do contrato.
    """
    items: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(payload, Mapping):
        items = [(str(key), value) for key, value in payload.items()]
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        for entry in payload:
            if not isinstance(entry, Mapping):
                raise ServiceError("INVALID_INPUT", "pergunta deve ser objeto", 422)
            items.append((str(entry.get("id", "")), entry))
    else:
        raise ServiceError("INVALID_INPUT", "questions deve ser lista ou mapa", 422)

    if not items:
        raise ServiceError("INVALID_INPUT", "nenhuma pergunta", 422)
    if len(items) > limits.max_questions:
        raise ServiceError("INVALID_INPUT", "perguntas acima do limite", 422)

    questions: list[Question] = []
    for identifier, entry in items:
        kind = entry.get("type")
        text = entry.get("instructions", entry.get("text", ""))
        criteria = entry.get("criteria", entry.get("labels"))
        try:
            if kind == "noul":
                mapping = criteria if isinstance(criteria, Mapping) else None
                questions.append(Question.noul(identifier, text, mapping))
            elif kind == "choice":
                if criteria is None:
                    raise ServiceError("INVALID_INPUT", "choice exige criteria", 422)
                questions.append(Question.choice(identifier, text, criteria))
            elif kind == "score":
                if not isinstance(criteria, Sequence) or isinstance(criteria, (str, bytes)):
                    raise ServiceError("INVALID_INPUT", "score exige lista de níveis", 422)
                questions.append(
                    Question.score(
                        identifier, text, list(criteria), float(entry.get("tolerance", 0.5))
                    )
                )
            else:
                raise ServiceError("INVALID_INPUT", f"tipo inválido: {kind!r}", 422)
        except ValueError as error:
            raise ServiceError("INVALID_INPUT", str(error), 422) from error
    return questions


class DecisionService:
    """Fachada autenticada sobre uma fábrica de engines por domínio."""

    __slots__ = ("_token", "_factory", "_max_requests", "_in_flight", "_limits", "counters")

    def __init__(
        self,
        *,
        token: str,
        engine_factory: Callable[[str], SpecterDecisionEngine] | None = None,
        max_requests: int = 2,
        limits: Limits = DEFAULT_LIMITS,
    ) -> None:
        """Configura o serviço.

        Args:
            token: segredo do operador (Bearer).
            engine_factory: cria um engine por domínio. Sem fábrica, o
                serviço responde NOT_READY — instalado, porém inerte.
            max_requests: requisições de inferência simultâneas.
            limits: limites de validação.

        Raises:
            ValueError: token vazio ou max_requests inválido.
        """
        if not token:
            raise ValueError("token do operador é obrigatório")
        if max_requests < 1:
            raise ValueError("max_requests deve ser >= 1")
        self._token = token
        self._factory = engine_factory
        self._max_requests = max_requests
        self._in_flight = 0
        self._limits = limits
        self.counters = {"ok": 0, "rejected": 0, "overloaded": 0}

    def authorize(self, authorization: str | None) -> None:
        """Valida o cabeçalho Bearer em tempo constante.

        Raises:
            ServiceError: UNAUTHORIZED (401) quando ausente ou inválido.
        """
        prefix = "Bearer "
        if not authorization or not authorization.startswith(prefix):
            raise ServiceError("UNAUTHORIZED", "token ausente", 401)
        if not hmac.compare_digest(authorization[len(prefix) :].strip(), self._token):
            raise ServiceError("UNAUTHORIZED", "token inválido", 401)

    def capabilities(self) -> dict[str, Any]:
        """Descreve o serviço sem alegar prontidão produtiva."""
        return {
            "contract": CONTRACT,
            "configured": self._factory is not None,
            "max_requests": self._max_requests,
            "in_flight": self._in_flight,
            "limits": {
                "max_questions": self._limits.max_questions,
                "max_state_bytes": self._limits.max_state_bytes,
                "max_choice_options": self._limits.max_choice_options,
                "max_score_levels": self._limits.max_score_levels,
            },
            "question_types": ["noul", "choice", "score"],
            "production_ready": False,
            "counters": dict(self.counters),
        }

    async def evaluate(self, document: Mapping[str, Any]) -> dict[str, Any]:
        """Avalia um documento já decodificado.

        Args:
            document: {"state": ..., "questions": ..., "domain": "..."}.

        Returns:
            Envelope com `answers` na ordem das perguntas.

        Raises:
            ServiceError: para qualquer condição tipada de aplicação.
        """
        if self._factory is None:
            raise ServiceError("NOT_READY", "nenhuma fábrica de engine registrada", 503)
        if not isinstance(document, Mapping):
            raise ServiceError("INVALID_INPUT", "corpo deve ser objeto JSON", 422)
        if "state" not in document:
            raise ServiceError("INVALID_INPUT", "campo state ausente", 422)
        domain = str(document.get("domain", "default"))
        questions = parse_questions(document.get("questions"), self._limits)

        if self._in_flight >= self._max_requests:
            self.counters["overloaded"] += 1
            raise ServiceError("OVERLOADED", "limite de concorrência atingido", 429)

        self._in_flight += 1
        started = time.perf_counter()
        try:
            engine = self._factory(domain)
            answers = await engine.decide(document["state"], questions)
        except ValueError as error:
            self.counters["rejected"] += 1
            raise ServiceError("INVALID_INPUT", str(error), 422) from error
        finally:
            self._in_flight -= 1

        self.counters["ok"] += 1
        return {
            "status": "ok",
            "contract": CONTRACT,
            "domain": domain,
            "model": getattr(engine.backend, "model_id", "unknown"),
            "request_id": uuid.uuid4().hex,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "answers": answers,
        }
