"""Adaptador para a API System One da TypeSafe (modelo Jev).

Por que existe: o kernel do Specter é bom no que é *engenharia* — tipos,
limites, calibração, deadline, auditoria. O que ele não tem é um modelo
semântico treinado. Este adaptador permite usar um backend System One
real por trás do mesmo contrato tipado, sem que a aplicação mude uma
linha: `SpecterDecisionEngine(JevRemoteBackend(...), artefato_calibrado)`.

Mapeamento (probabilidades -> logits):
    O kernel espera logits e aplica softmax com temperatura versionada.
    Como softmax(ln p) == p para T = 1, converter a distribuição devolvida
    pelo provedor em log-probabilidades preserva exatamente os números do
    provedor e ainda permite recalibrar a temperatura localmente.

Rede: `urllib.request` em thread separada (`asyncio.to_thread`), para não
bloquear o event loop. HTTPS obrigatório, sem redirects, com backoff
exponencial apenas em 429/529, como a documentação do provedor recomenda.

Nada aqui é ativado por padrão: sem chave de API o backend nem é
instanciável, e o kernel devolve UNCALIBRATED sem artefato compatível.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import urllib.error
import urllib.request
from typing import Any, Final, Sequence

from ..types import Question

__all__ = ["JevRemoteBackend"]

_DEFAULT_ENDPOINT: Final[str] = "https://api.typesafe.ai/v1/systemone"
_MIN_PROBABILITY: Final[float] = 1e-9
_RETRYABLE: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504, 529})


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Recusa redirects: credencial nunca segue para host não previsto."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        return None


class JevRemoteBackend:
    """Backend remoto compatível com `Backend` e `BatchBackend`.

    Attributes:
        model_id: identificador do modelo remoto (entra no vínculo do
            artefato de calibração).
    """

    __slots__ = ("model_id", "_endpoint", "_key", "_timeout", "_retries", "_opener", "calls")

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = "jev-latest",
        endpoint: str = _DEFAULT_ENDPOINT,
        timeout: float = 5.0,
        retries: int = 2,
    ) -> None:
        """Configura o cliente remoto.

        Args:
            api_key: credencial; se None, lê `TYPESAFE_API_KEY`.
            model: nome do modelo remoto.
            endpoint: URL absoluta HTTPS do endpoint de avaliação.
            timeout: deadline de rede por chamada, em segundos.
            retries: tentativas extras apenas para 429/5xx retryable.

        Raises:
            ValueError: sem credencial ou com endpoint não-HTTPS.
        """
        key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise ValueError("credencial ausente: defina TYPESAFE_API_KEY")
        if not endpoint.startswith("https://"):
            raise ValueError("endpoint deve ser HTTPS")
        self.model_id = f"typesafe:{model}"
        self._endpoint = endpoint
        self._key = key
        self._timeout = float(timeout)
        self._retries = max(0, int(retries))
        self._opener = urllib.request.build_opener(_NoRedirect)
        self.calls = 0

    # ------------------------------------------------------------- protocolo
    async def logits(self, state_json: str, question: Question) -> list[float]:
        """Avalia uma pergunta (uma chamada de rede)."""
        rows = await self.logits_batch(state_json, (question,))
        return rows[0]

    async def logits_batch(
        self, state_json: str, questions: Sequence[Question]
    ) -> list[list[float]]:
        """Avalia todas as perguntas em uma única chamada remota."""
        payload = self._build_payload(state_json, questions)
        body = await asyncio.to_thread(self._post, payload)
        answers = body.get("answers")
        if not isinstance(answers, dict):
            raise ValueError("resposta remota sem mapa de answers")
        return [
            _to_logits(question, answers.get(f"q{index}"))
            for index, question in enumerate(questions)
        ]

    # ------------------------------------------------------------- transporte
    def _build_payload(self, state_json: str, questions: Sequence[Question]) -> bytes:
        try:
            state: Any = json.loads(state_json)
        except ValueError:
            state = state_json
        mapped: dict[str, Any] = {}
        for index, question in enumerate(questions):
            entry: dict[str, Any] = {"type": question.type, "instructions": question.text}
            if question.type == "choice":
                entry["criteria"] = {
                    label: (question.criteria[position] or None)
                    if position < len(question.criteria)
                    else None
                    for position, label in enumerate(question.labels)
                }
            elif question.type == "score":
                entry["criteria"] = list(question.labels)
            elif question.criteria and any(question.criteria):
                entry["criteria"] = {
                    "false": question.criteria[0],
                    "true": question.criteria[1],
                }
            mapped[f"q{index}"] = entry
        document = {
            "state": state,
            "model": self.model_id.split(":", 1)[1],
            "questions": mapped,
        }
        return json.dumps(document, ensure_ascii=False).encode("utf-8")

    def _post(self, payload: bytes) -> dict[str, Any]:
        request = urllib.request.Request(
            self._endpoint,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        last: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                self.calls += 1
                with self._opener.open(request, timeout=self._timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                last = error
                if error.code not in _RETRYABLE or attempt == self._retries:
                    raise
            except urllib.error.URLError as error:
                last = error
                if attempt == self._retries:
                    raise
            delay = (2.0**attempt) * 0.25 + random.random() * 0.1
            asyncio_sleep_blocking(delay)
        raise last or RuntimeError("falha remota sem exceção registrada")


def asyncio_sleep_blocking(seconds: float) -> None:
    """Espera bloqueante — já executamos dentro de uma thread dedicada."""
    import time

    time.sleep(seconds)


def _to_logits(question: Question, answer: Any) -> list[float]:
    """Converte a resposta do provedor em logits alinhados aos rótulos."""
    if not isinstance(answer, dict):
        raise ValueError("resposta remota incompleta")
    kind = question.type
    if kind == "noul":
        probability = float(answer["noul"])
        distribution = [1.0 - probability, probability]
    elif kind == "choice":
        table = answer.get("probabilities") or {}
        distribution = [float(table.get(label, 0.0)) for label in question.labels]
    else:
        table = answer.get("probabilities") or {}
        distribution = [
            float(table.get(str(index), 0.0)) for index in range(question.cardinality)
        ]
    total = math.fsum(distribution)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("distribuição remota inválida")
    return [math.log(max(value / total, _MIN_PROBABILITY)) for value in distribution]
