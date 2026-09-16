"""Roteamento HTTP mínimo e cliente remoto síncrono.

O hook (`dispatch_decision`) é desenhado para ser chamado ANTES do leitor
de corpo legado do Core: ele só responde às três rotas de decisão e
devolve None para qualquer outra coisa, deixando o servidor existente
seguir seu caminho. Sem porta própria, sem CORS aberto, sem redirect.

Rotas:
    POST /v1/decision/evaluate
    GET  /v1/decision/capabilities
    GET  /v1/decision/openapi.json
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

from .service import CONTRACT, DecisionService, ServiceError
from .types import Question

__all__ = ["DecisionClient", "configure", "dispatch", "dispatch_decision", "openapi_document"]

_SERVICE: DecisionService | None = None
_JSON = {"Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store"}
_MAX_BODY = 256 * 1024


def configure(service: DecisionService | None) -> None:
    """Registra (ou remove) o serviço ativo do processo."""
    global _SERVICE
    _SERVICE = service


def openapi_document() -> dict[str, Any]:
    """Carrega o contrato OpenAPI distribuído com o pacote."""
    path = Path(__file__).with_name("openapi.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _response(status: int, payload: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
    body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    headers = dict(_JSON)
    headers["Content-Length"] = str(len(body))
    return status, headers, body


def dispatch_decision(
    method: str, path: str, headers: Mapping[str, str], body: bytes | None = None
) -> tuple[int, dict[str, str], bytes] | None:
    """Trata uma requisição de decisão, ou devolve None se a rota não é nossa.

    Args:
        method: verbo HTTP.
        path: caminho da requisição (query string é ignorada).
        headers: cabeçalhos, com busca case-insensitive.
        body: corpo bruto, no máximo 256 KiB.

    Returns:
        Tripla (status, headers, corpo) ou None quando a rota não pertence
        a esta API.
    """
    clean = urllib.parse.urlparse(path).path.rstrip("/") or "/"
    if not clean.startswith("/v1/decision"):
        return None
    lookup = {key.lower(): value for key, value in headers.items()}

    if method == "OPTIONS":
        return _response(204, {})
    if _SERVICE is None:
        return _response(503, {"status": "error", "error": "NOT_CONFIGURED"})

    try:
        _SERVICE.authorize(lookup.get("authorization"))
        if clean == "/v1/decision/capabilities" and method == "GET":
            return _response(200, {"status": "ok", **_SERVICE.capabilities()})
        if clean == "/v1/decision/openapi.json" and method == "GET":
            return _response(200, openapi_document())
        if clean == "/v1/decision/evaluate" and method == "POST":
            if body is None:
                raise ServiceError("INVALID_INPUT", "corpo ausente", 422)
            if len(body) > _MAX_BODY:
                raise ServiceError("INVALID_INPUT", "corpo acima de 256 KiB", 413)
            try:
                document = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as error:
                raise ServiceError("INVALID_INPUT", "JSON inválido", 422) from error
            envelope = asyncio.run(_SERVICE.evaluate(document))
            return _response(200, envelope)
        return _response(405, {"status": "error", "error": "METHOD_NOT_ALLOWED"})
    except ServiceError as error:
        return _response(
            error.status,
            {"status": "error", "error": error.code, "detail": error.detail},
        )


def dispatch(handler: Any, service: DecisionService | None = None) -> bool:
    """Compatibilidade direta com BaseHTTPRequestHandler do Specter Core.

    Devolve False para rotas que não pertencem a /v1/decision, deixando o
    servidor legado seguir seu fluxo normal.
    """
    path = getattr(handler, "path", "")
    clean = urllib.parse.urlparse(path).path.rstrip("/") or "/"
    if not clean.startswith("/v1/decision"):
        return False

    if service is not None:
        configure(service)

    method = getattr(handler, "command", "GET")
    raw_headers = getattr(handler, "headers", {})
    headers: dict[str, str] = {}
    if hasattr(raw_headers, "items"):
        for k, v in raw_headers.items():
            headers[str(k)] = str(v)

    body: bytes | None = None
    if method in ("POST", "PUT", "PATCH"):
        len_val = headers.get("content-length") or headers.get("Content-Length") or "0"
        try:
            content_length = int(len_val)
        except (ValueError, TypeError):
            content_length = 0
        if content_length > 0 and hasattr(handler, "rfile"):
            body = handler.rfile.read(content_length)
        else:
            body = b""

    result = dispatch_decision(method, path, headers, body)
    if result is None:
        return False

    status_code, resp_headers, resp_body = result
    handler.send_response(status_code)
    for k, v in resp_headers.items():
        handler.send_header(k, v)
    handler.end_headers()
    if resp_body and hasattr(handler, "wfile"):
        try:
            handler.wfile.write(resp_body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
    return True


class DecisionClient:
    """Cliente síncrono do endpoint de decisão.

    HTTPS é obrigatório fora de loopback; redirects são recusados; não há
    retry automático (uma decisão repetida não é idempotente para quem a
    consome). O envelope e cada decisão são validados antes de devolver.
    """

    __slots__ = ("_base", "_token", "_timeout", "_opener")

    def __init__(self, base_url: str, token: str, *, timeout: float = 5.0) -> None:
        """Configura o cliente.

        Args:
            base_url: raiz do serviço, com esquema.
            token: segredo do operador.
            timeout: deadline de rede, em segundos.

        Raises:
            ValueError: esquema não permitido ou token vazio.
        """
        parsed = urllib.parse.urlparse(base_url)
        loopback = parsed.hostname in {"127.0.0.1", "::1", "localhost"}
        if parsed.scheme != "https" and not (parsed.scheme == "http" and loopback):
            raise ValueError("HTTPS obrigatório fora de loopback")
        if not token:
            raise ValueError("token obrigatório")
        self._base = base_url.rstrip("/")
        self._token = token
        self._timeout = float(timeout)

        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args: Any, **kwargs: Any) -> None:
                return None

        self._opener = urllib.request.build_opener(_NoRedirect)

    def capabilities(self) -> dict[str, Any]:
        """Consulta as capacidades declaradas pelo serviço."""
        return self._request("GET", "/v1/decision/capabilities", None)

    def decide(
        self,
        state: Any,
        questions: Sequence[Question],
        *,
        domain: str = "default",
    ) -> dict[str, Any]:
        """Envia estado e perguntas, devolvendo o envelope validado.

        Args:
            state: estado serializável.
            questions: perguntas tipadas.
            domain: domínio lógico.

        Returns:
            Envelope com a lista `answers`.

        Raises:
            ValueError: resposta fora do contrato.
        """
        document = {
            "domain": domain,
            "state": state,
            "questions": [
                {
                    "id": question.id,
                    "type": question.type,
                    "instructions": question.text,
                    "criteria": (
                        {
                            label: (question.criteria[index] if index < len(question.criteria) else None)
                            for index, label in enumerate(question.labels)
                        }
                        if question.type == "choice"
                        else list(question.labels)
                        if question.type == "score"
                        else None
                    ),
                    "tolerance": question.tolerance,
                }
                for question in questions
            ],
        }
        envelope = self._request("POST", "/v1/decision/evaluate", document)
        self._validate(envelope, questions)
        return envelope

    # ------------------------------------------------------------- internos
    def _request(self, method: str, path: str, document: Any) -> dict[str, Any]:
        payload = (
            None
            if document is None
            else json.dumps(document, ensure_ascii=False, allow_nan=False).encode("utf-8")
        )
        request = urllib.request.Request(
            f"{self._base}{path}",
            data=payload,
            method=method,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                raw = response.read(_MAX_BODY + 1)
        except urllib.error.HTTPError as error:
            detail = error.read(_MAX_BODY).decode("utf-8", "replace")
            raise ValueError(f"HTTP {error.code}: {detail[:500]}") from error
        if len(raw) > _MAX_BODY:
            raise ValueError("resposta acima do limite de tamanho")
        body = json.loads(raw.decode("utf-8"))
        if not isinstance(body, dict):
            raise ValueError("envelope não é objeto JSON")
        return body

    @staticmethod
    def _validate(envelope: Mapping[str, Any], questions: Sequence[Question]) -> None:
        if envelope.get("contract") != CONTRACT:
            raise ValueError(f"contrato inesperado: {envelope.get('contract')!r}")
        answers = envelope.get("answers")
        if not isinstance(answers, list) or len(answers) != len(questions):
            raise ValueError("quantidade de respostas diferente da de perguntas")
        for answer, question in zip(answers, questions):
            if answer.get("id") != question.id or answer.get("type") != question.type:
                raise ValueError("resposta fora de ordem ou de tipo")
            if answer.get("status") == "ok":
                probabilities = answer.get("probabilities") or []
                if len(probabilities) != question.cardinality:
                    raise ValueError("cardinalidade inválida na resposta")
                if abs(sum(probabilities) - 1.0) > 1e-6:
                    raise ValueError("distribuição não soma 1")
