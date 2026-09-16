"""Kernel de decisão: estado -> snapshot -> amostrador paralelo -> projeção tipada.

Pipeline (idêntico ao desenho 0.1, com o caminho quente reescrito):

    state -> snapshot canônico -> validação -> planejamento
          -> [prefill único] -> logits por pergunta (lote ou fan-out)
          -> temperatura versionada -> softmax por segmento
          -> projeção tipada -> validação -> Decision

O que mudou em relação a 0.2 (e por quê):

* **Lote nativo.** Se o backend expõe `logits_batch`, o kernel faz UMA
  chamada com todas as perguntas: o estado é codificado uma vez só. Esse é
  o ganho estrutural do desenho System One; fan-out por pergunta vira o
  caminho de compatibilidade.
* **Deduplicação por assinatura.** Perguntas com conteúdo idêntico (comum
  em fan-out especulativo) são computadas uma vez e projetadas N vezes.
* **Admissão consciente do deadline.** Trabalho que já nasceu vencido não
  é iniciado; o resultado é TIMEOUT sem custo de backend.
* **Gate de calibração por pergunta.** Uma pergunta sem artefato compatível
  retorna UNCALIBRATED sem impedir as demais de decidir.
* **Semáforo por (instância, event loop).** Recriado quando o loop muda,
  eliminando a classe de bugs de semáforo compartilhado entre loops.
* **Projeção sem alocação supérflua** e soma com `math.fsum`.

O kernel continua não gerando texto, não executando ações, não fazendo
rede e não registrando o conteúdo do estado.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Iterable, Sequence

from .canonical import Snapshot, canonical_snapshot, fingerprint_of
from .numerics import argmax, expectation, softmax, validate_simplex
from .protocols import Backend, Calibration
from .types import (
    DEFAULT_LIMITS,
    Decision,
    ErrorCode,
    Limits,
    Question,
)

__all__ = ["SpecterDecisionEngine"]

_ENGINE_CONTRACT = "specter-decision/0.3"


def _reject(reason: str) -> None:
    raise ValueError(f"INVALID_INPUT: {reason}")


class _Plan:
    """Plano de execução de uma requisição (estrutura interna, não exportada)."""

    __slots__ = ("questions", "snapshot", "groups", "order", "results", "gated")

    def __init__(self, questions: tuple[Question, ...], snapshot: Snapshot) -> None:
        self.questions = questions
        self.snapshot = snapshot
        self.groups: dict[str, list[int]] = {}
        self.order: list[Question] = []
        self.results: dict[str, Any] = {}
        self.gated: set[int] = set()


class SpecterDecisionEngine:
    """Avalia perguntas independentes sobre um estado imutável.

    A instância é reutilizável e segura para chamadas concorrentes dentro do
    mesmo event loop: o limite de concorrência é global por instância.

    Attributes:
        backend: implementação de `Backend` (e opcionalmente `BatchBackend`).
        calibration: artefato de calibração ou None (tudo UNCALIBRATED).
        domain: domínio lógico vinculado ao artefato.
        concurrency: teto de chamadas simultâneas ao backend.
        timeout: deadline absoluto por requisição, em segundos (None desliga).
        limits: limites duros de validação.
        abstain_below: se definido, decisões com confiança menor viram
            ABSTAINED em vez de valor (política de abstenção explícita).
        receipts: inclui `receipt` (hash auditável) em cada decisão.
    """

    __slots__ = (
        "backend",
        "calibration",
        "domain",
        "concurrency",
        "timeout",
        "limits",
        "abstain_below",
        "receipts",
        "_semaphore",
        "_semaphore_loop",
        "_batch",
        "_counters",
    )

    def __init__(
        self,
        backend: Backend,
        calibration: Calibration | None = None,
        *,
        domain: str = "default",
        concurrency: int = 8,
        timeout: float | None = 2.0,
        limits: Limits = DEFAULT_LIMITS,
        abstain_below: float | None = None,
        receipts: bool = False,
    ) -> None:
        if backend is None:
            raise ValueError("backend é obrigatório")
        if concurrency < 1:
            raise ValueError("concurrency deve ser >= 1")
        if timeout is not None and (timeout <= 0 or timeout != timeout):
            raise ValueError("timeout deve ser > 0 ou None")
        if abstain_below is not None and not 0.0 <= abstain_below <= 1.0:
            raise ValueError("abstain_below deve estar em [0,1]")
        self.backend = backend
        self.calibration = calibration
        self.domain = domain
        self.concurrency = concurrency
        self.timeout = timeout
        self.limits = limits
        self.abstain_below = abstain_below
        self.receipts = receipts
        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None
        self._batch = callable(getattr(backend, "logits_batch", None))
        self._counters = {"requests": 0, "questions": 0, "deduplicated": 0, "batched": 0}

    # ------------------------------------------------------------------ API
    async def decide(
        self, state: Any, questions: Iterable[Question]
    ) -> list[Decision]:
        """Avalia todas as perguntas sobre o mesmo snapshot do estado.

        Args:
            state: estrutura JSON serializável (str, dict, list, número...).
            questions: perguntas tipadas, ids únicos, no máximo `max_questions`.

        Returns:
            Lista de decisões na ordem original das perguntas.

        Raises:
            ValueError: prefixada com INVALID_INPUT quando a requisição é
                inválida. Requisição inválida nunca chega ao backend.
            asyncio.CancelledError: propagada do chamador, cancelando o
                trabalho em voo.
        """
        plan = self._plan(state, questions)
        self._counters["requests"] += 1
        self._counters["questions"] += len(plan.questions)
        if plan.order:
            await self._execute(plan)
        return [
            self._project(index, question, plan)
            for index, question in enumerate(plan.questions)
        ]

    def decide_sync(self, state: Any, questions: Iterable[Question]) -> list[Decision]:
        """Versão síncrona para aplicações não-async.

        Raises:
            RuntimeError: se chamada de dentro de um event loop ativo.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.decide(state, questions))
        raise RuntimeError("decide_sync não pode ser usado dentro de um event loop")

    def capabilities(self) -> dict[str, Any]:
        """Descreve a configuração corrente sem alegar prontidão produtiva."""
        return {
            "contract": _ENGINE_CONTRACT,
            "domain": self.domain,
            "model_id": getattr(self.backend, "model_id", "unknown"),
            "batch_backend": self._batch,
            "calibrated": self.calibration is not None,
            "concurrency": self.concurrency,
            "timeout_s": self.timeout,
            "abstain_below": self.abstain_below,
            "limits": {
                "max_questions": self.limits.max_questions,
                "max_choice_options": self.limits.max_choice_options,
                "max_score_levels": self.limits.max_score_levels,
                "max_state_bytes": self.limits.max_state_bytes,
            },
            "question_types": ["noul", "choice", "score"],
            "error_codes": sorted(ErrorCode.ALL),
            "production_ready": False,
            "counters": dict(self._counters),
        }

    # ------------------------------------------------------------ planejamento
    def _plan(self, state: Any, questions: Iterable[Question]) -> _Plan:
        items = tuple(questions)
        if not items:
            _reject("nenhuma pergunta")
        if len(items) > self.limits.max_questions:
            _reject(f"{len(items)} perguntas acima do limite {self.limits.max_questions}")
        seen_ids: set[str] = set()
        for question in items:
            if not isinstance(question, Question):
                _reject("cada pergunta deve ser uma instância de Question")
            if question.id in seen_ids:
                _reject(f"id duplicado: {question.id!r}")
            seen_ids.add(question.id)

        snapshot = canonical_snapshot(
            state,
            max_bytes=self.limits.max_state_bytes,
            max_depth=self.limits.max_state_depth,
        )
        plan = _Plan(items, snapshot)

        model_id = getattr(self.backend, "model_id", "unknown")
        for index, question in enumerate(items):
            if not self._supports(question, model_id):
                plan.gated.add(index)
                continue
            bucket = plan.groups.get(question.signature)
            if bucket is None:
                plan.groups[question.signature] = [index]
                plan.order.append(question.anonymous())
            else:
                bucket.append(index)
                self._counters["deduplicated"] += 1
        return plan

    def _supports(self, question: Question, model_id: str) -> bool:
        """Gate de calibração; qualquer exceção do artefato vira UNCALIBRATED."""
        if self.calibration is None:
            return False
        try:
            return bool(self.calibration.supports(self.domain, model_id, question))
        except Exception:
            return False

    # -------------------------------------------------------------- execução
    async def _execute(self, plan: _Plan) -> None:
        loop = asyncio.get_running_loop()
        deadline = None if self.timeout is None else loop.time() + self.timeout
        signatures = list(plan.groups.keys())
        if self._batch:
            self._counters["batched"] += 1
            await self._execute_batch(plan, signatures, loop, deadline)
            return
        await self._execute_fanout(plan, signatures, loop, deadline)

    async def _execute_batch(
        self,
        plan: _Plan,
        signatures: list[str],
        loop: asyncio.AbstractEventLoop,
        deadline: float | None,
    ) -> None:
        remaining = None if deadline is None else deadline - loop.time()
        if remaining is not None and remaining <= 0:
            self._fill(plan, signatures, ErrorCode.TIMEOUT)
            return
        call = self.backend.logits_batch(plan.snapshot.json, tuple(plan.order))
        try:
            rows = await (
                call if remaining is None else asyncio.wait_for(call, remaining)
            )
        except (TimeoutError, asyncio.TimeoutError):
            self._fill(plan, signatures, ErrorCode.TIMEOUT)
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            self._fill(plan, signatures, ErrorCode.BACKEND_ERROR)
            return
        if not isinstance(rows, (list, tuple)) or len(rows) != len(signatures):
            self._fill(plan, signatures, ErrorCode.INVALID_OUTPUT)
            return
        for signature, row in zip(signatures, rows):
            plan.results[signature] = row

    async def _execute_fanout(
        self,
        plan: _Plan,
        signatures: list[str],
        loop: asyncio.AbstractEventLoop,
        deadline: float | None,
    ) -> None:
        semaphore = self._acquire_semaphore(loop)
        tasks = [
            loop.create_task(self._call_one(plan.snapshot.json, question, semaphore))
            for question in plan.order
        ]
        remaining = None if deadline is None else max(0.0, deadline - loop.time())
        try:
            done, pending = await asyncio.wait(tasks, timeout=remaining)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        if pending:
            for task in pending:
                task.cancel()
            # Aguardar as tarefas canceladas garante que os `finally` do
            # backend rodem antes de devolvermos o controle ao chamador.
            await asyncio.gather(*pending, return_exceptions=True)
        for signature, task in zip(signatures, tasks):
            if task in pending:
                plan.results[signature] = ErrorCode.TIMEOUT
                continue
            try:
                error = task.exception()
            except asyncio.CancelledError:
                plan.results[signature] = ErrorCode.TIMEOUT
                continue
            if error is None:
                plan.results[signature] = task.result()
            elif isinstance(error, (TimeoutError, asyncio.TimeoutError)):
                plan.results[signature] = ErrorCode.TIMEOUT
            else:
                plan.results[signature] = ErrorCode.BACKEND_ERROR

    async def _call_one(
        self, state_json: str, question: Question, semaphore: asyncio.Semaphore | None
    ) -> Sequence[float]:
        if semaphore is None:
            return await self.backend.logits(state_json, question)
        async with semaphore:
            return await self.backend.logits(state_json, question)

    def _acquire_semaphore(
        self, loop: asyncio.AbstractEventLoop
    ) -> asyncio.Semaphore | None:
        """Semáforo global por instância, recriado se o event loop mudou."""
        if self._semaphore is None or self._semaphore_loop is not loop:
            self._semaphore = asyncio.Semaphore(self.concurrency)
            self._semaphore_loop = loop
        return self._semaphore

    @staticmethod
    def _fill(plan: _Plan, signatures: list[str], code: str) -> None:
        for signature in signatures:
            plan.results[signature] = code

    # -------------------------------------------------------------- projeção
    def _project(self, index: int, question: Question, plan: _Plan) -> Decision:
        if index in plan.gated:
            return self._error(question, ErrorCode.UNCALIBRATED)
        raw = plan.results.get(question.signature)
        if isinstance(raw, str):
            return self._error(question, raw)
        if raw is None:
            return self._error(question, ErrorCode.BACKEND_ERROR)

        cardinality = question.cardinality
        if not isinstance(raw, (list, tuple)) or len(raw) != cardinality:
            return self._error(question, ErrorCode.INVALID_OUTPUT)

        try:
            temperature = float(self.calibration.temperature(question))  # type: ignore[union-attr]
            probabilities = softmax(raw, temperature)
        except Exception:
            return self._error(question, ErrorCode.INVALID_OUTPUT)
        if not validate_simplex(probabilities, size=cardinality):
            return self._error(question, ErrorCode.INVALID_OUTPUT)

        kind = question.type
        if kind == "noul":
            value: Any = probabilities[1]
        elif kind == "choice":
            value = question.labels[argmax(probabilities)]
        else:
            value = expectation(probabilities)
            if not 0.0 <= value <= cardinality - 1:
                return self._error(question, ErrorCode.INVALID_OUTPUT)

        try:
            confidence = float(
                self.calibration.confidence(question, probabilities, value)  # type: ignore[union-attr]
            )
        except Exception:
            return self._error(question, ErrorCode.INVALID_OUTPUT)
        if not 0.0 <= confidence <= 1.0 or confidence != confidence:
            return self._error(question, ErrorCode.INVALID_OUTPUT)
        if self.abstain_below is not None and confidence < self.abstain_below:
            return self._error(question, ErrorCode.ABSTAINED)

        decision: Decision = {
            "id": question.id,
            "type": kind,
            "status": "ok",
            "value": value,
            "probabilities": probabilities,
            "confidence": confidence,
            "legend": list(question.labels),
            "error": None,
        }
        if self.receipts:
            decision["receipt"] = self._receipt(decision, plan.snapshot)  # type: ignore[typeddict-unknown-key]
        return decision

    def _error(self, question: Question, code: str) -> Decision:
        decision: Decision = {
            "id": question.id,
            "type": question.type,
            "status": "error",
            "value": None,
            "probabilities": None,
            "confidence": None,
            "legend": list(question.labels),
            "error": code,
        }
        return decision

    @staticmethod
    def _receipt(decision: Decision, snapshot: Snapshot) -> str:
        """Hash auditável da decisão, sem expor o conteúdo do estado."""
        payload = json.dumps(
            {
                "state": snapshot.fingerprint,
                "id": decision["id"],
                "type": decision["type"],
                "value": decision["value"],
                "probabilities": decision["probabilities"],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return fingerprint_of(payload, prefix="specter-receipt/1")
