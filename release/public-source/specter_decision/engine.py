"""Typed decision kernel. No text generation, network, model weights or side effects."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Literal, Protocol, Sequence, TypedDict, cast

Kind = Literal['noul', 'choice', 'score']
Error = Literal['INVALID_INPUT', 'UNCALIBRATED', 'TIMEOUT', 'BACKEND_ERROR', 'INVALID_OUTPUT']


@dataclass(frozen=True)
class Question:
    id: str
    type: Kind
    instructions: str
    criteria: tuple[str, ...] = ()
    tolerance: float = 0.5

    @property
    def labels(self) -> tuple[str, ...]:
        return ('false', 'true') if self.type == 'noul' else self.criteria

    @property
    def signature(self) -> str:
        data = asdict(self)
        del data['id']  # IDs route results; they must not affect predictions.
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


class _Success(TypedDict):
    id: str
    status: Literal['ok']
    probabilities: list[float]
    confidence: float
    legend: list[str]
    error: None


class NoulDecision(_Success):
    type: Literal['noul']
    value: float


class ChoiceDecision(_Success):
    type: Literal['choice']
    value: str


class ScoreDecision(_Success):
    type: Literal['score']
    value: float


class ErrorDecision(TypedDict):
    id: str
    type: Kind
    status: Literal['error']
    value: None
    probabilities: None
    confidence: None
    legend: list[str]
    error: Error


Decision = NoulDecision | ChoiceDecision | ScoreDecision | ErrorDecision


class Backend(Protocol):
    """Trusted non-generative adapter; must honor cancellation and question isolation."""
    model_id: str

    async def logits(self, state_json: str, question: Question) -> Sequence[float]: ...


class Calibration(Protocol):
    """Inject only an independently validated, versioned artifact; not model self-report."""
    def supports(self, model_id: str, domain: str, question_signature: str) -> bool: ...

    def temperature(self, question_signature: str) -> float: ...

    def confidence(self, question: Question, probabilities: tuple[float, ...],
                   value: float | str) -> float: ...


def _number(value: object) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _valid_question(q: object) -> bool:
    return (
        isinstance(q, Question) and type(q.id) is str and 1 <= len(q.id) <= 128
        and type(q.instructions) is str and 1 <= len(q.instructions) <= 8192
        and bool(q.instructions.strip())
        and q.type in ('noul', 'choice', 'score')
        and type(q.criteria) is tuple
        and all(type(x) is str and 1 <= len(x) <= 4096 and bool(x.strip()) for x in q.criteria)
        and len(set(q.criteria)) == len(q.criteria)
        and ((q.type == 'noul' and not q.criteria)
             or (q.type == 'choice' and 2 <= len(q.criteria) <= 255)
             or (q.type == 'score' and 2 <= len(q.criteria) <= 10))
        and _number(q.tolerance) and q.tolerance > 0
    )


def _error(q: Question, code: Error) -> Decision:
    return dict(id=q.id, type=q.type, status='error', value=None,
                probabilities=None, confidence=None, legend=list(q.labels), error=code)


def _json_value(v: object) -> bool:
    if v is None or type(v) in (str, bool):
        return True
    if type(v) in (int, float):
        return _number(v)
    if type(v) is list:
        return all(_json_value(x) for x in v)
    if type(v) is dict:
        return all(type(k) is str and _json_value(x) for k, x in v.items())
    return False


def validate_input(state: str | dict, questions: Sequence[Question]) -> tuple[str, tuple[Question, ...]]:
    try:
        qs = tuple(questions)
        valid = (type(state) in (str, dict) and _json_value(state)
                 and 1 <= len(qs) <= 128 and all(_valid_question(q) for q in qs)
                 and len({q.id for q in qs}) == len(qs))
        if not valid:
            raise ValueError('INVALID_INPUT')
        snapshot = json.dumps(state, sort_keys=True, ensure_ascii=False, allow_nan=False)
        total = len(snapshot.encode('utf-8')) + sum(
            len(json.dumps(asdict(q), ensure_ascii=False).encode('utf-8')) for q in qs)
        if total > 262144:
            raise ValueError('INVALID_INPUT')
        return snapshot, qs
    except (TypeError, ValueError, RecursionError, OverflowError):
        raise ValueError('INVALID_INPUT') from None


class SpecterDecisionEngine:
    def __init__(self, backend: Backend, calibration: Calibration | None = None,
                 *, domain: str, concurrency: int = 8, timeout: float = 2.0):
        if type(concurrency) is not int or not 1 <= concurrency <= 128:
            raise ValueError('INVALID_INPUT')
        if not _number(timeout) or not 0 < timeout <= 30 or type(domain) is not str or not 1 <= len(domain) <= 128:
            raise ValueError('INVALID_INPUT')
        self.backend, self.calibration, self.domain = backend, calibration, domain
        self.timeout = timeout
        self._slots = asyncio.Semaphore(concurrency)

    async def decide(self, state: str | dict, questions: Sequence[Question]) -> list[Decision]:
        """Invalid requests raise a fixed error code; accepted requests yield typed outcomes.

        One immutable JSON snapshot, independent question calls, stable result order.
        A semaphore bounds concurrency across calls on this engine's event loop.
        """
        snapshot, qs = validate_input(state, questions)
        return list(await asyncio.gather(*(self._timed(snapshot, q) for q in qs)))

    async def _timed(self, snapshot: str, q: Question) -> Decision:
        try:
            # Deadline includes queueing, not just provider time.
            return await asyncio.wait_for(self._one(snapshot, q), timeout=self.timeout)
        except TimeoutError:
            return _error(q, 'TIMEOUT')
        except Exception:
            return _error(q, 'BACKEND_ERROR')

    async def _one(self, snapshot: str, q: Question) -> Decision:
        c = self.calibration
        if c is None or not c.supports(self.backend.model_id, self.domain, q.signature):
            return _error(q, 'UNCALIBRATED')
        async with self._slots:
            # Do not send the routing ID or any sibling question to the model.
            model_q = Question('_', q.type, q.instructions, q.criteria, q.tolerance)
            logits = tuple(await self.backend.logits(snapshot, model_q))
            t = c.temperature(q.signature)
            if (len(logits) != len(q.labels) or not all(_number(x) for x in logits)
                    or not _number(t) or t <= 0):
                return _error(q, 'INVALID_OUTPUT')
            peak = max(logits)
            weights = tuple(math.exp((x - peak) / t) for x in logits)
            total = math.fsum(weights)
            ps = tuple(x / total for x in weights)
            if not all(_number(p) and 0 <= p <= 1 for p in ps):
                return _error(q, 'INVALID_OUTPUT')
            value: float | str
            if q.type == 'noul':
                value = ps[1]
            elif q.type == 'choice':
                value = q.criteria[max(range(len(ps)), key=ps.__getitem__)]
            else:
                value = math.fsum(i * p for i, p in enumerate(ps))
            confidence = c.confidence(q, ps, value)
            if not _number(confidence) or not 0 <= confidence <= 1:
                return _error(q, 'INVALID_OUTPUT')
            return cast(Decision, dict(id=q.id, type=q.type, status='ok', value=value,
                        probabilities=list(ps), confidence=confidence,
                        legend=list(q.labels), error=None))
