"""Strict transport contract; no execution, dynamic imports, or remote configuration."""
from __future__ import annotations

import asyncio
import hmac
import json
import re
import threading
from collections.abc import Callable

from .engine import Question, SpecterDecisionEngine, validate_input
from . import __version__

MAX_BODY = 262144
ERRORS = frozenset(('NOT_CONFIGURED', 'UNAUTHORIZED', 'INVALID_INPUT', 'TOO_LARGE',
                   'UNSUPPORTED_MEDIA_TYPE', 'UNSUPPORTED_ENCODING', 'NOT_FOUND',
                   'METHOD_NOT_ALLOWED', 'BUSY', 'NOT_READY', 'INTERNAL_ERROR', 'READ_TIMEOUT'))


def error(code: str) -> dict:
    if code not in ERRORS:
        code = 'INTERNAL_ERROR'
    return {'api_version': '1', 'status': 'error', 'error': code, 'decisions': []}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('INVALID_INPUT')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('INVALID_INPUT')


def decode_json(raw: bytes):
    return json.loads(raw.decode('utf-8'), object_pairs_hook=_unique,
                      parse_constant=_reject_constant)


def decode_request(raw: bytes) -> tuple[str | dict, tuple[Question, ...], str]:
    if len(raw) > MAX_BODY:
        raise ValueError('TOO_LARGE')
    try:
        data = decode_json(raw)
        if type(data) is not dict or set(data) != {'state', 'questions', 'domain'}:
            raise ValueError('INVALID_INPUT')
        if type(data['domain']) is not str or not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', data['domain']):
            raise ValueError('INVALID_INPUT')
        if type(data['questions']) is not list or not 1 <= len(data['questions']) <= 128:
            raise ValueError('INVALID_INPUT')
        questions = []
        for q in data['questions']:
            if type(q) is not dict or not {'id', 'type', 'instructions'} <= set(q):
                raise ValueError('INVALID_INPUT')
            if set(q) - {'id', 'type', 'instructions', 'criteria', 'tolerance'}:
                raise ValueError('INVALID_INPUT')
            if 'criteria' in q and type(q['criteria']) is not list:
                raise ValueError('INVALID_INPUT')
            questions.append(Question(q['id'], q['type'], q['instructions'],
                                      tuple(q.get('criteria', [])), q.get('tolerance', .5)))
        _, qs = validate_input(data['state'], questions)
        return data['state'], qs, data['domain']
    except (UnicodeError, TypeError, ValueError, RecursionError, OverflowError):
        raise ValueError('INVALID_INPUT') from None


class DecisionService:
    """Factory returns a NEW engine in each request loop; adapters must be concurrency-safe.

    Token is process configuration, never client-supplied configuration. TLS is required
    for remote consumers. The default service is not ready and has no background thread.
    """
    def __init__(self, *, token: str | None = None,
                 engine_factory: Callable[[str], SpecterDecisionEngine] | None = None,
                 max_requests: int = 2):
        if token is not None and (type(token) is not str or not re.fullmatch(r'[A-Za-z0-9_-]{32,256}', token)):
            raise ValueError('INVALID_TOKEN_CONFIG')
        if type(max_requests) is not int or not 1 <= max_requests <= 16:
            raise ValueError('INVALID_INPUT')
        self._token = token
        self._factory = engine_factory
        self._slots = threading.BoundedSemaphore(max_requests)

    def authorize(self, authorization: str | None) -> tuple[int, dict] | None:
        if self._token is None:
            return 503, error('NOT_CONFIGURED')
        if type(authorization) is not str or not hmac.compare_digest(
                authorization.encode('utf-8'), ('Bearer ' + self._token).encode('ascii')):
            return 401, error('UNAUTHORIZED')
        return None

    def capabilities(self) -> dict:
        return {'api_version': '1', 'release': __version__,
                'backend_configured': self._factory is not None,
                'production_ready': False, 'question_types': ['noul', 'choice', 'score'],
                'text_generation': False, 'action_execution': False,
                'max_body_bytes': MAX_BODY, 'max_questions': 128}

    def evaluate(self, raw: bytes) -> tuple[int, dict]:
        # Admission precedes JSON parsing; HTTP adapter also gates body readers.
        if not self._slots.acquire(blocking=False):
            return 429, error('BUSY')
        try:
            try:
                state, questions, domain = decode_request(raw)
            except ValueError as exc:
                return (413 if str(exc) == 'TOO_LARGE' else 400), error(str(exc))
            if self._factory is None:
                return 503, error('NOT_READY')

            async def run():
                engine = self._factory(domain)
                if not isinstance(engine, SpecterDecisionEngine) or engine.domain != domain:
                    raise ValueError('INVALID_ENGINE_CONFIG')
                return await engine.decide(state, questions)

            try:
                decisions = asyncio.run(run())
                return 200, {'api_version': '1', 'status': 'ok', 'error': None, 'decisions': decisions}
            except Exception:
                return 503, error('INTERNAL_ERROR')
        finally:
            self._slots.release()
