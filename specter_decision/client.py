"""Small standard-library client. No redirects, retries, cookie jar or credential logging."""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .engine import Question, validate_input
from .service import ERRORS, MAX_BODY, decode_request, decode_json


class DecisionClientError(Exception):
    pass


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _finite(x):
    try:
        return type(x) in (int, float) and math.isfinite(x)
    except OverflowError:
        return False


def validate_response(body: object, questions: tuple[Question, ...]) -> dict:
    """Validate request-dependent invariants that generic JSON Schema cannot express."""
    try:
        if type(body) is not dict or set(body) != {'api_version', 'status', 'error', 'decisions'}:
            raise ValueError
        if body['api_version'] != '1' or type(body['decisions']) is not list:
            raise ValueError
        if body['status'] == 'error':
            if body['error'] not in ERRORS or body['decisions'] != []:
                raise ValueError
            return body
        if body['status'] != 'ok' or body['error'] is not None or len(body['decisions']) != len(questions):
            raise ValueError
        for q, item in zip(questions, body['decisions']):
            if type(item) is not dict or set(item) != {'id', 'type', 'status', 'value', 'probabilities', 'confidence', 'legend', 'error'}:
                raise ValueError
            if item['id'] != q.id or item['type'] != q.type or item['legend'] != list(q.labels):
                raise ValueError
            if item['status'] == 'error':
                if item['error'] not in ('INVALID_INPUT', 'UNCALIBRATED', 'TIMEOUT', 'BACKEND_ERROR', 'INVALID_OUTPUT'):
                    raise ValueError
                if any(item[k] is not None for k in ('value', 'probabilities', 'confidence')):
                    raise ValueError
                continue
            ps, confidence, value = item['probabilities'], item['confidence'], item['value']
            if item['status'] != 'ok' or item['error'] is not None or type(ps) is not list or len(ps) != len(q.labels):
                raise ValueError
            if not all(_finite(p) and 0 <= p <= 1 for p in ps) or not math.isclose(sum(ps), 1, abs_tol=1e-9):
                raise ValueError
            if not _finite(confidence) or not 0 <= confidence <= 1:
                raise ValueError
            if q.type == 'choice':
                if value != q.criteria[max(range(len(ps)), key=ps.__getitem__)]:
                    raise ValueError
            else:
                expected = ps[1] if q.type == 'noul' else math.fsum(i * p for i, p in enumerate(ps))
                if not _finite(value) or not math.isclose(value, expected, abs_tol=1e-9):
                    raise ValueError
        return body
    except (ValueError, TypeError, KeyError, OverflowError):
        raise DecisionClientError('INVALID_RESPONSE') from None


class DecisionClient:
    def __init__(self, base_url: str, token: str, *, timeout: float = 10):
        parsed = urlsplit(base_url)
        if (parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment or parsed.path not in ('', '/')
                or (parsed.scheme == 'http' and parsed.hostname not in ('127.0.0.1', '::1', 'localhost'))):
            raise ValueError('HTTPS_REQUIRED_EXCEPT_LOOPBACK')
        if type(token) is not str or not re.fullmatch(r'[A-Za-z0-9_-]{32,256}', token):
            raise ValueError('INVALID_TOKEN_CONFIG')
        if not _finite(timeout) or not 0 < timeout <= 60:
            raise ValueError('INVALID_INPUT')
        self._url, self._token, self._timeout = base_url.rstrip('/') + '/v1/decision/evaluate', token, timeout
        self._opener = build_opener(_NoRedirect())

    def decide(self, state: str | dict, questions: list[Question], *, domain: str) -> dict:
        _, qs = validate_input(state, questions)
        raw = json.dumps({'state': state, 'questions': [asdict(q) for q in qs], 'domain': domain},
                         ensure_ascii=False, allow_nan=False).encode('utf-8')
        if len(raw) > MAX_BODY:
            raise ValueError('TOO_LARGE')
        decode_request(raw)  # Same domain and request rules as the service.
        request = Request(self._url, data=raw, method='POST', headers={
            'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self._token})
        try:
            try:
                response = self._opener.open(request, timeout=self._timeout)
            except HTTPError as exc:
                response = exc
            with response:
                status = response.status
                content = response.read(2 * 1024 * 1024 + 1)
                if len(content) > 2 * 1024 * 1024:
                    raise DecisionClientError('INVALID_RESPONSE')
                body = decode_json(content)
            result = validate_response(body, qs)
            if (status == 200) != (result['status'] == 'ok'):
                raise DecisionClientError('INVALID_RESPONSE')
            return result
        except (URLError, TimeoutError, OSError):
            raise DecisionClientError('TRANSPORT_ERROR') from None
        except (ValueError, UnicodeError, RecursionError):
            raise DecisionClientError('INVALID_RESPONSE') from None
