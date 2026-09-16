"""Narrow Specter-Core route adapter. Importing this module performs no I/O."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from urllib.parse import urlsplit

from .service import DecisionService, MAX_BODY, error

PREFIX = '/v1/decision'
_service = None
_service_lock = threading.Lock()
_read_slots = threading.BoundedSemaphore(4)


def configure(service: DecisionService) -> None:
    """Call at server startup, from trusted application code only."""
    global _service
    if not isinstance(service, DecisionService):
        raise TypeError('INVALID_SERVICE')
    with _service_lock:
        _service = service


def _get_service():
    global _service
    with _service_lock:
        if _service is None:
            try:
                _service = DecisionService(token=os.environ.get('SPECTER_DECISION_TOKEN'))
            except ValueError:
                _service = DecisionService()  # Fail closed on bad configuration.
        return _service


def _send(handler, code, payload):
    body = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
    handler.close_connection = True
    handler.send_response(code)
    handler.send_header('Content-Type', 'application/json; charset=utf-8')
    handler.send_header('Content-Length', str(len(body)))
    handler.send_header('Cache-Control', 'no-store')
    handler.send_header('X-Content-Type-Options', 'nosniff')
    handler.send_header('Connection', 'close')
    if code == 401:
        handler.send_header('WWW-Authenticate', 'Bearer')
    if code == 429:
        handler.send_header('Retry-After', '1')
    handler.end_headers()
    handler.wfile.write(body)


def dispatch(handler, service: DecisionService | None = None) -> bool:
    """Returns False for unrelated routes, preserving the existing Core handlers."""
    path = urlsplit(handler.path).path
    if path != PREFIX and not path.startswith(PREFIX + '/'):
        return False
    service = service or _get_service()
    try:
        auth = handler.headers.get_all('Authorization', [])
        denial = service.authorize(auth[0] if len(auth) == 1 else None)
        if denial:
            _send(handler, *denial)
            return True
        if handler.command == 'GET' and path == PREFIX + '/capabilities':
            _send(handler, 200, service.capabilities())
            return True
        if handler.command == 'GET' and path == PREFIX + '/openapi.json':
            spec = json.loads(Path(__file__).with_name('openapi.json').read_text(encoding='utf-8'))
            _send(handler, 200, spec)
            return True
        if path != PREFIX + '/evaluate':
            _send(handler, 404, error('NOT_FOUND'))
            return True
        if handler.command != 'POST':
            _send(handler, 405, error('METHOD_NOT_ALLOWED'))
            return True
        if handler.headers.get_content_type() != 'application/json':
            _send(handler, 415, error('UNSUPPORTED_MEDIA_TYPE'))
            return True
        if handler.headers.get('Transfer-Encoding') or handler.headers.get('Content-Encoding'):
            _send(handler, 415, error('UNSUPPORTED_ENCODING'))
            return True
        lengths = handler.headers.get_all('Content-Length', [])
        if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
            _send(handler, 400, error('INVALID_INPUT'))
            return True
        if len(lengths[0]) > 8 or int(lengths[0]) > MAX_BODY:
            _send(handler, 413, error('TOO_LARGE'))
            return True
        size = int(lengths[0])
        if not _read_slots.acquire(blocking=False):
            _send(handler, 429, error('BUSY'))
            return True
        try:
            # Absolute body deadline, not a timeout refreshed by trickling bytes.
            import time
            deadline = time.monotonic() + 5.0
            raw = bytearray()
            while len(raw) < size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                handler.connection.settimeout(remaining)
                part = handler.rfile.read1(min(size - len(raw), 65536))
                if not part:
                    _send(handler, 400, error('INVALID_INPUT'))
                    return True
                raw.extend(part)
            handler.connection.settimeout(5.0)
            _send(handler, *service.evaluate(bytes(raw)))
        finally:
            _read_slots.release()
    except TimeoutError:
        try:
            _send(handler, 408, error('READ_TIMEOUT'))
        except OSError:
            pass
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
        pass
    return True
