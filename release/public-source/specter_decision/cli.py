"""Portable CLI. Commands emit JSON; argparse help is documentation, not model output."""
from __future__ import annotations

import argparse
import json
import os
import sys
from importlib.resources import files

from . import __version__
from .client import DecisionClient, DecisionClientError
from .service import MAX_BODY, decode_request


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='specter-decision', description='Typed decision framework; no model included.')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor', help='Offline installation check; no network or model calls')
    sub.add_parser('schema', help='Print the bundled OpenAPI contract')
    evaluate = sub.add_parser('evaluate', help='Submit JSON from stdin to a configured service')
    evaluate.add_argument('--url', required=True, help='HTTPS origin, or localhost HTTP')
    args = parser.parse_args(argv)
    try:
        if args.command == 'doctor':
            body = {'version': __version__, 'python': '.'.join(map(str, sys.version_info[:3])),
                    'status': 'installed', 'model_included': False, 'calibration_included': False,
                    'production_ready': False, 'license_status': 'pending'}
        elif args.command == 'schema':
            body = json.loads(files('specter_decision').joinpath('openapi.json').read_text(encoding='utf-8'))
        else:
            token = os.environ.get('SPECTER_DECISION_TOKEN')
            if not token:
                print(json.dumps({'status': 'error', 'error': 'MISSING_TOKEN'}))
                return 2
            raw = sys.stdin.buffer.read(MAX_BODY + 1)
            state, questions, domain = decode_request(raw)
            body = DecisionClient(args.url, token).decide(state, list(questions), domain=domain)
            print(json.dumps(body, ensure_ascii=True, allow_nan=False))
            return 0 if body['status'] == 'ok' and all(d['status'] == 'ok' for d in body['decisions']) else 1
        print(json.dumps(body, ensure_ascii=True, allow_nan=False))
        return 0
    except DecisionClientError as exc:
        print(json.dumps({'status': 'error', 'error': str(exc)}))
        return 1
    except (ValueError, TypeError, OSError, UnicodeError, RecursionError):
        print(json.dumps({'status': 'error', 'error': 'INVALID_INPUT'}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
