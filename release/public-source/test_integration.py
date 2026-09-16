import ast
import copy
import http.client
import http.server
import json
import os
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from jsonschema import Draft202012Validator
from specter_decision.client import DecisionClient, DecisionClientError, validate_response
from specter_decision.engine import Question, SpecterDecisionEngine
from specter_decision.http import dispatch
from specter_decision.service import DecisionService, decode_request, MAX_BODY
from test_engine import FixtureCalibration, FixtureModel

TOKEN = 'synthetic_test_token_not_a_secret_000000'
ROOT = Path(__file__).parent
SPEC = json.loads((ROOT / 'specter_decision/openapi.json').read_text(encoding='utf-8'))
REQUEST = {'state': {'ticket': 'example'}, 'domain': 'fixture',
           'questions': [{'id': 'route', 'type': 'choice', 'instructions': 'Route?',
                          'criteria': ['billing', 'support']}]}


def factory(domain):
    return SpecterDecisionEngine(FixtureModel(), FixtureCalibration(), domain=domain)


class ContractTests(unittest.TestCase):
    def test_schema_is_valid(self):
        for schema in SPEC['components']['schemas'].values():
            Draft202012Validator.check_schema(schema)
        Draft202012Validator(SPEC['components']['schemas']['Request']).validate(REQUEST)

    def test_service_success_schema_and_invariants(self):
        code, result = DecisionService(token=TOKEN, engine_factory=factory).evaluate(json.dumps(REQUEST).encode())
        self.assertEqual(code, 200)
        Draft202012Validator(SPEC['components']['schemas']['Response']).validate(result)
        q = Question('route', 'choice', 'Route?', ('billing', 'support'))
        validate_response(result, (q,))
        corrupted = copy.deepcopy(result)
        corrupted['decisions'][0]['value'] = 'invented-label'
        with self.assertRaises(DecisionClientError): validate_response(corrupted, (q,))

    def test_not_ready(self):
        code, body = DecisionService(token=TOKEN).evaluate(json.dumps(REQUEST).encode())
        self.assertEqual((code, body['error']), (503, 'NOT_READY'))
        Draft202012Validator(SPEC['components']['schemas']['Response']).validate(body)

    def test_bad_requests(self):
        for raw in (b'{}', b'{"state":1,"state":2}', b'NaN', b'\xff', b'[' * 1200):
            code, body = DecisionService().evaluate(raw)
            self.assertEqual(code, 400)
            Draft202012Validator(SPEC['components']['schemas']['Response']).validate(body)
        mutated = copy.deepcopy(REQUEST)
        mutated['backend'] = 'arbitrary-module'
        with self.assertRaises(ValueError): decode_request(json.dumps(mutated).encode())
        code, _ = DecisionService().evaluate(b'x' * (MAX_BODY + 1))
        self.assertEqual(code, 413)

    def test_global_admission(self):
        service = DecisionService(token=TOKEN, engine_factory=factory, max_requests=1)
        service._slots.acquire()
        try:
            self.assertEqual(service.evaluate(b'{}')[0], 429)
        finally:
            service._slots.release()

    def test_no_calibration_yields_no_decision(self):
        service = DecisionService(token=TOKEN, engine_factory=lambda d: SpecterDecisionEngine(FixtureModel(), domain=d))
        code, body = service.evaluate(json.dumps(REQUEST).encode())
        self.assertEqual(code, 200)
        self.assertEqual(body['decisions'][0]['error'], 'UNCALIBRATED')
        Draft202012Validator(SPEC['components']['schemas']['Response']).validate(body)

    def test_invalid_backend_config_fails_closed(self):
        service = DecisionService(token=TOKEN, engine_factory=lambda d: None)
        self.assertEqual(service.evaluate(json.dumps(REQUEST).encode())[1]['error'], 'INTERNAL_ERROR')

    def test_client_requires_secure_remote(self):
        with self.assertRaises(ValueError): DecisionClient('http://example.com', TOKEN)
        with self.assertRaises(ValueError): DecisionClient('https://user:secret@example.com', TOKEN)


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = DecisionService(token=TOKEN, engine_factory=factory)
        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            def do_GET(self):
                if not dispatch(self, HTTPTests.service): self.send_error(404)
            do_POST = do_GET
            do_OPTIONS = do_GET
            def log_message(self, *args): pass
        cls.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = 'http://127.0.0.1:' + str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, method='POST', path='/v1/decision/evaluate', body=None, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=3)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            response = conn.getresponse()
            return response.status, dict(response.getheaders()), json.loads(response.read())
        finally:
            conn.close()

    def test_auth_before_body_processing(self):
        code, headers, body = self.request(body='malformed')
        self.assertEqual((code, body['error']), (401, 'UNAUTHORIZED'))
        self.assertNotIn('Access-Control-Allow-Origin', headers)

    def test_capabilities_and_spec(self):
        auth = {'Authorization': 'Bearer ' + TOKEN}
        code, _, body = self.request('GET', '/v1/decision/capabilities', headers=auth)
        self.assertEqual(code, 200)
        self.assertFalse(body['production_ready'])
        Draft202012Validator(SPEC['components']['schemas']['Capabilities']).validate(body)
        code, _, body = self.request('GET', '/v1/decision/openapi.json', headers=auth)
        self.assertEqual(body, SPEC)

    def test_sdk_roundtrip(self):
        client = DecisionClient(self.url, TOKEN)
        result = client.decide('example', [Question('x', 'noul', 'True?')], domain='fixture')
        self.assertEqual(result['decisions'][0]['status'], 'ok')
        client = DecisionClient(self.url, 'wrong_' + TOKEN)
        self.assertEqual(client.decide('x', [Question('x', 'noul', 'True?')], domain='fixture')['error'], 'UNAUTHORIZED')

    def test_method_media_encoding_and_limits(self):
        auth = {'Authorization': 'Bearer ' + TOKEN}
        self.assertEqual(self.request('GET', headers=auth)[0], 405)
        self.assertEqual(self.request(body='{}', headers=auth)[0], 415)
        auth['Content-Type'] = 'application/json'
        auth['Content-Length'] = str(MAX_BODY + 1)
        self.assertEqual(self.request(body=b'', headers=auth)[0], 413)
        del auth['Content-Length']
        auth['Content-Encoding'] = 'gzip'
        self.assertEqual(self.request(body='{}', headers=auth)[0], 415)

    def test_bad_json_and_duplicate_fields(self):
        for raw in ('{"domain":1,"domain":2}', '{', 'NaN'):
            code, _, body = self.request(body=raw, headers={'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json'})
            self.assertEqual(code, 400)
            self.assertEqual(body['error'], 'INVALID_INPUT')


@unittest.skipUnless(os.environ.get('SPECTER_TEST_LOCAL_CORE') == '1', 'Optional local Specter-Core integration')
class CoreHookTests(unittest.TestCase):
    def test_installed_core_routes_before_legacy_body_reader(self):
        core = Path('C:/Specter/Core/specter_core_v3.py')
        tree = ast.parse(core.read_text(encoding='utf-8-sig'))
        handler = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SpecterHandler')
        namespace = {'http': __import__('http'), 'dispatch_decision': lambda obj: True}
        # Compile ONLY the handler class. Never import Core or start its workers/databases.
        exec(compile(ast.Module(body=[handler], type_ignores=[]), '<core-handler-only>', 'exec'), namespace)
        obj = object.__new__(namespace['SpecterHandler'])
        obj.do_POST()  # Succeeds without rfile/headers because the new route short-circuits.
        obj.do_GET()
        obj.do_OPTIONS()

    def test_real_core_handler_loopback(self):
        core = Path('C:/Specter/Core/specter_core_v3.py')
        tree = ast.parse(core.read_text(encoding='utf-8-sig'))
        handler = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SpecterHandler')
        service = DecisionService(token=TOKEN, engine_factory=factory)
        namespace = {'http': __import__('http'), 'dispatch_decision': lambda obj: dispatch(obj, service)}
        exec(compile(ast.Module(body=[handler], type_ignores=[]), '<core-handler-only>', 'exec'), namespace)
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), namespace['SpecterHandler'])
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            client = DecisionClient('http://127.0.0.1:' + str(server.server_port), TOKEN)
            result = client.decide({'x': 1}, [Question('n', 'noul', 'X?')], domain='fixture')
            self.assertEqual(result['decisions'][0]['status'], 'ok')
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)


if __name__ == '__main__': unittest.main()
