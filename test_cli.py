import contextlib
import io
import json
import unittest
from unittest.mock import patch

from specter_decision import __version__
from specter_decision.cli import main
from specter_decision.service import decode_json


class CLITests(unittest.TestCase):
    def invoke(self, args):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = main(args)
        return code, json.loads(stream.getvalue())

    def test_doctor_offline(self):
        code, result = self.invoke(['doctor'])
        self.assertEqual(code, 0)
        self.assertEqual(result['version'], __version__)
        self.assertFalse(result['production_ready'])
        self.assertFalse(result['model_included'])

    def test_schema_matches_version(self):
        code, result = self.invoke(['schema'])
        self.assertEqual(result['info']['version'], __version__)
        self.assertEqual(result['components']['schemas']['Capabilities']['properties']['release']['const'], __version__)

    def test_missing_token_does_not_read_stdin(self):
        with patch.dict('os.environ', {}, clear=True):
            code, result = self.invoke(['evaluate', '--url', 'http://127.0.0.1:8888'])
        self.assertEqual((code, result['error']), (2, 'MISSING_TOKEN'))

    def test_response_parser_rejects_ambiguous_json(self):
        for data in (b'{"a":1,"a":2}', b'NaN', b'Infinity'):
            with self.assertRaises(ValueError): decode_json(data)


if __name__ == '__main__': unittest.main()
