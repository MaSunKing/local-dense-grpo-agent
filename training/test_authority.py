import json
from pathlib import Path
import tempfile
import unittest

from authority import TrustedAuthorityBundle,build_for_test,REGISTRIES


class Tests(unittest.TestCase):
    def test_constructor_uses_the_exact_already_read_batch_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);batch=root/'batch.json';batch.write_text('{"x":1}',encoding='utf-8')
            original=batch.read_bytes();authority=root/'authority.json'
            authority.write_text(json.dumps(build_for_test(batch,{name:{} for name in REGISTRIES})),encoding='utf-8')
            batch.write_text('{"x":2}',encoding='utf-8')
            loaded=TrustedAuthorityBundle(authority,original)
            self.assertIsNone(loaded.tool_execution('missing'))

    def test_exact_batch_binding_and_registry_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);batch=root/'batch.json';batch.write_text('{"x":1}',encoding='utf-8')
            registries={name:{} for name in REGISTRIES}
            registries['tool_executions']['e']={'value':1}
            authority=root/'authority.json'
            authority.write_text(json.dumps(build_for_test(batch,registries)),encoding='utf-8')
            loaded=TrustedAuthorityBundle(authority,batch.read_bytes())
            row=loaded.tool_execution('e');row['value']=2
            self.assertEqual(loaded.tool_execution('e')['value'],1)
            batch.write_text('{"x":2}',encoding='utf-8')
            with self.assertRaises(ValueError):TrustedAuthorityBundle(authority,batch.read_bytes())

    def test_authority_identity_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);batch=root/'batch.json';batch.write_text('{}',encoding='utf-8')
            value=build_for_test(batch,{name:{} for name in REGISTRIES})
            value['registries']['stage_scores']['x']={}
            authority=root/'authority.json';authority.write_text(json.dumps(value),encoding='utf-8')
            with self.assertRaises(ValueError):TrustedAuthorityBundle(authority,batch.read_bytes())


if __name__=='__main__':unittest.main()
