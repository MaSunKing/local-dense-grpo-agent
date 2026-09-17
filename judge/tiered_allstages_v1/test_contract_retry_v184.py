import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'shared'))
from contract_retry import run_contract_retry

REQUEST={'model':'judge','temperature':0,'messages':[{'role':'system','content':'JSON schema'},{'role':'user','content':'same view'}]}

def validate(value):
    if not isinstance(value,dict) or set(value)!={'material_support'}:
        raise ValueError('invalid schema fields')
    if value['material_support'] not in {'yes','no','unobservable'}:
        raise ValueError('invalid enum label')

class RetryTests(unittest.TestCase):
    def test_valid_low_result_is_not_retried(self):
        calls=[]
        def call(request):calls.append(request);return '{"material_support":"no"}'
        with tempfile.TemporaryDirectory() as root:
            out=run_contract_retry(request=REQUEST,call=call,validate=validate,audit_dir=root)
        self.assertEqual(out['status'],'validated');self.assertEqual(len(calls),1)

    def test_contract_error_gets_one_reasoned_retry(self):
        calls=[]
        def call(request):
            calls.append(request)
            return '{"material_support":"maybe"}' if len(calls)==1 else '{"material_support":"yes"}'
        with tempfile.TemporaryDirectory() as root:
            out=run_contract_retry(request=REQUEST,call=call,validate=validate,audit_dir=root)
            self.assertTrue((Path(root)/'attempt-1/response.raw').is_file())
            self.assertTrue((Path(root)/'attempt-2/response.raw').is_file())
        self.assertEqual(out['status'],'validated');self.assertEqual(out['contract_retries'],1)
        self.assertEqual(calls[1]['messages'][:2],REQUEST['messages'])
        self.assertIn('invalid_value',calls[1]['messages'][2]['content'])
        self.assertIn('invalid enum label',calls[1]['messages'][2]['content'])
        self.assertEqual(out['attempts'][0]['validator_error'],'invalid enum label')

    def test_second_contract_error_becomes_pending_not_zero(self):
        calls=[]
        def call(request):calls.append(request);return '{}'
        with tempfile.TemporaryDirectory() as root:
            out=run_contract_retry(request=REQUEST,call=call,validate=validate,audit_dir=root)
        self.assertEqual(len(calls),2);self.assertEqual(out['status'],'pending')
        self.assertIsNone(out['result'])

    def test_transport_failure_is_not_automatically_retried(self):
        calls=[]
        def call(request):calls.append(request);raise TimeoutError('secret-free mock')
        with tempfile.TemporaryDirectory() as root:
            out=run_contract_retry(request=REQUEST,call=call,validate=validate,audit_dir=root)
        self.assertEqual(len(calls),1);self.assertEqual(out['status'],'pending')

if __name__=='__main__':unittest.main()
