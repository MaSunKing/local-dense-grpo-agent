import json
import sys
import tempfile
import unittest
from pathlib import Path

from isolated import validate
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'shared'))
from contract_retry import run_contract_retry


def view(*, cited=True, readable=True):
    evidence=[]
    if cited:
        evidence=[{
            'span_id':'E0001','source_id':'S0001','kind':'evidence',
            'text':'Readable result supporting part of the claim.' if readable else '   '
        }]
    return {
        'requirements':[{'id':'Q1','text':'Compare A with B.'}],
        'answer':[{
            'id':'A0001','text':'A differs from B.',
            'actual_citation_ids':['S0001'] if cited else [],
            'allowed_basis_ids':['E0001'] if cited else [],
            'attached_evidence':evidence,
        }],
        'output_row_ids':['A0001'],
    }


def result(verdict,basis=None):
    return {'rows':[{'id':'A0001','verdict':verdict,
                     'basis_ids':[] if basis is None else basis,
                     'reason':'A deterministic test reason.'}]}


class CitationObservabilityTests(unittest.TestCase):
    def test_readable_supported(self):
        self.assertEqual(validate('citation',view(),result('supported',['E0001']))['score'],1.)

    def test_readable_partial(self):
        self.assertEqual(validate('citation',view(),result('partial',['E0001']))['score'],.5)

    def test_readable_mismatch(self):
        self.assertEqual(validate('citation',view(),result('mismatch'))['score'],0.)

    def test_readable_unobservable_rejected(self):
        with self.assertRaisesRegex(ValueError,'visible attached evidence'):
            validate('citation',view(),result('unobservable'))

    def test_uncited_missing(self):
        self.assertEqual(validate('citation',view(cited=False),result('missing'))['score'],0.)

    def test_metadata_only_attachment_unobservable(self):
        self.assertIsNone(validate('citation',view(readable=False),result('unobservable'))['score'])

    def test_invalid_unobservable_gets_one_retry(self):
        calls=[]
        def call(_):
            calls.append(1)
            value=result('unobservable') if len(calls)==1 else result('partial',['E0001'])
            return json.dumps(value)
        with tempfile.TemporaryDirectory() as root:
            outcome=run_contract_retry(
                request={'messages':[{'role':'system','content':'citation contract'}]},
                call=call,validate=lambda value:validate('citation',view(),value),
                audit_dir=root)
            first=json.loads((Path(root)/'attempt-1/receipt.json').read_text())
            second=json.loads((Path(root)/'attempt-2/receipt.json').read_text())
        self.assertEqual(outcome['status'],'validated')
        self.assertEqual(outcome['contract_retries'],1)
        self.assertEqual(first['status'],'contract_rejected')
        self.assertIn('visible attached evidence',first['validator_error'])
        self.assertEqual(second['status'],'validated')
        self.assertNotEqual(first['response_sha256'],second['response_sha256'])

    def test_two_invalid_unobservable_attempts_become_pending(self):
        with tempfile.TemporaryDirectory() as root:
            outcome=run_contract_retry(
                request={'messages':[{'role':'system','content':'citation contract'}]},
                call=lambda _:json.dumps(result('unobservable')),
                validate=lambda value:validate('citation',view(),value),
                audit_dir=root)
        self.assertEqual(outcome['status'],'pending')
        self.assertIsNone(outcome['result'])
        self.assertEqual(len(outcome['attempts']),2)


if __name__=='__main__':unittest.main()
