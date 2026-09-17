import copy
import unittest

from active_policy_v9 import PROMPTS
from engine import prepare, validate
from fixtures import case


def row(rid, option, outcome, population, complete, evidence_ids):
    return {
        'id': rid,
        'any_requested_option_or_member_present': option,
        'requested_outcome_support': outcome,
        'population_applicable': population,
        'complete_requirement_support': complete,
        'evidence_ids': evidence_ids,
        'reason': 'test basis',
    }


class AtomicCoverageTests(unittest.TestCase):
    def view(self):
        return prepare(case('evidence')[0])['views']['coverage']

    def test_prompt_forbids_free_status(self):
        self.assertIn('Do not output status', PROMPTS['coverage'])
        self.assertIn('Code, not the Judge, maps', PROMPTS['coverage'])
        self.assertIn('only one requested option', PROMPTS['coverage'])

    def test_code_maps_atomic_rows(self):
        view=self.view()
        result={'coverage':[
            row('Q1',True,True,True,False,['POOL']),
            row('Q2',True,True,True,True,['SAFE']),
        ]}
        checked=validate('coverage',view,result)
        self.assertEqual(checked['coverage']['Q1']['status'],'partial')
        self.assertEqual(checked['coverage']['Q2']['status'],'direct')
        self.assertEqual(checked['score'],.75)

    def test_no_material_support_maps_unknown(self):
        view=self.view()
        result={'coverage':[
            row('Q1',True,False,True,False,[]),
            row('Q2',False,False,False,False,[]),
        ]}
        checked=validate('coverage',view,result)
        self.assertEqual([checked['coverage'][key]['status'] for key in ('Q1','Q2')],
                         ['unknown','unknown'])

    def test_legacy_status_output_rejected(self):
        view=self.view()
        result={'coverage':[
            {'id':'Q1','status':'partial','evidence_ids':['POOL'],'reason':'legacy'},
            {'id':'Q2','status':'direct','evidence_ids':['SAFE'],'reason':'legacy'},
        ]}
        with self.assertRaisesRegex(ValueError,'exact core fields'):
            validate('coverage',view,result)

    def test_complete_without_material_support_rejected(self):
        view=self.view()
        result={'coverage':[
            row('Q1',False,True,True,True,[]),
            row('Q2',True,True,True,True,['SAFE']),
        ]}
        with self.assertRaisesRegex(ValueError,'complete support requires'):
            validate('coverage',view,result)

    def test_partial_requires_visible_evidence_reference(self):
        view=self.view()
        result={'coverage':[
            row('Q1',True,True,True,False,[]),
            row('Q2',True,True,True,True,['SAFE']),
        ]}
        with self.assertRaisesRegex(ValueError,'supported status needs actual evidence'):
            validate('coverage',view,result)


if __name__ == '__main__':
    unittest.main()
