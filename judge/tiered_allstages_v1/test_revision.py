import unittest
from active_policy import PROMPTS,ORIGINAL
from active_policy_v9 import PROMPTS as EFFECTIVE_PROMPTS
from engine import prepare,aggregate,judgment_key
from retest_cases import state_case

class RevisionTests(unittest.TestCase):
    def test_only_checklist_prompt_changed(self):
        self.assertEqual([k for k in PROMPTS if PROMPTS[k]!=ORIGINAL[k]],['checklist'])
    def test_plan_contract_explicit(self):
        self.assertIn('NOT a Final answer',PROMPTS['checklist']);self.assertIn('not yet known',PROMPTS['checklist'])
    def test_state_expected_input_shared(self):
        a,b=prepare(state_case()),prepare(state_case(False))
        self.assertEqual(a['views']['state_truth'],b['views']['state_truth'])
    def test_partial_support_comparison(self):
        values=[]
        for good in (True,False):
            b=prepare(state_case(good))
            partial=dict(any_requested_option_or_member_present=True,requested_outcome_support=True,population_applicable=True,complete_requirement_support=False,evidence_ids=['FLOW'])
            direct=partial|dict(complete_requirement_support=True)
            bad=partial|dict(requested_outcome_support=False,evidence_ids=[])
            results={'state_truth':dict(coverage=[dict(id='P1',reason='one outcome missing',**partial),dict(id='P2',reason='voltage supplied',**direct)]),'state_citation':dict(items=[dict(id='P1',reason='scope',**(partial if good else bad)),dict(id='P2',reason='scope',**direct)])}
            js={k:dict(key=judgment_key('s',k,b['views'][k]),result=r) for k,r in results.items()}
            values.append(aggregate(b,js,'s')['score'])
        self.assertGreater(values[0],values[1])
    def test_effective_comparative_scope_contract(self):
        self.assertIn('one option',EFFECTIVE_PROMPTS['browse'])
        self.assertIn('normally partial',EFFECTIVE_PROMPTS['browse'])
        self.assertIn('any_requested_option_or_member_present',EFFECTIVE_PROMPTS['state_truth'])
        self.assertNotIn('"requested_option_support"',EFFECTIVE_PROMPTS['state_truth'])
        self.assertNotIn('"verdict":"correct|partial|incorrect|unobservable"',EFFECTIVE_PROMPTS['state_citation'])
        self.assertIn('Do not output requested_option_support, verdict',EFFECTIVE_PROMPTS['state_citation'])
        self.assertIn("claimed_status",EFFECTIVE_PROMPTS['state_citation'])
    def test_core_and_auxiliary_field_names_are_explicitly_separated(self):
        for task in ('search','stop'):
            self.assertIn('exactly two fields named verdict and reason',EFFECTIVE_PROMPTS[task])
            self.assertIn('Never output a field named grade inside judgments',EFFECTIVE_PROMPTS[task])
        self.assertIn('four named support atoms and reason',EFFECTIVE_PROMPTS['browse'])
        self.assertIn('Use the field name verdict',EFFECTIVE_PROMPTS['checklist'])
        self.assertIn('Do not output requested_option_support, verdict',EFFECTIVE_PROMPTS['state_citation'])
        self.assertIn('requested_option_support, status',EFFECTIVE_PROMPTS['state_truth'])
        self.assertIn('Do not output status',EFFECTIVE_PROMPTS['coverage'])
if __name__=='__main__':unittest.main()
