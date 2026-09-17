import copy
import json
import unittest

from fixtures import case
from pipeline import finish, plan, validate_task_result


CONFIG = {"model": "judge", "temperature": 0, "max_tokens": 3000,
          "response_format": {"type": "json_object"}}


def atom(complete=False):
    return dict(any_requested_option_or_member_present=True,
                requested_outcome_support=True,
                population_applicable=True,
                complete_requirement_support=complete)


class StateCoalescingTests(unittest.TestCase):
    def test_state_truth_and_citation_share_one_transport_task(self):
        payload, _ = case('state')
        planned = plan(payload, CONFIG)
        self.assertEqual(set(planned['tasks']), {'state_combined'})
        public = json.loads(planned['tasks']['state_combined']['request']['messages'][1]['content'])
        self.assertEqual(set(public), {'state_truth_view', 'state_citation_view'})
        self.assertNotIn('claimed_status', json.dumps(public))

    def test_both_nested_contracts_remain_fail_closed(self):
        payload, _ = case('state')
        planned = plan(payload, CONFIG)
        truth = [dict(id='P1', evidence_ids=['POOL'], reason='partial', **atom(False)),
                 dict(id='P2', evidence_ids=['SAFE'], reason='direct', **atom(True))]
        citations = [dict(id='P1', evidence_ids=['POOL'], reason='partial', **atom(False)),
                     dict(id='P2', evidence_ids=['SAFE'], reason='direct', **atom(True))]
        result={'state_truth':{'coverage':truth}, 'state_citation':{'items':citations}}
        validate_task_result(planned,'state_combined',result)
        completed=finish(planned,{'state_combined':result})
        self.assertEqual(completed['report']['stage'],'state')
        bad=copy.deepcopy(result);bad['state_truth']['coverage'][0]['extra']=1
        with self.assertRaises(ValueError):validate_task_result(planned,'state_combined',bad)

    def test_all_unknown_needs_only_truth_call(self):
        payload, _ = case('state')
        for row in json.loads(payload['records'][0]['raw_completion'])['updates']:
            row['status']='unknown';row['evidence_ids']=[]
        payload['records'][0]['raw_completion']=json.dumps({'updates':[
            {'id':'P1','status':'unknown','evidence_ids':[]},
            {'id':'P2','status':'unknown','evidence_ids':[]}]})
        planned=plan(payload,CONFIG)
        self.assertEqual(set(planned['tasks']),{'state_truth'})


if __name__=='__main__':unittest.main()
