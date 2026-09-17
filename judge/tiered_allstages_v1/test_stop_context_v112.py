
import copy
import unittest
from fixtures import stop_packet
from engine import digest, verify_stop
from pipeline import plan
from input_contract import canonical_stop_context

CFG=dict(model='qwen3.7-max-2026-06-08',temperature=0,
         max_tokens=3000,response_format={'type':'json_object'})

def bound(ctx):
    p,lookup=stop_packet()
    meta=p['records'][0]['runtime_termination']
    event,cap=lookup(meta['event_id'])
    event['stop_context']=copy.deepcopy(ctx)
    p['records'][0]['stop_context']=copy.deepcopy(ctx)
    meta['event_digest']=digest(event)
    def resolver(eid):
        if eid!=event['event_id']:
            raise ValueError('event ID')
        return copy.deepcopy(event),copy.deepcopy(cap)
    return p,resolver

def context():
    return dict(
        remaining_budget=2,voluntary=True,
        available_candidates=[
            dict(id='guide',title='Policy',snippet='Visible summary')],
        failed_reads=[
            dict(candidate_id='older-source',error_type='timeout')],
        task_constraints=['Use official sources'])

class StopFields(unittest.TestCase):
    def assert_blocked(self,ctx):
        p,resolver=bound(ctx)
        # Provenance passes: rejection must come from the semantic-field contract.
        self.assertEqual(verify_stop(p,resolver),'active')
        with self.assertRaises(ValueError):
            plan(p,CFG,resolver)

    def test_candidate_future_fields(self):
        for key in ('future_final','private_score','tool_result'):
            ctx=context()
            ctx['available_candidates'][0][key]='SECRET_FUTURE'
            self.assert_blocked(ctx)

    def test_failed_read_future_fields(self):
        ctx=context()
        ctx['failed_reads'][0]['future_final']='SECRET_FUTURE'
        self.assert_blocked(ctx)

    def test_nested_candidate_value(self):
        ctx=context()
        ctx['available_candidates'][0]['snippet']={'future_final':'SECRET_FUTURE'}
        self.assert_blocked(ctx)

    def test_nontext_constraint(self):
        ctx=context()
        ctx['task_constraints']=[{'future_final':'SECRET_FUTURE'}]
        self.assert_blocked(ctx)

    def test_duplicate_candidate(self):
        ctx=context()
        ctx['available_candidates']*=2
        self.assert_blocked(ctx)

    def test_missing_required_field(self):
        ctx=context()
        del ctx['failed_reads'][0]['error_type']
        self.assert_blocked(ctx)

    def test_valid_canonical_and_no_mutation(self):
        ctx=context()
        p,resolver=bound(ctx)
        before=copy.deepcopy(p)
        view=plan(p,CFG,resolver)['bundle']['views']['stop']
        self.assertEqual(view['stop_context'],ctx)
        self.assertEqual(p,before)
        self.assertNotIn('actual_output',view)
        clean=canonical_stop_context(ctx)
        clean['available_candidates'][0]['title']='Changed'
        self.assertEqual(ctx['available_candidates'][0]['title'],'Policy')

    def test_budget_and_top_level(self):
        for value in (True,-1,'2'):
            ctx=context()
            ctx['remaining_budget']=value
            with self.assertRaises(ValueError):
                canonical_stop_context(ctx)
        ctx=context()
        ctx['future_final']='SECRET_FUTURE'
        with self.assertRaises(ValueError):
            canonical_stop_context(ctx)
