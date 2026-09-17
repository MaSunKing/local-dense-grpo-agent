import copy
import json
import unittest
from pipeline import plan, finish
from engine import digest, task_scope_digest
from fixtures import case, pack, stop_packet
from retest_cases import state_case

CFG = dict(model='qwen3.7-max-2026-06-08', temperature=0, max_tokens=3000, response_format={'type':'json_object'})

def arithmetic():
    p=pack('final','Risk decreases from 10% to 7%. Calculate absolute risk reduction.',
        [dict(id='Q',description='Report absolute risk reduction in percentage points.',weight=1)], [],
        '<answer>Absolute risk reduction is 3 percentage points.</answer>')
    p['evaluation_task_mode']='self_contained'
    registry={p['question_id']:dict(mode='self_contained',task_scope_digest=task_scope_digest(p))}
    return p,registry

def final_results(planned, factual='supported'):
    views=planned['bundle']['final']['views']; aid=views['completeness']['answer'][0]['id']
    basis=views['fidelity']['evidence'][0]['span_id']
    return dict(completeness=dict(rows=[dict(id='Q',verdict='full',answer_ids=[aid],reason='answers requested calculation')]),
                fidelity=dict(rows=[dict(id=aid,verdict=factual,basis_ids=[basis],reason='checked against question-given numbers')]))

def state_results(planned, truths=('partial','direct')):
    def signal(s):
        if s=='unobservable':return (None,)*4
        if s=='unknown':return False,False,False,False
        return True,True,True,s=='direct'
    rows=[]
    for i,s in enumerate(truths):
        option,outcome,population,complete=signal(s)
        rows.append(dict(id=f'P{i+1}',any_requested_option_or_member_present=option,
            requested_outcome_support=outcome,population_applicable=population,
            complete_requirement_support=complete,
            evidence_ids=[] if s in {'unknown','unobservable'} else ['FLOW'],reason='test evidence'))
    logical={'state_truth':dict(coverage=rows)}
    if 'state_citation' in planned['bundle']['views']:
        citation=[]
        for r in planned['bundle']['views']['state_citation']['items']:
            citation.append(dict(id=r['id'],any_requested_option_or_member_present=True,
                requested_outcome_support=True,population_applicable=True,
                complete_requirement_support=r['claimed_status']=='direct',
                evidence_ids=['FLOW'],reason='test'))
        logical['state_citation']=dict(items=citation)
    if 'state_combined' in planned['tasks']:
        return {'state_combined':logical}
    return logical

class ModeTests(unittest.TestCase):
    def test_self_contained_end_to_end(self):
        p,m=arithmetic();planned=plan(p,CFG,task_modes=m)
        self.assertEqual(set(planned['tasks']),{'completeness','fidelity'})
        out=finish(planned,final_results(planned),task_modes=m)['report']
        self.assertEqual(out['task_mode'],'self_contained')
        self.assertEqual(out['structurally_excluded'],['citation_support'])
        self.assertIsNone(out['core']['citation_support']);self.assertEqual(out['score'],1)
        self.assertEqual(out['scorer'],planned['profile']['scorer'])
        self.assertEqual(out['binding_digest'],planned['bundle']['final']['bound']['binding_digest'])
        self.assertFalse(out['compiler_connected']);self.assertFalse(out['training_ready'])
    def test_false_fact_still_penalized(self):
        p,m=arithmetic();p['records'][0]['raw_completion']='<answer>Absolute risk reduction is 30 percentage points.</answer>'
        planned=plan(p,CFG,task_modes=m)
        out=finish(planned,final_results(planned,'contradicted'),task_modes=m)['report']
        self.assertAlmostEqual(out['score'],.4/.75)
    def test_agent_declared_mode_not_trusted(self):
        p,m=arithmetic()
        with self.assertRaises(ValueError):plan(p,CFG)
    def test_changed_scope_rejected(self):
        p,m=arithmetic();p['requirements'][0]['description']='Different task'
        with self.assertRaises(ValueError):plan(p,CFG,task_modes=m)
    def test_finish_requires_same_external_registry(self):
        p,m=arithmetic();planned=plan(p,CFG,task_modes=m)
        with self.assertRaises(ValueError):finish(planned,final_results(planned))
    def test_conflicting_declaration_rejected(self):
        p,m=arithmetic();p['evaluation_task_mode']='evidence_grounded'
        with self.assertRaises(ValueError):plan(p,CFG,task_modes=m)
    def test_extra_citation_result_rejected(self):
        p,m=arithmetic();planned=plan(p,CFG,task_modes=m)
        with self.assertRaises(ValueError):finish(planned,final_results(planned)|{'citation':{}},task_modes=m)
    def test_external_evidence_disallows_self_contained(self):
        p,m=arithmetic();p['records'][0]['evidence']=[dict(source_id='EXT',text='External claim')]
        with self.assertRaises(ValueError):plan(p,CFG,task_modes=m)
    def test_frozen_mode_without_payload_declaration(self):
        p,m=arithmetic();del p['evaluation_task_mode']
        self.assertNotIn('citation',plan(p,CFG,task_modes=m)['tasks'])
    def test_evidence_grounded_citation_retained(self):
        p,m=arithmetic();del p['evaluation_task_mode']
        self.assertIn('citation',plan(p,CFG)['tasks'])

class StateTests(unittest.TestCase):
    def unknown_plan(self):
        p=state_case();p['records'][0]['raw_completion']=json.dumps(dict(updates=[dict(id=k,status='unknown',evidence_ids=[]) for k in ('P1','P2')]))
        return plan(p,CFG)
    def test_all_wrong_unknown_zero(self):
        planned=self.unknown_plan();r=finish(planned,state_results(planned))['report']
        self.assertEqual(r['score'],0);self.assertNotIn('scope_fidelity',r['core'])
        self.assertEqual(r['citation_diagnostics'],{'P1':None,'P2':None})
    def test_correct_unknown_full(self):
        planned=self.unknown_plan();r=finish(planned,state_results(planned,('unknown','unknown')))['report']
        self.assertEqual(r['score'],1);self.assertNotIn('state_citation',planned['tasks'])
    def test_half_correct_unknown_half(self):
        planned=self.unknown_plan();r=finish(planned,state_results(planned,('unknown','direct')))['report']
        self.assertEqual(r['score'],.5)
    def test_unobservable_not_invented(self):
        planned=self.unknown_plan();r=finish(planned,state_results(planned,('unobservable','direct')))['report']
        self.assertIsNone(r['score'])
    def test_correct_claims_full(self):
        planned=plan(state_case(),CFG)
        self.assertEqual(finish(planned,state_results(planned))['report']['score'],1)
    def test_citation_cannot_rescue_wrong_status(self):
        planned=plan(state_case(),CFG)
        self.assertEqual(finish(planned,state_results(planned,('direct','partial')))['report']['score'],0)

class InputTests(unittest.TestCase):
    def test_unknown_evidence_keys_rejected_all_stages(self):
        for kind in ('search','browse','state','evidence'):
            for field in ('future_final','old_score','arbitrary_hidden_field'):
                p,_=case(kind);p['records'][0]['evidence'][0][field]='HIDDEN_SENTINEL'
                p['records'][0]['evidence_snapshot']['digest']=digest(p['records'][0]['evidence'])
                with self.subTest(kind=kind,field=field),self.assertRaises(ValueError):plan(p,CFG)
    def test_header_injection_rejected(self):
        p,_=case('search');sid=p['records'][0]['evidence'][0]['source_id']
        for headers in ({'future_answer':'HIDDEN'}, {sid:{'future_answer':'HIDDEN'}}, {sid:{'title':{'future_answer':'HIDDEN'}}}):
            p['records'][0]['source_headers']=headers
            with self.assertRaises(ValueError):plan(p,CFG)
    def test_allowed_metadata_preserved_original_unchanged(self):
        p,_=case('search');r=p['records'][0];r['evidence'][0]['title']='Source title'
        r['source_headers']={r['evidence'][0]['source_id']:{'published_at':'2026-01-01'}}
        r['evidence_snapshot']['digest']=digest(r['evidence']);before=copy.deepcopy(p)
        view=plan(p,CFG)['bundle']['views']['search']
        self.assertEqual(p,before);self.assertEqual(view['source_headers'],r['source_headers'])
    def test_final_unknown_evidence_rejected(self):
        p,m=arithmetic();del p['evaluation_task_mode']
        p['records'][0]['evidence']=[dict(source_id='X',text='Text',future_final='HIDDEN')]
        with self.assertRaises(ValueError):plan(p,CFG)
    def test_stop_output_not_visible_still_bound(self):
        p,resolver=stop_packet();planned=plan(p,CFG,resolver)
        self.assertNotIn('actual_output',planned['bundle']['views']['stop'])
        self.assertNotIn('FINAL_READY',json.dumps(planned['tasks']))
        p['records'][0]['raw_completion']='Persuasive replacement'
        with self.assertRaises(ValueError):plan(p,CFG,resolver)

if __name__=='__main__': unittest.main()
