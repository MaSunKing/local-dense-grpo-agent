import copy,json,unittest
from engine import prepare,validate,aggregate,judgment_key,digest,coverage_delta
from fixtures import case,stop_packet,cases,SEV

class Tests(unittest.TestCase):
    def test_all_stages_prepare(self):
        for name,p,resolver in cases():
            with self.subTest(name=name):self.assertTrue(prepare(p,resolver)['views'])
    def test_free_float_rejected(self):
        view=prepare(case('search')[0])['views']['search']
        with self.assertRaises(ValueError):validate('search',view,dict(judgments={'gap_relevance':1,'scope_fidelity':1},target_requirement_ids=['Q2']))
    def test_aux_omission_not_mask(self):
        view=prepare(case('search')[0])['views']['search'];result=dict(judgments={d:dict(verdict='correct',reason='x') for d in ('gap_relevance','scope_fidelity')},target_requirement_ids=['Q2'])
        for aux in (None,{},'bad',dict(grade='unobservable',reason='x')):
            self.assertEqual(validate('search',view,result|dict(auxiliary=aux))['score'],1)
    def test_unknown_core_preserved(self):
        view=prepare(case('search')[0])['views']['search'];r=dict(judgments={'gap_relevance':dict(verdict='unobservable',reason='x'),'scope_fidelity':dict(verdict='correct',reason='x')},target_requirement_ids=[])
        out=validate('search',view,r);self.assertIsNone(out['score']);self.assertEqual(out['core']['scope_fidelity'],1)
    def test_incomplete_checklist_fixed_denominator(self):
        view=prepare(case('checklist','bad')[0])['views']['checklist']
        self.assertEqual(len(view['requirements']),3);self.assertEqual(len(view['model_items']),1)
        r=dict(coverage=[dict(id=f'Q{i}',verdict='full' if i==1 else 'missing',item_ids=['P1'] if i==1 else [],reason='x') for i in (1,2,3)],scope=dict(verdict='correct',reason='x'))
        self.assertAlmostEqual(validate('checklist',view,r)['core']['coverage'],1/3)
        r['coverage'].pop()
        with self.assertRaises(ValueError):validate('checklist',view,r)
    def test_combined_item_allowed(self):
        p,_=case('checklist');d=json.loads(p['records'][0]['raw_completion']);d['items']=[dict(id='P1',description='Retention, recovery permissions and guarantee?')];p['records'][0]['raw_completion']=json.dumps(d)
        view=prepare(p)['views']['checklist'];r=dict(coverage=[dict(id=f'Q{i}',verdict='full',item_ids=['P1'],reason='x') for i in (1,2,3)],scope=dict(verdict='correct',reason='x'))
        self.assertEqual(validate('checklist',view,r)['score'],1)
    def test_foreign_model_item_rejected(self):
        view=prepare(case('checklist')[0])['views']['checklist'];r=dict(coverage=[dict(id=f'Q{i}',verdict='full',item_ids=['invented'],reason='x') for i in (1,2,3)],scope=dict(verdict='correct',reason='x'))
        with self.assertRaises(ValueError):validate('checklist',view,r)
    def test_search_only_one_target(self):
        view=prepare(case('search')[0])['views']['search'];r=dict(judgments={d:dict(verdict='correct',reason='x') for d in ('gap_relevance','scope_fidelity')},target_requirement_ids=['Q2'])
        self.assertEqual(validate('search',view,r)['score'],1)
    def test_future_fields_not_sent(self):
        p,_=case('browse');base=prepare(p)['views'];p['records'][0]['future_final']='future';p['records'][0]['later_tool_output']='new information'
        self.assertEqual(base,prepare(p)['views'])
    def test_candidate_body_rejected(self):
        p,_=case('browse');p['records'][0]['visible_context']['candidates'][0]['body']='future document'
        with self.assertRaises(ValueError):prepare(p)
    def test_unseen_browse_rejected(self):
        p,_=case('browse');d=json.loads(p['records'][0]['raw_completion']);d['arguments']['candidate_id']='unseen';p['records'][0]['raw_completion']=json.dumps(d)
        with self.assertRaises(ValueError):prepare(p)
    def test_state_truth_independent_of_prediction(self):
        a,b=(prepare(case('state',v)[0]) for v in ('good','bad'))
        self.assertEqual(a['views']['state_truth'],b['views']['state_truth'])
        self.assertNotEqual(a['views']['state_citation'],b['views']['state_citation'])
    def test_no_previous_state_truth_leak(self):
        p,_=case('state');a=prepare(p)['views']['state_truth'];p['records'][0]['previous_state']={'P1':'direct'}
        self.assertEqual(a,prepare(p)['views']['state_truth'])
    def test_missing_state_item_rejected(self):
        p,_=case('state');d=json.loads(p['records'][0]['raw_completion']);d['updates'].pop();p['records'][0]['raw_completion']=json.dumps(d)
        with self.assertRaises(ValueError):prepare(p)
    def test_state_cannot_rewrite_scope(self):
        p,_=case('state');d=json.loads(p['records'][0]['raw_completion']);d['updates'][0]['description']='pooled';p['records'][0]['raw_completion']=json.dumps(d)
        with self.assertRaises(ValueError):prepare(p)
    def test_future_state_citation(self):
        p,_=case('state');d=json.loads(p['records'][0]['raw_completion']);d['updates'][0]['evidence_ids']=['future'];p['records'][0]['raw_completion']=json.dumps(d)
        with self.assertRaises(ValueError):prepare(p)
    def test_wrong_attached_state_citation(self):
        v=prepare(case('state')[0])['views']['state_citation'];r=dict(items=[dict(id='P1',verdict='correct',evidence_ids=['SAFE'],reason='x'),dict(id='P2',verdict='correct',evidence_ids=['SAFE'],reason='x')])
        with self.assertRaises(ValueError):validate('state_citation',v,r)
    def test_empty_evidence_snapshot(self):
        p,_=case('evidence');p['records'][0]['evidence']=[];p['records'][0]['evidence_snapshot']['digest']=digest([])
        v=prepare(p)['views']['coverage'];r=dict(coverage=[dict(id=f'Q{i}',any_requested_option_or_member_present=False,requested_outcome_support=False,population_applicable=False,complete_requirement_support=False,evidence_ids=[],reason='x') for i in (1,2)])
        self.assertEqual(validate('coverage',v,r)['score'],0)
    def test_snapshot_tamper(self):
        p,_=case('state');p['records'][0]['evidence'][0]['text']='changed'
        with self.assertRaises(ValueError):prepare(p)
    def test_stop_needs_resolver(self):
        p,_=stop_packet()
        with self.assertRaises(ValueError):prepare(p)
    def test_stop_forged_verified_flag(self):
        p,r=stop_packet();p['records'][0]['runtime_termination']=dict(verified=True,type='active')
        with self.assertRaises(ValueError):prepare(p,r)
    def test_forced_stop_not_score(self):
        p,r=stop_packet(termination='quota_exhausted');b=prepare(p,r)
        self.assertFalse(b['views']);self.assertIsNone(aggregate(b,{},'s',r)['score'])
    def test_stop_context_tamper(self):
        p,r=stop_packet();p['records'][0]['stop_context']['remaining_budget']=999
        with self.assertRaises(ValueError):prepare(p,r)
    def test_stop_capture_tamper(self):
        p,r=stop_packet();p['records'][0]['raw_completion']='changed'
        with self.assertRaises(ValueError):prepare(p,r)
    def test_stop_task_tamper(self):
        p,r=stop_packet();p['requirements'][0]['description']='new'
        with self.assertRaises(ValueError):prepare(p,r)
    def test_scorer_binding(self):
        b=prepare(case('search')[0]);j={'search':dict(key=judgment_key('old','search',b['views']['search']),result={})}
        with self.assertRaises(ValueError):aggregate(b,j,'new')

class CoverageTests(unittest.TestCase):
    def row(self,variant='good'):
        b=prepare(case('evidence',variant)[0]);v=b['views']['coverage']
        r=dict(coverage=[
            dict(id='Q1',any_requested_option_or_member_present=True,requested_outcome_support=True,population_applicable=True,complete_requirement_support=False,evidence_ids=['POOL'],reason='scope'),
            dict(id='Q2',any_requested_option_or_member_present=variant=='good',requested_outcome_support=variant=='good',population_applicable=variant=='good',complete_requirement_support=variant=='good',evidence_ids=['SAFE'] if variant=='good' else [],reason='x')])
        js={'coverage':dict(key=judgment_key('s','coverage',v),result=r)}
        return dict(bundle=b,report=aggregate(b,js,'s'))
    def test_unchanged_zero(self):
        a=self.row();self.assertEqual(coverage_delta(a,a,[])['delta'],0)
    def test_new_evidence_not_automatic_reward(self):
        d=coverage_delta(self.row('bad'),self.row(),['SAFE']);self.assertEqual(d['delta'],.5);self.assertFalse(d['reward_export_authorized'])
    def test_deleted_no_new_fails(self):
        with self.assertRaises(ValueError):coverage_delta(self.row(),self.row('bad'),[])
    def test_changed_snapshot_fails(self):
        a=self.row();b=copy.deepcopy(a);b['bundle']['payload']['records'][0]['evidence'][0]['text']='tampered'
        with self.assertRaises(ValueError):coverage_delta(a,b,[])
    def test_conflict_rejected(self):
        a=self.row();b=copy.deepcopy(a);b['report']['coverage']['Q1']['status']='direct'
        with self.assertRaises(ValueError):coverage_delta(a,b,[])
    def test_state_cannot_mint_gain(self):
        x=dict(bundle=prepare(case('state')[0]),report={})
        with self.assertRaises(ValueError):coverage_delta(x,x,[])

if __name__=='__main__':unittest.main()
