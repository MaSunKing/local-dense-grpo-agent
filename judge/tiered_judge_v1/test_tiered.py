import copy
import unittest
from reward_policy import scalar,POLICY
from final_contract import prepare,validate
from fixtures import case,parse_answer

class Tiered(unittest.TestCase):
    def setUp(self):
        self.packet=prepare(case(),parse_answer)
        basis=[s['span_id'] for s in self.packet['context']['basis_spans'] if s['source_id']=='D1']
        self.result=dict(units=[dict(answer_id='A0001',factual_support='supported',citation_support='supported',factual_basis_ids=basis,citation_basis_ids=basis,reason='bound')],content=[dict(requirement_id=q,status='full',answer_ids=['A0001'],reason='present') for q in ('Q1','Q2')])
    def test_optional_missing(self):self.assertEqual(validate(self.packet,self.result,parse_answer)['score'],1)
    def test_alias_only(self):
        r=dict(core_units=self.result['units'],core_content=self.result['content'])
        self.assertEqual(validate(self.packet,r,parse_answer)['score'],1)
    def test_alias_conflict(self):
        with self.assertRaises(ValueError):validate(self.packet,self.result|dict(core_units=self.result['units']),parse_answer)
    def test_optional_invalid(self):
        for x in (None,{},'bad',dict(grade=42,reason='x'),dict(grade='unobservable',reason='x')):
            self.assertEqual(validate(self.packet,self.result|{'auxiliary':x},parse_answer)['score'],1)
    def test_required_limit_retained(self):
        self.result['content'][1].update(status='missing',answer_ids=[])
        self.result['auxiliary']=dict(grade='good',reason='clear')
        r=validate(self.packet,self.result,parse_answer)
        self.assertAlmostEqual(r['score'],.8);self.assertEqual(r['auxiliary_adjustment'],0)
    def test_attachment(self):
        self.result['units'][0]['citation_basis_ids']=[s['span_id'] for s in self.packet['context']['basis_spans'] if s['source_id']=='D2']
        with self.assertRaises(ValueError):validate(self.packet,self.result,parse_answer)
    def test_unknown_id(self):
        self.result['units'][0]['factual_basis_ids']=['future']
        with self.assertRaises(ValueError):validate(self.packet,self.result,parse_answer)
    def test_mutation(self):
        self.packet['context']['requirements'][0]['description']='changed'
        with self.assertRaises(ValueError):validate(self.packet,self.result,parse_answer)
    def test_duplicate(self):
        self.result['content'][1]=copy.deepcopy(self.result['content'][0])
        with self.assertRaises(ValueError):validate(self.packet,self.result,parse_answer)
    def test_core_unknown(self):
        self.result['units'][0]['factual_support']='unobservable'
        r=validate(self.packet,self.result,parse_answer)
        self.assertIsNone(r['score']);self.assertEqual(r['core']['completeness'],1)
    def test_stages(self):
        for stage,p in POLICY['stages'].items():
            with self.subTest(stage=stage):
                r=scalar(stage,{k:.5 for k in p['core_weights']},'good')
                self.assertLessEqual(r['score'],.525);self.assertFalse(r['training_ready'])
    def test_aux_bound(self):
        c=dict(completeness=.5,fidelity=.5,citation_support=.5)
        self.assertAlmostEqual(scalar('final',c,'good')['score']-scalar('final',c,'poor')['score'],.05)
    def test_no_rescue(self):
        c=dict(completeness=1,fidelity=0,citation_support=1)
        self.assertEqual(scalar('final',c,'good',material_core_error=True)['auxiliary_adjustment'],0)
    def test_bad_core(self):
        for x in (True,float('nan'),-1,2):
            with self.assertRaises(ValueError):scalar('final',dict(completeness=x,fidelity=1,citation_support=1))
    def test_exclusion_not_inferred(self):
        c=dict(completeness=1,fidelity=1,citation_support=None)
        self.assertIsNone(scalar('final',c)['score'])
        self.assertEqual(scalar('final',c,excluded=['citation_support'])['score'],1)
    def test_invalid_policy(self):
        for cap in (True,float('nan'),.5):
            p=copy.deepcopy(POLICY);p['auxiliary_adjustment_max']=cap
            with self.assertRaises(ValueError):scalar('final',dict(completeness=1,fidelity=1,citation_support=1),policy=p)
    def test_contract_failure(self):self.assertIsNone(scalar('stop',dict(stop_appropriateness=1),contract_valid=False)['score'])

if __name__=='__main__':unittest.main()
