import copy,unittest
from isolated import packets,validate,aggregate,digest
from cases import case

class Tests(unittest.TestCase):
    def test_citation_invariance(self):
        a,b=packets(case()),packets(case('wrong_citation'))
        for d in ('completeness','fidelity'):self.assertEqual(a['views'][d],b['views'][d])
        self.assertNotEqual(a['views']['citation'],b['views']['citation'])
        self.assertNotEqual(a['bound']['binding_digest'],b['bound']['binding_digest'])
    def test_no_leak(self):
        v=packets(case())['views']
        self.assertEqual(set(v['completeness']),{'question','requirements','constraints','answer','output_row_ids'})
        for d in ('completeness','fidelity'):
            self.assertNotIn('<cite',str(v[d]));self.assertNotIn('actual_citation_ids',str(v[d]))
    def test_no_citation_label_leak(self):
        a=case();b=copy.deepcopy(a);b['records'][0]['raw_completion']=b['records'][0]['raw_completion'].replace('>Policy</cite>','>Other display label</cite>')
        self.assertEqual(packets(a)['views'],packets(b)['views'])
    def test_fact_change_not_reused(self):
        for d in ('completeness','fidelity','citation'):self.assertNotEqual(packets(case())['views'][d],packets(case('wrong_fact'))['views'][d])
    def test_delete_change_not_reused(self):self.assertNotEqual(packets(case())['views']['completeness'],packets(case('missing'))['views']['completeness'])
    def test_evidence_change_content_stable(self):
        b=case();b['records'][0]['evidence'][0]['text']='Changed factual evidence.'
        a,c=packets(case()),packets(b)
        self.assertEqual(a['views']['completeness'],c['views']['completeness']);self.assertNotEqual(a['views']['fidelity'],c['views']['fidelity'])
    def test_requirement_change(self):
        b=case();b['requirements'][0]['description']='A different request.'
        self.assertNotEqual(packets(case())['views']['completeness'],packets(b)['views']['completeness'])
    def test_wrong_citation_basis_rejected(self):
        v=packets(case('wrong_citation'))['views']['citation']
        with self.assertRaises(ValueError):validate('citation',v,dict(rows=[dict(id='A0001',verdict='supported',basis_ids=['E1_0001'],reason='x')]))
    def test_citation_declares_span_ids_not_source_ids(self):
        v=packets(case())['views']['citation'];target=v['answer'][0]
        expected=[row['span_id'] for row in target['attached_evidence']]
        self.assertEqual(target['allowed_basis_ids'],expected)
        self.assertTrue(set(target['actual_citation_ids']).isdisjoint(expected))
        with self.assertRaises(ValueError):
            validate('citation',v,dict(rows=[dict(id=target['id'],verdict='supported',
                basis_ids=[target['actual_citation_ids'][0]],reason='source IDs are invalid span references')]))
    def test_foreign_id(self):
        v=packets(case())['views']['fidelity']
        with self.assertRaises(ValueError):validate('fidelity',v,dict(rows=[dict(id='A0001',verdict='supported',basis_ids=['future'],reason='x')]))
    def test_missing_is_not_unknown(self):
        v=packets(case())['views']['completeness'];r=dict(rows=[dict(id=q['id'],verdict='missing',answer_ids=[],reason='x') for q in v['requirements']])
        self.assertEqual(validate('completeness',v,r)['score'],0)
        for q in r['rows']:q['verdict']='unobservable'
        self.assertIsNone(validate('completeness',v,r)['score'])
    def test_duplicate(self):
        v=packets(case())['views']['completeness'];r=dict(rows=[dict(id='R1',verdict='full',answer_ids=['A0001'],reason='x')]*3)
        with self.assertRaises(ValueError):validate('completeness',v,r)
    def test_binding_tamper(self):
        b=packets(case());b['mapping'][0]['plain']='tampered'
        with self.assertRaises(ValueError):aggregate(b,{},'s')
    def test_scorer_cache_key(self):
        v=packets(case())['views']['completeness']
        self.assertNotEqual(digest(dict(scorer='a',view=v)),digest(dict(scorer='b',view=v)))
    def test_bound_removal(self):
        b=packets(case());m=b['mapping'][0];raw=m['raw_span']['quote'];s=m['removed'][0]
        self.assertEqual(raw[s['start']:s['end']],s['text'])
        self.assertEqual((raw[:s['start']]+raw[s['end']:]).strip(),m['plain'])
    def test_exact_duplicate_cannot_dilute_material_error(self):
        answer=[dict(id=f'A{i}',text='A supported policy statement.') for i in range(9)]
        answer.append(dict(id='BAD',text='A different unsupported assertion.'))
        view=dict(question='q',requirements=[dict(id='R',description='d')],constraints={},
                  answer=answer,evidence=[dict(span_id='E',source_id='S',kind='evidence',text='support')],
                  output_row_ids=[row['id'] for row in answer])
        rows=[dict(id=f'A{i}',verdict='supported',basis_ids=['E'],reason='supported') for i in range(9)]
        rows.append(dict(id='BAD',verdict='unsupported',basis_ids=[],reason='unsupported'))
        self.assertEqual(validate('fidelity',view,dict(rows=rows))['score'],.5)
    def test_output_row_contract(self):
        views=packets(case())['views']
        for dim,view in views.items():
            target='requirements' if dim=='completeness' else 'answer'
            self.assertEqual(view['output_row_ids'],[row['id'] for row in view[target]])
        view=copy.deepcopy(views['fidelity']);e=view['evidence'][0]['span_id']
        view['answer'].append(dict(id='A0002',text='A second material claim.'))
        view['output_row_ids'].append('A0002')
        rows=[dict(id=aid,verdict='supported',basis_ids=[e],reason='x') for aid in reversed(view['output_row_ids'])]
        with self.assertRaises(ValueError):validate('fidelity',view,dict(rows=rows))
if __name__=='__main__':unittest.main()
