import unittest
from engine import validate

VIEW={
    'requirements':[{'id':'Q','description':'Compare A with B','weight':1}],
    'evidence':[{'source_id':'E','text':'A plus B versus A'}],
}

class BinaryScopeTests(unittest.TestCase):
    def test_browse_code_maps_three_levels(self):
        base={'target_requirement_ids':['Q']}
        for helpful,complete,score in ((False,False,0),(True,False,.5),(True,True,1)):
            row={'material_help':helpful,'complete_scope':complete,'reason':'x'}
            result=base|{'judgments':{'source_relevance':row,'scope_fidelity':row}}
            self.assertEqual(validate('browse',VIEW,result)['score'],score)
    def test_state_code_maps_three_levels(self):
        for helpful,complete,status in ((False,False,'unknown'),(True,False,'partial'),(True,True,'direct')):
            result={'coverage':[{'id':'Q','material_help':helpful,'complete_scope':complete,
                'evidence_ids':['E'] if helpful else [],'reason':'x'}]}
            out=validate('state_truth',VIEW|{'policy_items':VIEW['requirements']},result)
            self.assertEqual(out['coverage']['Q']['status'],status)
    def test_invalid_combinations_fail_closed(self):
        bad={'coverage':[{'id':'Q','material_help':False,'complete_scope':True,'evidence_ids':[],'reason':'x'}]}
        with self.assertRaises(ValueError):
            validate('state_truth',VIEW|{'policy_items':VIEW['requirements']},bad)

if __name__=='__main__':unittest.main()
