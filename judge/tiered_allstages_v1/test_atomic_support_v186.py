import unittest

from engine import atomic_support_signal, validate


ATOMS = dict(
    any_requested_option_or_member_present=True,
    requested_outcome_support=True,
    population_applicable=True,
    complete_requirement_support=False,
    reason='one requested component and outcome are supported',
)


class AtomicSupportTests(unittest.TestCase):
    def test_code_derives_partial_without_other_comparison_arm(self):
        self.assertEqual(atomic_support_signal(ATOMS), (.5, 'partial'))

    def test_missing_comparison_does_not_erase_material_support(self):
        view={
            'requirements':[{'id':'Q','description':'Compare A with B for outcome O','weight':1}],
            'decision_contract':'atomic_component_outcome_population_v2',
        }
        result={'target_requirement_ids':['Q'],'judgments':{
            'source_relevance':dict(ATOMS),
            'scope_fidelity':dict(ATOMS),
        }}
        self.assertEqual(validate('browse',view,result)['score'],.5)

    def test_state_atoms_map_to_unknown_partial_direct(self):
        view={
            'policy_items':[{'id':'Q','description':'Compare A with B for outcome O','weight':1}],
            'evidence':[{'source_id':'E','text':'A has outcome O'}],
            'decision_contract':'atomic_component_outcome_population_v2',
        }
        cases=(
            (dict(ATOMS,any_requested_option_or_member_present=False,requested_outcome_support=False,
                  population_applicable=False,evidence_ids=[]),'unknown'),
            (dict(ATOMS,evidence_ids=['E']),'partial'),
            (dict(ATOMS,complete_requirement_support=True,evidence_ids=['E']),'direct'),
        )
        for row,status in cases:
            out=validate('state_truth',view,{'coverage':[dict(id='Q',**row)]})
            self.assertEqual(out['coverage']['Q']['status'],status)

    def test_partial_citation_does_not_require_missing_arm(self):
        view={
            'items':[{'id':'Q','description':'Compare A with B','claimed_status':'partial',
                      'evidence':[{'source_id':'E','text':'A has outcome O'}]}],
            'decision_contract':'atomic_component_outcome_population_v2',
        }
        row=dict(id='Q',evidence_ids=['E'],**ATOMS)
        self.assertEqual(validate('state_citation',view,{'items':[row]})['items']['Q'],1.)

    def test_invalid_complete_without_atoms_rejected(self):
        bad=dict(ATOMS,any_requested_option_or_member_present=False,complete_requirement_support=True)
        with self.assertRaises(ValueError):
            atomic_support_signal(bad)


if __name__ == '__main__':
    unittest.main()
