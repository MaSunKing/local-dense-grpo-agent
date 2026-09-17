"""Strict atomic delta schema, conjunction, lineage and no-downgrade tests."""
import copy
import itertools
import unittest
from test_incremental_coverage_v1814 import view, delta
from gain_contract import validate_coverage_delta_result, merge_coverage_delta


class Tests(unittest.TestCase):
    def test_all_eight_atomic_combinations(self):
        names = ('any_requested_option_or_member_present',
                 'requested_outcome_support', 'population_applicable')
        for flags in itertools.product((False, True), repeat=3):
            result = delta(material=all(flags))
            result['coverage'][0].update(zip(names, flags))
            normalized = validate_coverage_delta_result(view('unknown'), result)
            self.assertEqual(normalized['coverage']['R1']['new_material_support'], all(flags))
            merged, _ = merge_coverage_delta(view('unknown'), result)
            self.assertEqual(merged['coverage'][0]['status'], 'partial' if all(flags) else 'unknown')

    def test_legacy_label_is_rejected(self):
        result = delta(material=True)
        result['coverage'][0]['new_material_support'] = True
        with self.assertRaises(ValueError):
            validate_coverage_delta_result(view(), result)

    def test_no_null_or_numeric_boolean(self):
        for bad in (None, 1, 'true'):
            result = delta(material=True)
            result['coverage'][0]['population_applicable'] = bad
            with self.assertRaises(ValueError):
                validate_coverage_delta_result(view(), result)

    def test_complete_requires_all_atoms(self):
        result = delta(material=True, complete=True)
        result['coverage'][0]['requested_outcome_support'] = False
        with self.assertRaises(ValueError):
            validate_coverage_delta_result(view(), result)

    def test_true_support_requires_actual_new_basis(self):
        for refs in ([], ['E1'], ['FOREIGN'], ['E2', 'E2']):
            with self.assertRaises(ValueError):
                validate_coverage_delta_result(view(), delta(material=True, refs=refs))

    def test_partial_inherits_and_contradiction_is_not_gain(self):
        merged, conflicts = merge_coverage_delta(view(), delta(contradiction=True))
        self.assertEqual(merged['coverage'][0]['status'], 'partial')
        self.assertEqual(merged['coverage'][0]['evidence_ids'], ['E1'])
        self.assertTrue(conflicts)


if __name__ == '__main__':
    unittest.main()
