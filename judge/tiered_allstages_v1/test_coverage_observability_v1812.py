import copy
import unittest

from active_policy_v10 import PROMPTS
from engine import prepare, validate
from fixtures import case


ATOMS = (
    'any_requested_option_or_member_present',
    'requested_outcome_support',
    'population_applicable',
    'complete_requirement_support',
)


def row(rid, values, evidence_ids=None, reason='test basis'):
    return {
        'id': rid,
        **dict(zip(ATOMS, values)),
        'evidence_ids': list(evidence_ids or []),
        'reason': reason,
    }


class CoverageObservabilityTests(unittest.TestCase):
    def view(self):
        return prepare(case('evidence')[0])['views']['coverage']

    def test_prompt_marks_empty_snapshot_observable(self):
        prompt = PROMPTS['coverage']
        self.assertIn('empty evidence list is itself fully observable', prompt)
        self.assertIn('entire cumulative evidence list', prompt)
        self.assertIn('does not erase material support', prompt)

    def test_null_atoms_rejected_for_materialized_coverage(self):
        view = self.view()
        result = {'coverage': [
            row('Q1', (None, None, None, None)),
            row('Q2', (False, False, False, False)),
        ]}
        with self.assertRaisesRegex(ValueError, 'materialized coverage snapshot'):
            validate('coverage', view, result)

    def test_false_atoms_are_unknown_zero(self):
        view = self.view()
        result = {'coverage': [
            row('Q1', (False, False, False, False)),
            row('Q2', (False, False, False, False)),
        ]}
        checked = validate('coverage', view, result)
        self.assertEqual(checked['score'], 0.0)
        self.assertEqual(
            [checked['coverage'][rid]['status'] for rid in ('Q1', 'Q2')],
            ['unknown', 'unknown'],
        )

    def test_state_truth_can_remain_genuinely_unobservable(self):
        payload, _ = case('state')
        view = prepare(payload)['views']['state_truth']
        result = {'coverage': [
            row(item['id'], (None, None, None, None))
            for item in view['policy_items']
        ]}
        checked = validate('state_truth', view, result)
        self.assertIsNone(checked['score'])


if __name__ == '__main__':
    unittest.main()

