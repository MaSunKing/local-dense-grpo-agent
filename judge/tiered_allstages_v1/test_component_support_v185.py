import unittest

from active_policy_v4 import PROMPTS


class ComponentSupportPromptTests(unittest.TestCase):
    def test_comparison_does_not_gate_material_help(self):
        for task in ('browse', 'state_truth', 'coverage'):
            prompt = PROMPTS[task]
            self.assertIn('material_help does NOT require', prompt)
            self.assertIn('concrete requested outcome', prompt)
            self.assertIn('complete_scope must be false', prompt)
            self.assertIn('not by ranking it against other visible candidates', prompt)

    def test_partial_citation_accepts_component_result(self):
        prompt = PROMPTS['state_citation']
        self.assertIn('When the Agent claims partial', prompt)
        self.assertIn('Do not require the citation', prompt)
        self.assertIn('must not by itself produce an incorrect verdict', prompt)

    def test_rule_remains_scope_safe(self):
        for task in ('browse', 'state_truth', 'coverage'):
            prompt = PROMPTS[task]
            self.assertIn('Never infer the missing arm', prompt)
            self.assertIn('Same-topic mentions without a requested result', prompt)


if __name__ == '__main__':
    unittest.main()
