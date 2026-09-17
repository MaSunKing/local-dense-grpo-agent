import copy
import unittest
from pipeline import plan, finish
from fixtures import case, stop_packet

CONFIG = dict(model='qwen3.7-max-2026-06-08', temperature=0,
              max_tokens=3000, response_format={'type':'json_object'})

class Portable(unittest.TestCase):
    def test_forced_stop_no_api(self):
        payload, resolver = stop_packet(termination='quota_exhausted')
        p = plan(payload, CONFIG, resolver)
        self.assertEqual(p['tasks'], {})
        self.assertFalse(finish(p, {}, resolver)['training_ready'])

    def test_unbound_config_rejected(self):
        payload, resolver = case('search')
        with self.assertRaises(ValueError):
            plan(payload, CONFIG | {'seed': 1}, resolver)

    def test_missing_result_rejected(self):
        payload, resolver = case('search')
        p = plan(payload, CONFIG, resolver)
        self.assertTrue(p['tasks'])
        with self.assertRaises(ValueError):
            finish(p, {}, resolver)

    def test_tampered_plan_rejected(self):
        payload, resolver = case('search')
        p = copy.deepcopy(plan(payload, CONFIG, resolver))
        p['profile']['scorer'] = 'forged'
        with self.assertRaises(ValueError):
            finish(p, {}, resolver)

    def test_json_instruction_all_requests(self):
        for kind in ('checklist', 'search', 'browse', 'state', 'evidence'):
            payload, resolver = case(kind)
            p = plan(payload, CONFIG, resolver)
            for task in p['tasks'].values():
                self.assertIn('json', task['request']['messages'][0]['content'].lower())

if __name__ == '__main__':
    unittest.main()
