import json
import tempfile
import unittest
from pathlib import Path

from validated_judge_cache import load_validated, run_cached


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.request = {"model": "m", "messages": [{"role": "user", "content": "x"}]}

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def validate(result):
        if result != {"value": "ok"}:
            raise ValueError("invalid")
        return result

    def test_second_exact_request_uses_zero_calls(self):
        calls = []
        def call(_):
            calls.append(1); return json.dumps({"value": "ok"})
        first = run_cached(cache_dir=self.root, scorer="s", task="t",
            request=self.request, call=call, validate=self.validate,
            audit_dir=self.root / "audit-1")
        second = run_cached(cache_dir=self.root, scorer="s", task="t",
            request=self.request, call=lambda _: self.fail("cache miss"),
            validate=self.validate, audit_dir=self.root / "audit-2")
        self.assertEqual(len(calls), 1)
        self.assertFalse(first["cache_hit"]); self.assertTrue(second["cache_hit"])
        self.assertEqual(first["result"], second["result"])

    def test_scorer_or_request_change_misses(self):
        calls=[]
        def call(_):calls.append(1);return json.dumps({"value":"ok"})
        run_cached(cache_dir=self.root,scorer="s",task="t",request=self.request,
                   call=call,validate=self.validate,audit_dir=self.root/"a")
        self.assertIsNone(load_validated(cache_dir=self.root,scorer="s2",task="t",
                          request=self.request,validate=self.validate))
        changed=self.request|{"max_tokens":1}
        self.assertIsNone(load_validated(cache_dir=self.root,scorer="s",task="t",
                          request=changed,validate=self.validate))

    def test_invalid_result_is_not_cached(self):
        outcome=run_cached(cache_dir=self.root,scorer="s",task="t",request=self.request,
            call=lambda _:json.dumps({"value":"bad"}),validate=self.validate,
            audit_dir=self.root/"bad")
        self.assertNotEqual(outcome["status"],"validated")
        self.assertIsNone(load_validated(cache_dir=self.root,scorer="s",task="t",
                          request=self.request,validate=self.validate))
        receipt=json.loads((self.root/'bad/final-receipt.json').read_text())
        self.assertEqual(receipt['status'],'pending')
        self.assertEqual(receipt['attempt_count'],2)
        self.assertEqual(receipt['validator_error'],'invalid')
        self.assertIsNotNone(receipt['first_invalid_response_sha256'])
        self.assertIsNone(receipt['final_valid_response_sha256'])

    def test_retry_receipt_preserves_both_response_hashes(self):
        calls=[]
        def call(_):
            calls.append(1)
            return json.dumps({'value':'bad' if len(calls)==1 else 'ok'})
        outcome=run_cached(cache_dir=self.root,scorer='s',task='t',request=self.request,
            call=call,validate=self.validate,audit_dir=self.root/'retry')
        receipt=json.loads((self.root/'retry/final-receipt.json').read_text())
        self.assertEqual(outcome['status'],'validated')
        self.assertEqual(receipt['attempt_count'],2)
        self.assertEqual(receipt['validator_error'],'invalid')
        self.assertIsNotNone(receipt['first_invalid_response_sha256'])
        self.assertIsNotNone(receipt['final_valid_response_sha256'])
        self.assertNotEqual(receipt['first_invalid_response_sha256'],
                            receipt['final_valid_response_sha256'])
        self.assertEqual(receipt['scorer'],'s')

    def test_tamper_is_rejected(self):
        result=run_cached(cache_dir=self.root,scorer="s",task="t",request=self.request,
            call=lambda _:json.dumps({"value":"ok"}),validate=self.validate,
            audit_dir=self.root/"good")
        path=next(self.root.rglob(result["cache_key"]+".json"))
        row=json.loads(path.read_text());row["result"]={"value":"bad"}
        path.write_text(json.dumps(row))
        with self.assertRaises(ValueError):
            load_validated(cache_dir=self.root,scorer="s",task="t",
                           request=self.request,validate=self.validate)


if __name__=="__main__":unittest.main()
