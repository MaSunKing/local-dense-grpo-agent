
import copy
import unittest
import torch
from core import digest
from grammar_support import (
    DISTRIBUTION, IMPLEMENTATION, validate_support,
    masked_logits, selected_supported_logps,
)

def record(ids=(0,2), token=2):
    ids = list(ids)
    key = digest(ids)
    body = dict(vocab_size=4, pool={key:ids}, refs=[key])
    schema = {"type":"object"}
    return dict(output_ids=[token],
        sampling=dict(
            distribution=DISTRIBUTION, implementation=IMPLEMENTATION,
            temperature=1.0, top_p=1.0, top_k=0,
            grammar=dict(engine="lm-format-enforcer",version="test",
                         schema=schema,schema_digest=digest(schema))),
        support=dict(body,digest=digest(body)))

class SupportTests(unittest.TestCase):
    def test_actual_distribution_and_gradient(self):
        rec = record()
        validate_support(rec)
        logits = torch.tensor([[1.,8.,3.,9.]], requires_grad=True)
        masked, _ = masked_logits(logits,[0,2])
        expected = torch.softmax(masked, -1)[0,2].log()
        actual = selected_supported_logps(
            logits.unsqueeze(1),torch.tensor([[2]]),[0],rec)[0,0]
        torch.testing.assert_close(actual,expected)
        actual.backward()
        self.assertEqual(float(logits.grad[0,1]),0.)
        self.assertEqual(float(logits.grad[0,3]),0.)

    def test_forced_token_has_zero_logprob(self):
        rec = record((2,),2)
        value = selected_supported_logps(
            torch.randn(1,1,4),torch.tensor([[2]]),[0],rec)
        torch.testing.assert_close(value,torch.zeros_like(value))

    def test_forbidden_token_rejected(self):
        rec = record()
        rec["output_ids"] = [1]
        with self.assertRaises(ValueError): validate_support(rec)

    def test_support_tampering_rejected(self):
        rec = record()
        rec["support"]["vocab_size"] = 5
        with self.assertRaises(ValueError): validate_support(rec)

    def test_schema_tampering_rejected(self):
        rec = record()
        rec["sampling"]["grammar"]["schema"]["type"] = "array"
        with self.assertRaises(ValueError): validate_support(rec)

    def test_empty_support_rejected(self):
        with self.assertRaises(ValueError):
            masked_logits(torch.zeros(1,4), [])

    def test_subset_positions(self):
        rec = record()
        key = digest([1,3])
        rec["support"]["pool"][key] = [1,3]
        rec["support"]["refs"].append(key)
        rec["output_ids"].append(3)
        body = {k:v for k,v in rec["support"].items() if k!="digest"}
        rec["support"]["digest"] = digest(body)
        validate_support(rec)
        logits = torch.tensor([[[2.,4.,8.,6.]]])
        actual = selected_supported_logps(logits,torch.tensor([[3]]),[1],rec)
        expected = 6.-torch.logsumexp(torch.tensor([4.,6.]),0)
        torch.testing.assert_close(actual[0,0],expected)

if __name__ == "__main__":
    unittest.main()
