import copy
import unittest

from gain_contract import (canonical_coverage_result, coverage_delta_key, digest,
                           make_coverage_receipt, merge_coverage_delta,
                           validate_coverage_receipt)


class ReceiptIdentityTests(unittest.TestCase):
    def result(self, reason="First explanation", status="partial", refs=None):
        return {"coverage": [{"id": "R1", "status": status,
                              "evidence_ids": refs if refs is not None else ["E1"],
                              "reason": reason}]}

    def receipt(self, result):
        return make_coverage_receipt(scorer="S", task_scope_digest="T",
                                     coverage_key="K", result=result)

    def view(self, reason="First explanation"):
        return {"question": "Compare A and B", "requirements": [{"id": "R1"}],
                "constraints": [], "prior_coverage": self.result(reason)["coverage"],
                "retained_evidence": [{"source_id": "E1", "text": "A outcomes"}],
                "new_evidence": [{"source_id": "E2", "text": "B outcomes"}],
                "source_headers": {}}

    def test_reason_only_changes_have_one_receipt(self):
        a, b = self.result(), self.result("Different wording")
        self.assertNotEqual(digest(a), digest(b))
        self.assertEqual(self.receipt(a), self.receipt(b))
        self.assertEqual(a["coverage"][0]["reason"], "First explanation")

    def test_status_and_basis_changes_remain_distinct(self):
        baseline = self.receipt(self.result())
        self.assertNotEqual(baseline, self.receipt(self.result(status="direct")))
        self.assertNotEqual(baseline, self.receipt(self.result(refs=["E2"])))

    def test_duplicate_or_extra_fields_are_rejected(self):
        value = self.result(refs=["E1", "E1"])
        with self.assertRaises(ValueError):
            self.receipt(value)
        value = self.result()
        value["coverage"][0]["contradiction"] = True
        with self.assertRaises(ValueError):
            self.receipt(value)

    def test_rehashed_noncanonical_explanation_is_rejected(self):
        receipt = self.receipt(self.result())
        receipt["result"]["coverage"][0]["reason"] = "Changed explanation"
        receipt["receipt_id"] = digest({k: v for k, v in receipt.items() if k != "receipt_id"})
        with self.assertRaisesRegex(ValueError, "not canonical"):
            validate_coverage_receipt(receipt, scorer="S", task_scope_digest="T", coverage_key="K")

    def test_next_request_key_ignores_prior_explanation_wording(self):
        self.assertEqual(coverage_delta_key("S", self.view()),
                         coverage_delta_key("S", self.view("Other wording")))

    def test_no_change_inherits_exact_receipt(self):
        view = self.view()
        result = {"coverage": [{"id": "R1", "any_requested_option_or_member_present": False,
                                "requested_outcome_support": False, "population_applicable": False,
                                "combined_requirement_complete": False,
                                "contradiction": False, "basis_new_evidence_ids": [],
                                "reason": "No help"}]}
        after, contradictions = merge_coverage_delta(view, result)
        self.assertEqual(self.receipt(self.result()), self.receipt(after))
        self.assertEqual(contradictions, [])

    def test_true_delta_flags_cannot_be_changed_by_normalization(self):
        view = self.view()
        delta = {"coverage": [{"id": "R1", "any_requested_option_or_member_present": True,
                               "requested_outcome_support": True, "population_applicable": True,
                               "combined_requirement_complete": True,
                               "contradiction": False, "basis_new_evidence_ids": ["E2"],
                               "reason": "Now complete"}]}
        after, _ = merge_coverage_delta(view, delta)
        self.assertEqual(after["coverage"][0]["status"], "direct")
        self.assertEqual(after["coverage"][0]["evidence_ids"], ["E1", "E2"])
        changed = copy.deepcopy(delta)
        changed["coverage"][0]["reason"] = "Different explanation"
        self.assertEqual(merge_coverage_delta(view, delta), merge_coverage_delta(view, changed))
        changed["coverage"][0]["combined_requirement_complete"] = False
        self.assertNotEqual(merge_coverage_delta(view, delta), merge_coverage_delta(view, changed))


if __name__ == "__main__":
    unittest.main()
