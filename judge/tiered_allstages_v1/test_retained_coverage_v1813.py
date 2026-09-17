import copy
import unittest

from engine import prepare
from evidence_gain import validate_retained_coverage
from fixtures import case


ATOMS = (
    "any_requested_option_or_member_present",
    "requested_outcome_support",
    "population_applicable",
    "complete_requirement_support",
)


def row(rid, status, evidence_ids=None):
    values = {
        "unknown": (False, False, False, False),
        "partial": (True, True, True, False),
        "direct": (True, True, True, True),
    }[status]
    return {"id": rid, **dict(zip(ATOMS, values)),
            "evidence_ids": list(evidence_ids or []), "reason": "test basis"}


class RetainedCoverageTests(unittest.TestCase):
    def pair(self):
        before = prepare(case("evidence")[0])["views"]["coverage"]
        after = copy.deepcopy(before)
        return before, after

    def result(self, q1, q2):
        return {"coverage": [
            row("Q1", q1, ["POOL"] if q1 != "unknown" else []),
            row("Q2", q2, ["SAFE"] if q2 != "unknown" else []),
        ]}

    def test_append_only_partial_to_unknown_is_contract_error(self):
        before, after = self.pair()
        with self.assertRaisesRegex(ValueError,
                                    "retained evidence was downgraded"):
            validate_retained_coverage(
                before, self.result("partial", "unknown"),
                after, self.result("unknown", "unknown"))

    def test_append_only_direct_to_partial_is_contract_error(self):
        before, after = self.pair()
        with self.assertRaisesRegex(ValueError,
                                    "retained evidence was downgraded"):
            validate_retained_coverage(
                before, self.result("direct", "unknown"),
                after, self.result("partial", "unknown"))

    def test_equal_retained_coverage_is_accepted(self):
        before, after = self.pair()
        checked = validate_retained_coverage(
            before, self.result("partial", "unknown"),
            after, self.result("partial", "unknown"))
        self.assertEqual(checked["after"]["coverage"]["Q1"]["status"],
                         "partial")

    def test_helper_does_not_copy_old_result(self):
        before, after = self.pair()
        invalid = self.result("unknown", "unknown")
        with self.assertRaises(ValueError):
            validate_retained_coverage(
                before, self.result("partial", "unknown"), after, invalid)
        self.assertEqual(invalid["coverage"][0]["evidence_ids"], [])


if __name__ == "__main__":
    unittest.main()
