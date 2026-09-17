"""V18.14 deterministic incremental coverage regressions."""
import copy
import unittest

from gain_contract import (
    canonical_coverage_delta_view, merge_coverage_delta,
    validate_coverage_delta_result,
)


def view(status="partial"):
    refs = ["E1"] if status != "unknown" else []
    return canonical_coverage_delta_view({
        "question": "Compare A with B for outcome O.",
        "requirements": [{"id": "R1", "description": "Compare O", "weight": 1}],
        "constraints": [],
        "prior_coverage": [{"id": "R1", "status": status,
                            "evidence_ids": refs, "reason": "trusted prior"}],
        "retained_evidence": ([{"source_id": "E1", "text": "A reports O."}]
                              if refs else []),
        "new_evidence": [{"source_id": "E2", "text": "new evidence"}],
        "source_headers": {},
    })


def delta(*, material=False, complete=False, contradiction=False, refs=None):
    if refs is None:
        refs = ["E2"] if any((material, complete, contradiction)) else []
    return {"coverage": [{
        "id": "R1", "any_requested_option_or_member_present": material,
        "requested_outcome_support": material, "population_applicable": material,
        "combined_requirement_complete": complete,
        "contradiction": contradiction,
        "basis_new_evidence_ids": refs,
        "reason": "bounded delta judgment",
    }]}


class IncrementalCoverageTests(unittest.TestCase):
    def test_irrelevant_new_evidence_cannot_downgrade_partial(self):
        merged, conflicts = merge_coverage_delta(view("partial"), delta())
        self.assertEqual(merged["coverage"][0]["status"], "partial")
        self.assertEqual(merged["coverage"][0]["evidence_ids"], ["E1"])
        self.assertEqual(conflicts, [])

    def test_new_material_support_upgrades_unknown_to_partial(self):
        merged, _ = merge_coverage_delta(
            view("unknown"), delta(material=True))
        self.assertEqual(merged["coverage"][0]["status"], "partial")
        self.assertEqual(merged["coverage"][0]["evidence_ids"], ["E2"])

    def test_completion_upgrades_partial_to_direct(self):
        merged, _ = merge_coverage_delta(
            view("partial"), delta(material=True, complete=True))
        self.assertEqual(merged["coverage"][0]["status"], "direct")
        self.assertEqual(merged["coverage"][0]["evidence_ids"], ["E1", "E2"])

    def test_contradiction_is_recorded_without_erasing_support(self):
        merged, conflicts = merge_coverage_delta(
            view("partial"), delta(contradiction=True))
        self.assertEqual(merged["coverage"][0]["status"], "partial")
        self.assertEqual(merged["coverage"][0]["evidence_ids"], ["E1"])
        self.assertEqual(conflicts[0]["evidence_ids"], ["E2"])

    def test_retained_or_foreign_basis_is_rejected(self):
        for bad in (["E1"], ["FOREIGN"]):
            with self.assertRaises(ValueError):
                validate_coverage_delta_result(
                    view("partial"), delta(material=True, refs=bad))

    def test_tampered_prior_binding_is_rejected(self):
        broken = copy.deepcopy(view("partial"))
        broken["prior_coverage"][0]["evidence_ids"] = ["E2"]
        with self.assertRaises(ValueError):
            canonical_coverage_delta_view(broken)


if __name__ == "__main__":
    unittest.main()
