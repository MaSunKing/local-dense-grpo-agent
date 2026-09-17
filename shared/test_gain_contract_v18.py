"""V18 coverage identity normalization and metadata-lineage tests."""
import unittest

from gain_contract import (
    canonical_source_headers, coverage_basis_digest, coverage_key, digest,
    make_evidence_transition,
)
from test_gain_contract_v17 import execution


def view(headers):
    return {
        "question": "Q?",
        "requirements": [{"id": "R1", "description": "Outcome", "weight": 1}],
        "constraints": [],
        "evidence": [{"source_id": "E1", "text": "Result"}],
        "source_headers": headers,
    }


class CoverageIdentityTests(unittest.TestCase):
    def test_empty_header_forms_have_one_identity(self):
        self.assertEqual(canonical_source_headers({}, {"E1"}), {})
        self.assertEqual(canonical_source_headers({"E1": {}}, {"E1"}), {})
        self.assertEqual(coverage_key("scorer", view({})),
                         coverage_key("scorer", view({"E1": {}})))
        self.assertEqual(coverage_basis_digest(view({})),
                         coverage_basis_digest(view({"E1": {}})))

    def test_no_change_canonicalizes_empty_header_representation(self):
        tool = execution(status="succeeded", evidence=[
            {"source_id": "E1", "text": "Result"}])
        snap1 = {"step": 1, "digest": digest(view({})["evidence"])}
        snap2 = {"step": 2, "digest": digest(view({})["evidence"])}
        transition = make_evidence_transition(
            question_id="q", rollout_id="r", browse_record_id="browse",
            tool_execution=tool, policy="p", token_digest="t",
            task_scope_digest="s", before_snapshot=snap1,
            after_snapshot=snap2, before_evidence=view({})["evidence"],
            after_evidence=view({})["evidence"], before_source_headers={},
            after_source_headers={"E1": {}})
        self.assertEqual(transition["before_source_headers"], {})
        self.assertEqual(transition["after_source_headers"], {})
        self.assertEqual(transition["change"], "no_change")

    def test_informative_metadata_cannot_appear_on_no_change(self):
        tool = execution(status="succeeded", evidence=[
            {"source_id": "E1", "text": "Result"}])
        snap1 = {"step": 1, "digest": digest(view({})["evidence"])}
        snap2 = {"step": 2, "digest": digest(view({})["evidence"])}
        with self.assertRaisesRegex(ValueError, "metadata update"):
            make_evidence_transition(
                question_id="q", rollout_id="r", browse_record_id="browse",
                tool_execution=tool, policy="p", token_digest="t",
                task_scope_digest="s", before_snapshot=snap1,
                after_snapshot=snap2, before_evidence=view({})["evidence"],
                after_evidence=view({})["evidence"], before_source_headers={},
                after_source_headers={"E1": {"title": "Late title"}})


if __name__ == "__main__":
    unittest.main()
