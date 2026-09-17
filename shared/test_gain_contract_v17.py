import copy
import json
import unittest

from gain_contract import (canonical_action, canonical_evidence, digest,
                           make_evidence_transition,
                           validate_evidence_transition)


def execution(*, status="succeeded", evidence=None):
    raw = json.dumps({"tool": "browse", "arguments": {
        "candidate_id": "candidate", "focus": "outcome"}})
    action = canonical_action(raw)
    rows = canonical_evidence(evidence or [])
    payload = {"tool": "browse", "evidence": rows}
    core = {"version": "trusted_tool_execution_v1", "question_id": "q",
            "rollout_id": "r", "record_id": "browse", "policy": "p",
            "token_digest": "t", "task_scope_digest": "s",
            "capture": {"capture_id": "c", "capture_sha256": "a" * 64,
                        "parser_version": "json_tool_action_v1",
                        "raw_completion": raw, "action": action,
                        "action_digest": digest(action)},
            "response": {"payload": payload, "payload_digest": digest(payload)},
            "status": status}
    return {"tool_execution_id": digest(core), **core}


def transition(tool, before, after):
    return make_evidence_transition(
        question_id="q", rollout_id="r", browse_record_id="browse",
        tool_execution=tool, policy="p", token_digest="t",
        task_scope_digest="s",
        before_snapshot={"step": 4, "digest": digest(canonical_evidence(before))},
        after_snapshot={"step": 5, "digest": digest(canonical_evidence(after))},
        before_evidence=before, after_evidence=after)


class EvidenceTransitionTests(unittest.TestCase):
    def test_append_and_no_change_are_distinct(self):
        row = {"source_id": "E1", "text": "result"}
        append = transition(execution(evidence=[row]), [], [row])
        self.assertEqual(validate_evidence_transition(
            append, tool_execution=execution(evidence=[row]))["change"], "append")
        unchanged = transition(execution(evidence=[]), [row], [row])
        self.assertEqual(validate_evidence_transition(
            unchanged, tool_execution=execution(evidence=[]))["change"], "no_change")

    def test_deletion_or_mutation_fails_closed(self):
        row = {"source_id": "E1", "text": "result"}
        with self.assertRaisesRegex(ValueError, "deleted or modified"):
            transition(execution(evidence=[]), [row], [])
        changed = {"source_id": "E1", "text": "changed"}
        with self.assertRaisesRegex(ValueError, "deleted or modified"):
            transition(execution(evidence=[]), [row], [changed])

    def test_failed_read_preserves_ledger(self):
        row = {"source_id": "E1", "text": "result"}
        failed = transition(execution(status="failed", evidence=[]), [row], [row])
        self.assertEqual(failed["change"], "failed_no_change")
        bad_tool = execution(status="failed", evidence=[row])
        with self.assertRaisesRegex(ValueError, "failed Browse"):
            transition(bad_tool, [], [row])


if __name__ == "__main__":
    unittest.main()
