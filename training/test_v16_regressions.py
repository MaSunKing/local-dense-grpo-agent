import copy
import unittest

from core import digest
from judge import stop_boundary_event
from test_joint import CONFIG, compile_fixture, fixture

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from gain_contract import (  # noqa: E402
    make_coverage_receipt, make_gain_disposition, record_target_digest,
    digest as shared_digest,
)


def browse_record(batch, rollout_index=0):
    roll = batch["rollouts"][rollout_index]
    record = next(row for row in roll["records"] if row["stage"] == "decision")
    return roll, record


def replace_disposition(batch, roll, record, *, status, artifact_key):
    batch["private_gain_dispositions"][record["tool_execution_id"]] = make_gain_disposition(
        status=status, artifact_key=artifact_key, question_id=roll["question_id"],
        rollout_id=roll["rollout_id"], browse_record_id=record["id"],
        after_record_id=next(row for row in roll["records"] if row["stage"] == "state")["id"],
        tool_execution_id=record["tool_execution_id"], policy=roll["policy"],
        token_digest=record["token_digest"], task_scope_digest=roll["task_scope_digest"])


class V16Regressions(unittest.TestCase):
    def test_missing_gain_disposition_is_not_silent_zero(self):
        batch = fixture(); _, record = browse_record(batch)
        del batch["private_gain_dispositions"][record["tool_execution_id"]]
        with self.assertRaisesRegex(ValueError, "gain disposition"):
            compile_fixture(batch)

    def test_pending_gain_masks_tool_return_instead_of_becoming_cost_only(self):
        batch = fixture(); roll, record = browse_record(batch)
        record["task_events"] = [event for event in record["task_events"]
                                 if event["kind"] != "evidence_gain"]
        replace_disposition(batch, roll, record, status="pending", artifact_key=None)
        rows = compile_fixture(batch)
        tool = next(row for row in rows if row["rollout_id"] == roll["rollout_id"]
                    and row["channel"] == "tool")
        self.assertIsNone(tool["local_score"])
        self.assertIsNone(tool["local_advantage"])

    def test_observed_zero_is_distinct_from_pending(self):
        batch = fixture(); roll, _ = browse_record(batch, 3)
        rows = compile_fixture(batch)
        tool = next(row for row in rows if row["rollout_id"] == roll["rollout_id"]
                    and row["channel"] == "tool")
        self.assertAlmostEqual(tool["local_score"], -CONFIG["task"]["tool_cost"])

    def test_unobservable_assessment_masks_tool_return(self):
        batch = fixture(); roll, record = browse_record(batch)
        event = next(event for event in record["task_events"]
                     if event["kind"] == "evidence_gain")
        old_key = event["proof"]["artifact_key"]
        artifact = copy.deepcopy(batch["private_evidence_scores"].pop(old_key))
        for row in artifact["after_result"]["coverage"]:
            row.update(status="unobservable", evidence_ids=[], reason="cannot determine")
        artifact["after_score"] = None
        artifact["transitions"] = [
            dict(id=row["id"], before=row["status"], after="unobservable",
                 delta=None, new_evidence_ids=[])
            for row in artifact["before_result"]["coverage"]
        ]
        artifact["disposition"] = "unobservable"; artifact["delta"] = None
        new_key = digest(artifact)
        batch["private_evidence_scores"][new_key] = artifact
        batch["private_coverage_receipts"][artifact["after_key"]] = make_coverage_receipt(
            scorer=batch["identity"]["scorer"],
            task_scope_digest=roll["task_scope_digest"],
            coverage_key=artifact["after_key"], result=artifact["after_result"])
        record["task_events"] = [row for row in record["task_events"]
                                 if row["kind"] != "evidence_gain"]
        replace_disposition(batch, roll, record, status="unobservable",
                            artifact_key=new_key)
        rows = compile_fixture(batch)
        tool = next(row for row in rows if row["rollout_id"] == roll["rollout_id"]
                    and row["channel"] == "tool")
        self.assertIsNone(tool["local_score"])

    def test_conflicting_result_for_same_coverage_key_fails_closed(self):
        batch = fixture(); _, record = browse_record(batch, 2)
        event = next(row for row in record["task_events"] if row["kind"] == "evidence_gain")
        artifact = copy.deepcopy(batch["private_evidence_scores"][event["proof"]["artifact_key"]])
        row = artifact["after_result"]["coverage"][0]
        row["status"] = "direct" if row["status"] == "unknown" else "unknown"
        row["evidence_ids"] = ([artifact["after_view"]["evidence"][0]["source_id"]]
                               if row["status"] == "direct" else [])
        new_key = digest(artifact)
        batch["private_evidence_scores"][new_key] = artifact
        batch["private_gain_dispositions"][record["tool_execution_id"]]["artifact_key"] = new_key
        disposition = batch["private_gain_dispositions"][record["tool_execution_id"]]
        core = copy.deepcopy(disposition); core.pop("disposition_id")
        disposition["disposition_id"] = shared_digest(core)
        event["proof"]["artifact_key"] = new_key
        with self.assertRaisesRegex(ValueError, "canonical receipt"):
            compile_fixture(batch)

    def test_reason_only_difference_preserves_compiled_rewards(self):
        batch = fixture()
        baseline = compile_fixture(copy.deepcopy(batch))
        _, record = browse_record(batch, 2)
        event = next(row for row in record["task_events"] if row["kind"] == "evidence_gain")
        artifact = copy.deepcopy(batch["private_evidence_scores"][event["proof"]["artifact_key"]])
        artifact["after_result"]["coverage"][0]["reason"] = "same decision, different explanation"
        new_key = digest(artifact)
        batch["private_evidence_scores"][new_key] = artifact
        disposition = batch["private_gain_dispositions"][record["tool_execution_id"]]
        disposition["artifact_key"] = new_key
        core = copy.deepcopy(disposition); core.pop("disposition_id")
        disposition["disposition_id"] = shared_digest(core)
        event["proof"]["artifact_key"] = new_key
        actual = compile_fixture(batch)
        scoring_fields = ("rollout_id", "record_id", "channel", "local_score", "advantage")
        self.assertEqual([{key: row.get(key) for key in scoring_fields} for row in actual],
                         [{key: row.get(key) for key in scoring_fields} for row in baseline])

    def test_forced_stop_is_not_applicable_and_does_not_poison_prior_tool(self):
        batch = fixture(); roll = batch["rollouts"][0]
        stop = next(row for row in roll["records"] if row["stage"] == "stop")
        old_event = stop["task_events"][0]
        old_termination = batch["private_terminations"].pop(
            old_event["proof"]["termination_event_id"])
        termination = copy.deepcopy(old_termination)
        termination["type"] = "quota_exhausted"
        termination["stop_context"]["voluntary"] = False
        termination["stop_context"]["remaining_budget"] = 0
        core = copy.deepcopy(termination); core.pop("event_id")
        termination["event_id"] = shared_digest(core)
        batch["private_terminations"][termination["event_id"]] = termination
        target = record_target_digest(question_id=roll["question_id"],
            rollout_id=roll["rollout_id"], record=stop, channel="stop",
            task_scope_digest=roll["task_scope_digest"])
        expected = dict(question_id=roll["question_id"], rollout_id=roll["rollout_id"],
            record_id=stop["id"], token_digest=stop["token_digest"], policy=roll["policy"],
            task_scope_digest=roll["task_scope_digest"], target_digest=target)
        stop["task_events"] = [stop_boundary_event(None, termination,
            event_id=old_event["id"], config=CONFIG, scorer=batch["identity"]["scorer"],
            expected=expected)]
        rows = compile_fixture(batch)
        stop_row = next(row for row in rows if row["rollout_id"] == roll["rollout_id"]
                        and row["channel"] == "stop")
        tool_row = next(row for row in rows if row["rollout_id"] == roll["rollout_id"]
                        and row["channel"] == "tool")
        self.assertIsNone(stop_row["local_score"])
        self.assertAlmostEqual(tool_row["local_score"], 1.0-CONFIG["task"]["tool_cost"])


if __name__ == "__main__":
    unittest.main()
