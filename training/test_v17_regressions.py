"""V17 evidence-ledger and no-State Browse regression tests."""
import copy
import unittest

from core import digest
from test_joint import (CONFIG, attach_execution_manifest, attach_stage_score,
                        compile_fixture, fixture)
from judge import evidence_gain_event
from gain_contract import (canonical_action, canonical_evidence,
                           make_coverage_receipt, make_evidence_transition,
                           make_gain_disposition)


def first_browse(batch):
    roll = batch["rollouts"][0]
    record = next(row for row in roll["records"]
                  if row["stage"] == "decision")
    state = next(row for row in roll["records"] if row["stage"] == "state")
    return roll, record, state


def replace_execution(batch, record, *, status, evidence, capture_id):
    old = batch["private_tool_executions"][record["tool_execution_id"]]
    execution = copy.deepcopy(old)
    execution.pop("tool_execution_id")
    execution["record_id"] = record["id"]
    execution["status"] = status
    execution["capture"]["capture_id"] = capture_id
    payload = {"tool": "browse", "evidence": canonical_evidence(evidence)}
    execution["response"] = {"payload": payload,
                             "payload_digest": digest(payload)}
    execution["tool_execution_id"] = digest(execution)
    record["tool_execution_id"] = execution["tool_execution_id"]
    for event in record["task_events"]:
        event["tool_execution_id"] = execution["tool_execution_id"]
        if event["kind"] == "tool_cost":
            event["proof"]["tool_execution_id"] = execution["tool_execution_id"]
    batch["private_tool_executions"][execution["tool_execution_id"]] = execution
    return execution


def configure_no_state(batch, *, failed=False):
    roll, record, state = first_browse(batch)
    record["task_events"] = [event for event in record["task_events"]
                             if event["kind"] != "evidence_gain"]
    execution = replace_execution(
        batch, record, status="failed" if failed else "succeeded",
        evidence=[], capture_id="no-state-failed" if failed else "no-state-zero")
    before = {"step": 1, "digest": digest([])}
    after = {"step": 2, "digest": digest([])}
    record["evidence_snapshot"] = before
    transition = make_evidence_transition(
        question_id=roll["question_id"], rollout_id=roll["rollout_id"],
        browse_record_id=record["id"], tool_execution=execution,
        policy=roll["policy"], token_digest=record["token_digest"],
        task_scope_digest=roll["task_scope_digest"], before_snapshot=before,
        after_snapshot=after, before_evidence=[], after_evidence=[])
    batch["private_evidence_transitions"][execution["tool_execution_id"]] = transition
    if failed:
        disposition = make_gain_disposition(
            status="failed_no_evidence", artifact_key=None,
            question_id=roll["question_id"], rollout_id=roll["rollout_id"],
            browse_record_id=record["id"], after_record_id=None,
            tool_execution_id=execution["tool_execution_id"], policy=roll["policy"],
            token_digest=record["token_digest"],
            task_scope_digest=roll["task_scope_digest"])
    else:
        disposition = make_gain_disposition(
            status="observed_zero_no_change", artifact_key=None,
            question_id=roll["question_id"], rollout_id=roll["rollout_id"],
            browse_record_id=record["id"], after_record_id=None,
            tool_execution_id=execution["tool_execution_id"], policy=roll["policy"],
            token_digest=record["token_digest"],
            task_scope_digest=roll["task_scope_digest"])
    batch["private_gain_dispositions"][execution["tool_execution_id"]] = disposition
    roll["records"].remove(state)
    attach_execution_manifest(batch, roll)
    return roll, record


def add_second_browse(batch, *, reset):
    roll, first, first_state = first_browse(batch)
    second = copy.deepcopy(first)
    second["id"] = "decision2"
    second["task_events"] = [event for event in second["task_events"]
                             if event["kind"] == "tool_cost"]
    second["task_events"][0]["id"] = "cost-second"
    second_state = copy.deepcopy(first_state)
    second_state["id"] = "state2"
    existing = canonical_evidence([
        {"source_id": "E-" + roll["rollout_id"],
         "text": "Observed result for " + roll["rollout_id"]}])
    before_evidence = [] if reset else existing
    added = {"source_id": "E-second", "text": "Second Browse result"}
    after_evidence = canonical_evidence(before_evidence + [added])
    second["evidence_snapshot"] = {"step": 2, "digest": digest(before_evidence)}
    second_state["evidence_snapshot"] = {"step": 3,
                                          "digest": digest(after_evidence)}
    execution = replace_execution(batch, second, status="succeeded",
                                  evidence=[added], capture_id="second")
    transition = make_evidence_transition(
        question_id=roll["question_id"], rollout_id=roll["rollout_id"],
        browse_record_id=second["id"], tool_execution=execution,
        policy=roll["policy"], token_digest=second["token_digest"],
        task_scope_digest=roll["task_scope_digest"],
        before_snapshot=second["evidence_snapshot"],
        after_snapshot=second_state["evidence_snapshot"],
        before_evidence=before_evidence, after_evidence=after_evidence)
    batch["private_evidence_transitions"][execution["tool_execution_id"]] = transition
    batch["private_gain_dispositions"][execution["tool_execution_id"]] = (
        make_gain_disposition(
            status="pending", artifact_key=None,
            question_id=roll["question_id"], rollout_id=roll["rollout_id"],
            browse_record_id=second["id"], after_record_id=second_state["id"],
            tool_execution_id=execution["tool_execution_id"], policy=roll["policy"],
            token_digest=second["token_digest"],
            task_scope_digest=roll["task_scope_digest"]))
    roll["records"][3:3] = [second, second_state]
    attach_stage_score(batch, roll, second, "browse_source_focus", 1.0)
    attach_stage_score(batch, roll, second_state, "state", 1.0)
    attach_execution_manifest(batch, roll)
    return roll, second


class V17Regressions(unittest.TestCase):
    def test_cross_browse_ledger_reset_is_rejected(self):
        batch = fixture()
        add_second_browse(batch, reset=True)
        with self.assertRaisesRegex(ValueError, "ledger is not continuous"):
            compile_fixture(batch)

    def test_continuous_cross_browse_ledger_is_accepted(self):
        batch = fixture()
        add_second_browse(batch, reset=False)
        rows = compile_fixture(batch)
        self.assertTrue(any(row["record"]["id"] == "decision2" for row in rows))

    def test_success_without_new_evidence_or_state_is_zero_gain(self):
        batch = fixture()
        roll, record = configure_no_state(batch, failed=False)
        rows = compile_fixture(batch)
        tool = next(row for row in rows if row["rollout_id"] == roll["rollout_id"]
                    and row["record"]["id"] == record["id"]
                    and row["channel"] == "tool")
        self.assertAlmostEqual(tool["local_score"], -CONFIG["task"]["tool_cost"])

    def test_failed_browse_without_state_is_accounted(self):
        batch = fixture()
        roll, record = configure_no_state(batch, failed=True)
        rows = compile_fixture(batch)
        tool = next(row for row in rows if row["rollout_id"] == roll["rollout_id"]
                    and row["record"]["id"] == record["id"]
                    and row["channel"] == "tool")
        self.assertAlmostEqual(tool["local_score"], -CONFIG["task"]["tool_cost"])

    def test_transition_deletion_is_rejected(self):
        batch = fixture()
        _, record, _ = first_browse(batch)
        transition = copy.deepcopy(
            batch["private_evidence_transitions"][record["tool_execution_id"]])
        transition["after_evidence"] = []
        transition["after_snapshot"]["digest"] = digest([])
        core = copy.deepcopy(transition)
        core.pop("transition_id")
        transition["transition_id"] = digest(core)
        batch["private_evidence_transitions"][record["tool_execution_id"]] = transition
        with self.assertRaisesRegex(ValueError, "response absent|ledger"):
            compile_fixture(batch)

    def test_gain_artifact_must_match_transition(self):
        batch = fixture()
        _, record, _ = first_browse(batch)
        transition = copy.deepcopy(
            batch["private_evidence_transitions"][record["tool_execution_id"]])
        transition["after_snapshot"]["step"] += 1
        core = copy.deepcopy(transition)
        core.pop("transition_id")
        transition["transition_id"] = digest(core)
        batch["private_evidence_transitions"][record["tool_execution_id"]] = transition
        with self.assertRaisesRegex(ValueError, "steps must be adjacent"):
            compile_fixture(batch)


if __name__ == "__main__":
    unittest.main()
