"""V18 step-independent coverage receipt inheritance regressions."""
import copy
import unittest

from core import digest
from judge import evidence_gain_event
from test_joint import (CONFIG, attach_execution_manifest, attach_stage_score,
                        compile_fixture, fixture)
from test_v17_regressions import first_browse, replace_execution
from gain_contract import (
    canonical_coverage_view, canonical_evidence, coverage_key,
    make_coverage_receipt, make_evidence_transition, make_gain_disposition,
)


def _coverage_rows(requirements, evidence_id, *, direct):
    return [
        {"id": row["id"], "status": "direct" if direct else "unknown",
         "evidence_ids": [evidence_id] if direct else [],
         "reason": "covered" if direct else "not covered"}
        for row in requirements
    ]


def add_no_change_then_append(batch, *, conflicting_before):
    roll, first, first_state = first_browse(batch)
    evidence1 = canonical_evidence([{
        "source_id": "E-" + roll["rollout_id"],
        "text": "Observed result for " + roll["rollout_id"],
    }])
    evidence2 = {"source_id": "E-second", "text": "Second result"}

    # A duplicate/empty-value read advances the trusted collector step but
    # neither changes the semantic evidence ledger nor emits a State record.
    repeat = copy.deepcopy(first)
    repeat["id"] = "decision-repeat"
    repeat["task_events"] = [copy.deepcopy(next(
        event for event in first["task_events"] if event["kind"] == "tool_cost"))]
    repeat["task_events"][0]["id"] = "cost-repeat"
    repeat["evidence_snapshot"] = {"step": 2, "digest": digest(evidence1)}
    repeat_execution = replace_execution(
        batch, repeat, status="succeeded", evidence=evidence1,
        capture_id="repeat")
    repeat_transition = make_evidence_transition(
        question_id=roll["question_id"], rollout_id=roll["rollout_id"],
        browse_record_id=repeat["id"], tool_execution=repeat_execution,
        policy=roll["policy"], token_digest=repeat["token_digest"],
        task_scope_digest=roll["task_scope_digest"],
        before_snapshot=repeat["evidence_snapshot"],
        after_snapshot={"step": 3, "digest": digest(evidence1)},
        before_evidence=evidence1, after_evidence=evidence1,
        before_source_headers={}, after_source_headers={"E-" + roll["rollout_id"]: {}})
    batch["private_evidence_transitions"][repeat_execution["tool_execution_id"]] = repeat_transition
    batch["private_gain_dispositions"][repeat_execution["tool_execution_id"]] = make_gain_disposition(
        status="observed_zero_no_change", artifact_key=None,
        question_id=roll["question_id"], rollout_id=roll["rollout_id"],
        browse_record_id=repeat["id"], after_record_id=None,
        tool_execution_id=repeat_execution["tool_execution_id"],
        policy=roll["policy"], token_digest=repeat["token_digest"],
        task_scope_digest=roll["task_scope_digest"])
    attach_stage_score(batch, roll, repeat, "browse_source_focus", 1.0)

    third = copy.deepcopy(first)
    third["id"] = "decision-third"
    third["task_events"] = [copy.deepcopy(next(
        event for event in first["task_events"] if event["kind"] == "tool_cost"))]
    third["task_events"][0]["id"] = "cost-third"
    third["evidence_snapshot"] = {"step": 3, "digest": digest(evidence1)}
    third_state = copy.deepcopy(first_state)
    third_state["id"] = "state-third"
    after_evidence = canonical_evidence(evidence1 + [evidence2])
    third_state["evidence_snapshot"] = {"step": 4,
                                         "digest": digest(after_evidence)}
    third_execution = replace_execution(
        batch, third, status="succeeded", evidence=[evidence2],
        capture_id="third")
    third_transition = make_evidence_transition(
        question_id=roll["question_id"], rollout_id=roll["rollout_id"],
        browse_record_id=third["id"], tool_execution=third_execution,
        policy=roll["policy"], token_digest=third["token_digest"],
        task_scope_digest=roll["task_scope_digest"],
        before_snapshot=third["evidence_snapshot"],
        after_snapshot=third_state["evidence_snapshot"],
        before_evidence=evidence1, after_evidence=after_evidence,
        before_source_headers={"E-" + roll["rollout_id"]: {}},
        after_source_headers={})
    batch["private_evidence_transitions"][third_execution["tool_execution_id"]] = third_transition

    requirements = batch["task_scopes"][roll["question_id"]]["requirements"]
    base = {"question": "Question?", "requirements": requirements,
            "constraints": []}
    before_view = canonical_coverage_view(base | {
        "evidence": evidence1,
        "source_headers": {"E-" + roll["rollout_id"]: {}}})
    after_view = canonical_coverage_view(base | {
        "evidence": after_evidence, "source_headers": {}})
    evidence1_id = evidence1[0]["source_id"]
    if conflicting_before:
        before_result = {"coverage": _coverage_rows(
            requirements, evidence1_id, direct=False)}
        after_result = {"coverage": _coverage_rows(
            requirements, "E-second", direct=True)}
    else:
        first_gain = next(event for event in first["task_events"]
                          if event["kind"] == "evidence_gain")
        first_artifact = batch["private_evidence_scores"][
            first_gain["proof"]["artifact_key"]]
        # Inheritance is exact, including the audited reason text.
        before_result = copy.deepcopy(first_artifact["after_result"])
        after_result = copy.deepcopy(before_result)
    scorer = batch["identity"]["scorer"]
    before_key = coverage_key(scorer, before_view)
    after_key = coverage_key(scorer, after_view)
    before_score = 0.0 if conflicting_before else 1.0
    after_score = 1.0
    transitions = []
    for row in requirements:
        transitions.append({
            "id": row["id"],
            "before": "unknown" if conflicting_before else "direct",
            "after": "direct",
            "delta": 1.0 if conflicting_before else 0.0,
            "new_evidence_ids": ["E-second"] if conflicting_before else [],
        })
    status = "observed_positive" if conflicting_before else "observed_zero"
    artifact = {
        "version": "browse_evidence_gain_v4", "scorer": scorer,
        "reward_config": digest(CONFIG), "task_modes_digest": digest({}),
        "task_scope_digest": roll["task_scope_digest"],
        "tool_execution": third_execution,
        "binding": {
            "question_id": roll["question_id"], "rollout_id": roll["rollout_id"],
            "browse_record_id": third["id"], "after_record_id": third_state["id"],
            "policy": roll["policy"], "before_snapshot": third["evidence_snapshot"],
            "after_snapshot": third_state["evidence_snapshot"],
            "selected_candidate_id": "candidate", "new_evidence_ids": ["E-second"],
            "tool_execution_id": third_execution["tool_execution_id"],
            "task_scope_digest": roll["task_scope_digest"],
        },
        "before_view": before_view, "after_view": after_view,
        "before_result": before_result, "after_result": after_result,
        "before_key": before_key, "after_key": after_key,
        "before_score": before_score, "after_score": after_score,
        "transitions": transitions, "disposition": status,
        "delta": 1.0 if conflicting_before else 0.0,
    }
    artifact_key = digest(artifact)
    # A key is a single canonical receipt slot.  A contradictory re-score of
    # the same evidence necessarily replaces the earlier receipt and makes
    # the full batch fail closed.
    for key, result in ((before_key, before_result), (after_key, after_result)):
        batch["private_coverage_receipts"][key] = make_coverage_receipt(
            scorer=scorer, task_scope_digest=roll["task_scope_digest"],
            coverage_key=key, result=result)
    batch["private_evidence_scores"][artifact_key] = artifact
    batch["private_evidence_transitions"][third_execution["tool_execution_id"]] = third_transition
    batch["private_gain_dispositions"][third_execution["tool_execution_id"]] = make_gain_disposition(
        status=status, artifact_key=artifact_key,
        question_id=roll["question_id"], rollout_id=roll["rollout_id"],
        browse_record_id=third["id"], after_record_id=third_state["id"],
        tool_execution_id=third_execution["tool_execution_id"],
        policy=roll["policy"], token_digest=third["token_digest"],
        task_scope_digest=roll["task_scope_digest"])
    if conflicting_before:
        third["task_events"].insert(0, evidence_gain_event(
            artifact, event_id="gain-third", config=CONFIG, scorer=scorer,
            coverage_receipt_lookup=lambda key:
            batch["private_coverage_receipts"].get(key),
            evidence_transition=third_transition))
    attach_stage_score(batch, roll, third, "browse_source_focus", 1.0)
    attach_stage_score(batch, roll, third_state, "state", 1.0)
    insertion = roll["records"].index(first_state) + 1
    roll["records"][insertion:insertion] = [repeat, third, third_state]
    attach_execution_manifest(batch, roll)
    return roll


class V18Regressions(unittest.TestCase):
    def test_empty_metadata_cannot_create_second_coverage_identity(self):
        batch = fixture()
        add_no_change_then_append(batch, conflicting_before=True)
        with self.assertRaisesRegex(ValueError,
                                    "canonical receipt|identical evidence basis"):
            compile_fixture(batch)

    def test_no_change_inherits_consistent_coverage(self):
        batch = fixture()
        roll = add_no_change_then_append(batch, conflicting_before=False)
        rows = compile_fixture(batch)
        gain_events = [event for record in roll["records"]
                       for event in record.get("task_events", [])
                       if event["kind"] == "evidence_gain"]
        self.assertEqual(len(gain_events), 1)
        self.assertTrue(any(row["record"]["id"] == "decision-third"
                            for row in rows))


if __name__ == "__main__":
    unittest.main()
