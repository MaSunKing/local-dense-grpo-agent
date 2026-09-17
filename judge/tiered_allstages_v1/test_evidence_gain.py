import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import sys

from engine import digest
from evidence_gain import (finish_gain, plan_delta_task, plan_gain,
                           record_no_state_browse, _coverage_view)
from fixtures import ALL, REQ, case
from guards import task_scope_digest as mode_scope_digest
from pipeline import profile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"shared"))
from gain_contract import (canonical_action,canonical_evidence,coverage_key,
    digest as shared_digest,make_coverage_receipt,task_scope,
    validate_gain_artifact)

CFG = dict(model="test", temperature=0, max_tokens=3000,
           response_format={"type": "json_object"})
REWARD = {"task": {"coverage_values": {"unknown": 0, "partial": .5, "direct": 1}}}


def atomic_row(rid, status, evidence_ids=None, reason="fixture"):
    atoms = {
        "direct": (True, True, True, True),
        "partial": (True, True, True, False),
        "unknown": (False, False, False, False),
        "unobservable": (None, None, None, None),
    }[status]
    return dict(
        id=rid,
        any_requested_option_or_member_present=atoms[0],
        requested_outcome_support=atoms[1],
        population_applicable=atoms[2],
        complete_requirement_support=atoms[3],
        evidence_ids=(evidence_ids or []) if status in {"direct", "partial"} else [],
        reason=reason,
    )


def delta_row(rid, material=False, complete=False, contradiction=False,
              evidence_ids=None, reason="fixture delta"):
    flags = (material, complete, contradiction)
    return dict(
        id=rid, any_requested_option_or_member_present=material,
        requested_outcome_support=material, population_applicable=material,
        combined_requirement_complete=complete,
        contradiction=contradiction,
        basis_new_evidence_ids=(evidence_ids if evidence_ids is not None else
                                (["PERM"] if any(flags) else [])),
        reason=reason,
    )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Tests(unittest.TestCase):
    @staticmethod
    def lookup(execution):
        return lambda key: copy.deepcopy(execution) if key == execution["tool_execution_id"] else None

    def pair(self, root):
        before = case("browse")[0]
        before["records"][0]["record_id"] = "browse-1"
        before["records"][0]["evidence_snapshot"] = {
            "step": 2, "digest": digest(before["records"][0]["evidence"])}
        after = copy.deepcopy(before)
        after["records"][0].update(
            record_id="state-1", kind="state_update",
            raw_completion=json.dumps({"updates": []}),
            evidence=copy.deepcopy(ALL),
            evidence_snapshot={"step": 3, "digest": digest(ALL)},
            visible_context={"items": []})
        stage = root / "alignment.json"
        captures = []
        for index, payload in enumerate((before, after)):
            rec = payload["records"][0]
            cap = root / f"capture-{index}.json"
            value = dict(capture_id=f"cap-{index}", rollout_id=payload["rollout_id"],
                         policy=payload["policy"], raw_completion=rec["raw_completion"])
            cap.write_text(json.dumps(value), encoding="utf-8")
            captures.append(cap)
        stage.write_text(json.dumps([{"sampling_capture": {"path": str(p),
            "sha256": sha(p)}} for p in captures]), encoding="utf-8")
        bindings = []
        for index, (payload, cap) in enumerate(zip((before, after), captures)):
            rec = payload["records"][0]
            bindings.append(dict(
                question_id=payload["question_id"], rollout_id=payload["rollout_id"],
                record_id=rec["record_id"], source_stage="decision" if index == 0 else "state_update",
                canonical_kind="browse" if index == 0 else "state",
                stage_file=str(stage), stage_file_sha256=sha(stage), stage_row_index=index,
                capture_file=str(cap), capture_id=f"cap-{index}", capture_sha256=sha(cap),
                original_completion_sha256=hashlib.sha256(rec["raw_completion"].encode()).hexdigest(),
                canonical_completion_sha256=hashlib.sha256(rec["raw_completion"].encode()).hexdigest(),
                policy=payload["policy"]))
        raw = before["records"][0]["raw_completion"]
        action = canonical_action(raw)
        response_payload = {"tool": "browse", "evidence": canonical_evidence(
            [row for row in ALL if row["source_id"] == "PERM"])}
        core = dict(version="trusted_tool_execution_v1",
            question_id=before["question_id"],rollout_id=before["rollout_id"],
            record_id="browse-1",policy=before["policy"],token_digest="token-digest",
            task_scope_digest=task_scope(before["question"],before["requirements"],
                                         before["constraints"],"evidence_grounded"),
            capture=dict(capture_id="cap-0",capture_sha256=sha(captures[0]),
                parser_version="json_tool_action_v1",raw_completion=raw,
                action=action,action_digest=shared_digest(action)),
            response=dict(payload=response_payload,payload_digest=shared_digest(response_payload)),
            status="succeeded")
        execution=copy.deepcopy(core);execution["tool_execution_id"]=shared_digest(core)
        return before, after, bindings, execution

    def results(self):
        def rows(states):
            return {"coverage": [atomic_row(
                f"Q{i}", state, ["RET"] if i == 1 else ["PERM"])
                for i, state in enumerate(states, 1)]}
        return {"before": rows(("direct", "unknown", "unknown")),
                "delta": {"coverage": [
                    delta_row("Q1"),
                    delta_row("Q2", True, True),
                    delta_row("Q3", True, True),
                ]}}

    def test_prompt_separates_choice_and_outcome(self):
        from active_policy_v2 import PROMPTS
        prompt = PROMPTS["browse"]
        self.assertIn("prospective information value", prompt)
        self.assertIn("indirect source may be partially relevant", prompt)
        self.assertIn("Candidate snippets inform selection only", prompt)
        self.assertIn("Do not use the returned body", prompt)

    def test_delta_prompt_treats_one_requested_arm_as_material_support(self):
        from active_policy_v11 import DELTA_COVERAGE_PROMPT
        self.assertIn("Absence of the other arm NEVER makes this atom false",
                      DELTA_COVERAGE_PROMPT)
        self.assertIn("MRA study reporting a requested", DELTA_COVERAGE_PROMPT)
        self.assertIn("Such one-sided evidence remains incomplete",
                      DELTA_COVERAGE_PROMPT)

    def test_bound_gain_event(self):
        with tempfile.TemporaryDirectory() as d:
            before, after, bindings, execution = self.pair(Path(d))
            lookup=self.lookup(execution)
            planned = plan_gain(before, after, *bindings, execution["tool_execution_id"], CFG, lookup)
            out = finish_gain(planned, self.results(), event_id="gain-1",
                              reward_config=REWARD,collector_lookup=lookup)
            self.assertTrue(out["reward_export_authorized"])
            self.assertAlmostEqual(out["event"]["value"], 2 / 3)
            self.assertEqual(out["event"]["proof"]["artifact_key"],
                             digest(out["artifact"]))
            self.assertEqual(out["artifact"]["binding"]["new_evidence_ids"], ["PERM"])
            self.assertEqual(out["evidence_transition"]["change"], "append")

    def test_successful_empty_browse_without_state_is_zero(self):
        with tempfile.TemporaryDirectory() as d:
            before, _, bindings, execution = self.pair(Path(d))
            core = copy.deepcopy(execution); core.pop("tool_execution_id")
            payload = {"tool": "browse", "evidence": []}
            core["response"] = {"payload": payload,
                                "payload_digest": shared_digest(payload)}
            execution = copy.deepcopy(core)
            execution["tool_execution_id"] = shared_digest(core)
            out = record_no_state_browse(
                before, bindings[0], execution["tool_execution_id"], CFG,
                self.lookup(execution))
            self.assertEqual(out["gain_disposition"]["status"],
                             "observed_zero_no_change")
            self.assertEqual(out["evidence_transition"]["change"], "no_change")
            self.assertIsNone(out["event"])

    def test_failed_browse_without_state_is_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            before, _, bindings, execution = self.pair(Path(d))
            core = copy.deepcopy(execution); core.pop("tool_execution_id")
            payload = {"tool": "browse", "evidence": []}
            core["response"] = {"payload": payload,
                                "payload_digest": shared_digest(payload)}
            core["status"] = "failed"
            execution = copy.deepcopy(core)
            execution["tool_execution_id"] = shared_digest(core)
            out = record_no_state_browse(
                before, bindings[0], execution["tool_execution_id"], CFG,
                self.lookup(execution))
            self.assertEqual(out["gain_disposition"]["status"],
                             "failed_no_evidence")
            self.assertEqual(out["evidence_transition"]["change"],
                             "failed_no_change")

    def test_nonadjacent_or_mutated_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            before, after, bindings, execution = self.pair(Path(d))
            bad = copy.deepcopy(bindings[1]); bad["stage_row_index"] = 0
            with self.assertRaises(ValueError):
                plan_gain(before, after, bindings[0], bad, execution["tool_execution_id"], CFG,
                          self.lookup(execution))
            after["records"][0]["evidence"][0]["text"] = "changed"
            after["records"][0]["evidence_snapshot"]["digest"] = digest(
                after["records"][0]["evidence"])
            with self.assertRaises(ValueError):
                plan_gain(before, after, *bindings, execution["tool_execution_id"], CFG,
                          self.lookup(execution))

    def test_irrelevant_delta_cannot_downgrade_retained_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            before, after, bindings, execution = self.pair(Path(d))
            lookup=self.lookup(execution)
            planned = plan_gain(before, after, *bindings, execution["tool_execution_id"], CFG, lookup)
            results = self.results()
            results["delta"] = {"coverage": [delta_row(f"Q{i}")
                                               for i in (1, 2, 3)]}
            out = finish_gain(planned, results, event_id="gain-1",
                              reward_config=REWARD, collector_lookup=lookup)
            self.assertEqual([row["after"] for row in
                              out["artifact"]["transitions"]],
                             ["direct", "unknown", "unknown"])
            self.assertEqual(out["artifact"]["delta"], 0.0)

    def test_contradiction_is_separate_and_does_not_erase_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            before, after, bindings, execution = self.pair(Path(d))
            lookup=self.lookup(execution)
            planned = plan_gain(before, after, *bindings, execution["tool_execution_id"], CFG, lookup)
            results = self.results()
            results["delta"]["coverage"][0] = delta_row(
                "Q1", contradiction=True)
            out = finish_gain(planned, results, event_id="gain-1",
                              reward_config=REWARD, collector_lookup=lookup)
            self.assertEqual(out["artifact"]["transitions"][0]["after"],
                             "direct")
            self.assertEqual(out["artifact"]["contradictions"][0]["id"],
                             "Q1")

    def test_retained_partial_is_accepted_without_silent_upgrade(self):
        with tempfile.TemporaryDirectory() as d:
            before, after, bindings, execution = self.pair(Path(d))
            lookup = self.lookup(execution)
            planned = plan_gain(before, after, *bindings,
                                execution["tool_execution_id"], CFG, lookup)
            results = self.results()
            results["before"] = {"coverage": [
                atomic_row("Q1", "partial", ["RET"]),
                atomic_row("Q2", "unknown"),
                atomic_row("Q3", "unknown"),
            ]}
            results["delta"] = {"coverage": [
                delta_row("Q1"),
                delta_row("Q2", True, True),
                delta_row("Q3", True, True),
            ]}
            out = finish_gain(planned, results, event_id="gain-1",
                              reward_config=REWARD, collector_lookup=lookup)
            self.assertEqual(out["artifact"]["transitions"][0]["before"],
                             "partial")
            self.assertEqual(out["artifact"]["transitions"][0]["after"],
                             "partial")
            self.assertTrue(out["reward_export_authorized"])

    def test_prior_after_receipt_is_handed_to_next_browse_without_regrading(self):
        with tempfile.TemporaryDirectory() as d:
            before, after, bindings, execution = self.pair(Path(d))
            lookup = self.lookup(execution)
            scorer = profile(CFG)["scorer"]
            scope = task_scope(before["question"], before["requirements"],
                               before["constraints"], "evidence_grounded")
            normalized = {"coverage": [
                {"id": "Q1", "status": "partial",
                 "evidence_ids": ["RET"], "reason": "trusted prior"},
                {"id": "Q2", "status": "unknown",
                 "evidence_ids": [], "reason": "trusted prior"},
                {"id": "Q3", "status": "unknown",
                 "evidence_ids": [], "reason": "trusted prior"},
            ]}
            view = _coverage_view(before)
            key = coverage_key(scorer, view)
            receipt = make_coverage_receipt(
                scorer=scorer, task_scope_digest=scope,
                coverage_key=key, result=normalized)
            planned = plan_gain(
                before, after, *bindings, execution["tool_execution_id"], CFG,
                lookup, prior_coverage_receipt=receipt)
            self.assertEqual(planned["tasks"], {})
            delta_plan = plan_delta_task(planned)
            self.assertIn("task", delta_plan)
            results = {"delta": {"coverage": [
                delta_row("Q1"),
                delta_row("Q2", True, True),
                delta_row("Q3", True, True),
            ]}}
            out = finish_gain(
                planned, results, event_id="gain-handoff",
                reward_config=REWARD, collector_lookup=lookup)
            self.assertEqual(
                out["coverage_delta_receipt"]["previous_receipt_id"],
                receipt["receipt_id"])
            self.assertEqual(
                out["coverage_receipts"][key]["receipt_id"],
                receipt["receipt_id"])

            tampered = copy.deepcopy(receipt)
            tampered["result"]["coverage"][0]["status"] = "unknown"
            with self.assertRaises(ValueError):
                plan_gain(before, after, *bindings,
                          execution["tool_execution_id"], CFG, lookup,
                          prior_coverage_receipt=tampered)

    def test_canonical_action_and_tool_response_are_causal_bindings(self):
        with tempfile.TemporaryDirectory() as d:
            before, after, bindings, execution = self.pair(Path(d))
            changed=copy.deepcopy(before)
            changed["records"][0]["raw_completion"]=json.dumps(
                {"tool":"browse","arguments":{"candidate_id":"other","focus":"x"}})
            bindings[0]["canonical_completion_sha256"]=hashlib.sha256(
                changed["records"][0]["raw_completion"].encode()).hexdigest()
            with self.assertRaises(ValueError):
                plan_gain(changed,after,*bindings,execution["tool_execution_id"],CFG,
                          self.lookup(execution))
            before, after, bindings, execution = self.pair(Path(d))
            bad=copy.deepcopy(execution)
            bad["response"]["payload"]["evidence"][0]["text"]="fabricated"
            bad["response"]["payload_digest"]=shared_digest(bad["response"]["payload"])
            core=copy.deepcopy(bad);core.pop("tool_execution_id")
            bad["tool_execution_id"]=shared_digest(core)
            with self.assertRaises(ValueError):
                plan_gain(before,after,*bindings,bad["tool_execution_id"],CFG,
                          self.lookup(bad))

    def test_no_new_body_cannot_be_metadata_gain(self):
        with tempfile.TemporaryDirectory() as d:
            before, after, bindings, execution = self.pair(Path(d))
            after["records"][0]["evidence"]=copy.deepcopy(before["records"][0]["evidence"])
            after["records"][0]["evidence_snapshot"]={
                "step":3,"digest":digest(after["records"][0]["evidence"])}
            with self.assertRaises(ValueError):
                plan_gain(before,after,*bindings,execution["tool_execution_id"],CFG,
                          self.lookup(execution))

    def test_nonempty_task_mode_registry_uses_same_scorer(self):
        with tempfile.TemporaryDirectory() as d:
            before, after, bindings, execution = self.pair(Path(d))
            modes={before["question_id"]:{"mode":"evidence_grounded",
                "task_scope_digest":mode_scope_digest(before)}}
            planned=plan_gain(before,after,*bindings,execution["tool_execution_id"],CFG,
                              self.lookup(execution),modes)
            self.assertEqual(planned["profile"]["scorer"],profile(CFG,modes)["scorer"])

    def test_materialized_coverage_unobservable_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            before,after,bindings,execution=self.pair(Path(d));lookup=self.lookup(execution)
            planned=plan_gain(before,after,*bindings,execution["tool_execution_id"],CFG,lookup)
            results=self.results()
            results["delta"]["coverage"][0]["any_requested_option_or_member_present"] = None
            with self.assertRaisesRegex(ValueError, 'flags must be boolean'):
                finish_gain(planned,results,event_id="gain-1",reward_config=REWARD,
                            collector_lookup=lookup)

    def test_v15_shared_rejects_header_drift_and_failed_execution(self):
        with tempfile.TemporaryDirectory() as d:
            before,after,bindings,execution=self.pair(Path(d));lookup=self.lookup(execution)
            before["records"][0]["source_headers"]={"RET":{"title":"Original"}}
            after["records"][0]["source_headers"]=copy.deepcopy(before["records"][0]["source_headers"])
            planned=plan_gain(before,after,*bindings,execution["tool_execution_id"],CFG,lookup)
            out=finish_gain(planned,self.results(),event_id="gain-1",reward_config=REWARD,
                            collector_lookup=lookup)
            artifact=copy.deepcopy(out["artifact"])
            artifact["after_view"]["source_headers"]["RET"]["title"]="Changed"
            artifact["after_key"]=digest({"scorer":artifact["scorer"],"task":"coverage",
                                           "view":artifact["after_view"]})
            with self.assertRaises(ValueError):validate_gain_artifact(
                artifact,scorer=artifact["scorer"],
                reward_config_digest=digest(REWARD),
                coverage_values=REWARD["task"]["coverage_values"])

            before,after,bindings,execution=self.pair(Path(d))
            failed=copy.deepcopy(execution);failed["status"]="failed"
            core=copy.deepcopy(failed);core.pop("tool_execution_id")
            failed["tool_execution_id"]=shared_digest(core)
            with self.assertRaises(ValueError):plan_gain(
                before,after,*bindings,failed["tool_execution_id"],CFG,self.lookup(failed))


if __name__ == "__main__":
    unittest.main()
