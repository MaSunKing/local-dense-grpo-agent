"""Bind post-Browse coverage gain to one captured action and tool response."""
import copy
import hashlib
import json
import sys
from pathlib import Path

from active_policy_v11 import PROMPTS, DELTA_COVERAGE_PROMPT
from engine import digest, indexed, need, validate
from input_contract import semantic_evidence, validate_inputs
from pipeline import profile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "shared"))
from gain_contract import (  # noqa: E402
    GAIN_VERSION, canonical_action, canonical_coverage_view, canonical_coverage_result,
    canonical_coverage_delta_view, canonical_evidence, coverage_delta_key,
    coverage_key, coverage_rows, digest as shared_digest,
    make_coverage_delta_receipt, make_coverage_receipt,
    make_evidence_transition, make_gain_disposition, merge_coverage_delta,
    task_scope, validate_coverage_delta_result, validate_gain_assessment,
    validate_coverage_receipt, validate_tool_execution,
)

BRIDGE_VERSION = GAIN_VERSION
ALIASES = {"state_update": "state"}
COVERAGE_RANK = {"unknown": 0, "partial": 1, "direct": 2}
COVERAGE_VALUES = {"unknown": 0.0, "partial": 0.5, "direct": 1.0}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _one(payload):
    need(isinstance(payload, dict) and len(payload.get("records", [])) == 1,
         "one record payload required")
    validate_inputs(payload)
    return payload["records"][0]


def _verify_source_binding(payload, binding):
    required = {
        "question_id", "rollout_id", "record_id", "source_stage",
        "canonical_kind", "stage_file", "stage_file_sha256",
        "stage_row_index", "capture_file", "capture_id", "capture_sha256",
        "original_completion_sha256", "canonical_completion_sha256", "policy",
    }
    need(isinstance(binding, dict) and set(binding) == required,
         "source binding fields")
    rec = _one(payload)
    need(binding["question_id"] == payload["question_id"] and
         binding["rollout_id"] == payload["rollout_id"] and
         binding["record_id"] == rec["record_id"] and
         binding["policy"] == payload["policy"], "source binding identity")
    need(binding["canonical_kind"] == ALIASES.get(rec["kind"], rec["kind"]),
         "source binding kind")
    need(type(binding["stage_row_index"]) is int and binding["stage_row_index"] >= 0,
         "source row index")
    stage_path, capture_path = Path(binding["stage_file"]), Path(binding["capture_file"])
    need(stage_path.is_file() and _sha(stage_path) == binding["stage_file_sha256"],
         "source stage file drift")
    need(capture_path.is_file() and _sha(capture_path) == binding["capture_sha256"],
         "source capture drift")
    rows = json.loads(stage_path.read_text(encoding="utf-8"))
    need(isinstance(rows, list) and binding["stage_row_index"] < len(rows),
         "source row absent")
    ref = rows[binding["stage_row_index"]].get("sampling_capture", {})
    need(ref.get("path") == str(capture_path) and
         ref.get("sha256") == binding["capture_sha256"], "stage/capture mismatch")
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    need(capture.get("capture_id") == binding["capture_id"] and
         capture.get("rollout_id") == payload["rollout_id"] and
         capture.get("policy") == payload["policy"], "capture identity mismatch")
    original = capture.get("raw_completion")
    need(isinstance(original, str) and hashlib.sha256(original.encode()).hexdigest() ==
         binding["original_completion_sha256"], "original completion drift")
    need(hashlib.sha256(rec["raw_completion"].encode()).hexdigest() ==
         binding["canonical_completion_sha256"], "canonical completion drift")
    return rec, capture


def _coverage_view(payload):
    rec = _one(payload)
    evidence = semantic_evidence(rec)
    snap = rec.get("evidence_snapshot", {})
    need(type(snap.get("step")) is int and snap["step"] >= 0 and
         snap.get("digest") in {digest(rec["evidence"]), shared_digest(evidence)},
         "evidence snapshot drift")
    return canonical_coverage_view({key: copy.deepcopy(payload[key]) for key in
            ("question", "requirements", "constraints")} | {
        "evidence": evidence,
        "source_headers": copy.deepcopy(rec["source_headers"]),
    })


def _task(scorer, config, view):
    prompt = PROMPTS["coverage"] + "\n\n" + (
        ROOT / "private_judge_addendum.md").read_text(encoding="utf-8")
    normalized = canonical_coverage_view(view)
    key = coverage_key(scorer, normalized)
    request = copy.deepcopy(config) | {"messages": [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(normalized, ensure_ascii=False)},
    ]}
    return {"key": key, "request": request}


def _normalized_coverage_result(view, result):
    """Accept either a fresh atomic result or a previously frozen receipt result."""
    rows = result.get("coverage") if isinstance(result, dict) else None
    if (isinstance(rows, list) and rows and all(
            isinstance(row, dict) and set(row) ==
            {"id", "status", "evidence_ids", "reason"} for row in rows)):
        indexed_rows, score = coverage_rows(view, result, COVERAGE_VALUES)
        need(score is not None, "prior coverage receipt cannot be unobservable")
        return canonical_coverage_result({"coverage": [copy.deepcopy(indexed_rows[row["id"]])
                             for row in view["requirements"]]})
    checked = validate("coverage", view, result)
    return canonical_coverage_result({"coverage": [copy.deepcopy(checked["coverage"][row["id"]])
                         for row in view["requirements"]]})


def _delta_view(before_view, after_view, before_result):
    fixed = ("question", "requirements", "constraints")
    need(all(before_view.get(key) == after_view.get(key) for key in fixed),
         "coverage delta task drift")
    before = _normalized_coverage_result(before_view, before_result)
    old = indexed(canonical_evidence(before_view["evidence"]), "source_id")
    new = indexed(canonical_evidence(after_view["evidence"]), "source_id")
    need(set(old) <= set(new) and
         all(old[key] == new[key] for key in old),
         "coverage delta evidence is not append-only")
    added_ids = sorted(set(new) - set(old))
    return canonical_coverage_delta_view({
        "question": copy.deepcopy(before_view["question"]),
        "requirements": copy.deepcopy(before_view["requirements"]),
        "constraints": copy.deepcopy(before_view["constraints"]),
        "prior_coverage": [copy.deepcopy(indexed(before["coverage"], "id")[row["id"]])
                           for row in before_view["requirements"]],
        "retained_evidence": [copy.deepcopy(old[key]) for key in sorted(old)],
        "new_evidence": [copy.deepcopy(new[key]) for key in added_ids],
        "source_headers": copy.deepcopy(after_view["source_headers"]),
    })


def _delta_task(scorer, config, view):
    normalized = canonical_coverage_delta_view(view)
    key = coverage_delta_key(scorer, normalized)
    prompt = DELTA_COVERAGE_PROMPT + "\n\n" + (
        ROOT / "private_judge_addendum.md").read_text(encoding="utf-8")
    request = copy.deepcopy(config) | {"messages": [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(normalized, ensure_ascii=False)},
    ]}
    return {"key": key, "request": request}


def validate_retained_coverage(before_view, before_result,
                               after_view, after_result):
    """Validate one append-only coverage pair without inventing a score.

    A later materialized snapshot may add evidence, but it cannot erase the
    coverage supplied by unchanged retained evidence.  A downgrade is rejected
    as a contract error so the caller may perform its single bounded retry.  If
    the retry also fails, the caller must keep the result pending; this function
    never copies the earlier label or applies a high-water mark.
    """
    fixed = ("question", "requirements", "constraints")
    need(all(before_view.get(key) == after_view.get(key) for key in fixed),
         "cumulative coverage task drift")
    old_evidence = indexed(canonical_evidence(before_view["evidence"]),
                           "source_id")
    new_evidence = indexed(canonical_evidence(after_view["evidence"]),
                           "source_id")
    need(set(old_evidence) <= set(new_evidence) and
         all(old_evidence[key] == new_evidence[key] for key in old_evidence),
         "cumulative coverage evidence is not append-only")
    before = validate("coverage", before_view, before_result)
    after = validate("coverage", after_view, after_result)
    need(set(before["coverage"]) == set(after["coverage"]),
         "cumulative coverage denominator drift")
    for requirement_id, prior in before["coverage"].items():
        current = after["coverage"][requirement_id]
        need(COVERAGE_RANK[current["status"]] >=
             COVERAGE_RANK[prior["status"]],
             "cumulative coverage conflict: retained evidence was downgraded "
             f"for {requirement_id} ({prior['status']} -> {current['status']})")
    return {"before": before, "after": after}


def plan_gain(browse_payload, after_payload, browse_binding, after_binding,
              tool_execution_id, config, collector_lookup=None, task_modes=None,
              prior_coverage_receipt=None):
    before, capture = _verify_source_binding(browse_payload, browse_binding)
    after, _ = _verify_source_binding(after_payload, after_binding)
    fixed = ("question_id", "rollout_id", "question", "requirements",
             "requirements_digest", "constraints", "policy")
    need(all(browse_payload.get(key) == after_payload.get(key) for key in fixed),
         "task or policy drift across Browse")
    need(before["kind"] == "browse" and ALIASES.get(after["kind"], after["kind"]) == "state",
         "Browse must be followed by State snapshot")
    need(browse_binding["stage_file"] == after_binding["stage_file"] and
         browse_binding["stage_file_sha256"] == after_binding["stage_file_sha256"] and
         after_binding["stage_row_index"] == browse_binding["stage_row_index"] + 1,
         "Browse/State records are not adjacent")
    need(after["evidence_snapshot"]["step"] == before["evidence_snapshot"]["step"] + 1,
         "non-adjacent evidence snapshots")
    old, new = indexed(before["evidence"], "source_id"), indexed(after["evidence"], "source_id")
    need(set(old) <= set(new) and all(old[key] == new[key] for key in old),
         "evidence deleted or modified")
    new_ids = sorted(set(new) - set(old))
    need(callable(collector_lookup), "trusted collector resolver required")
    tool_execution = collector_lookup(tool_execution_id)
    need(isinstance(tool_execution, dict) and
         tool_execution.get("tool_execution_id") == tool_execution_id,
         "trusted tool execution absent")
    action, returned = validate_tool_execution(tool_execution)
    need(tool_execution["status"] == "succeeded",
         "failed Browse execution cannot export normal evidence gain")
    need(tool_execution["question_id"] == browse_payload["question_id"] and
         tool_execution["rollout_id"] == browse_payload["rollout_id"] and
         tool_execution["record_id"] == before["record_id"] and
         tool_execution["policy"] == browse_payload["policy"] and
         tool_execution["capture"]["capture_id"] == browse_binding["capture_id"] and
         tool_execution["capture"]["capture_sha256"] == browse_binding["capture_sha256"] and
         tool_execution["capture"]["raw_completion"] == capture["raw_completion"],
         "tool execution/capture binding mismatch")
    need(action == canonical_action(before["raw_completion"]),
         "tool execution/canonical action mismatch")
    candidates = indexed(before["visible_context"]["candidates"])
    selected = action["arguments"]["candidate_id"]
    need(selected in candidates, "selected candidate was not visible")
    returned_by_id = {row["source_id"]: row for row in returned}
    canonical_new = {row["source_id"]: row for row in canonical_evidence(
        [new[key] for key in new_ids])}
    need(all(key in returned_by_id and returned_by_id[key] == canonical_new[key]
             for key in new_ids), "new evidence not bound to tool response")
    frozen = profile(config, task_modes)
    mode = (task_modes or {}).get(browse_payload["question_id"], {}).get(
        "mode", "evidence_grounded")
    need(mode == "evidence_grounded", "Browse gain is not defined for self-contained tasks")
    scope = task_scope(browse_payload["question"], browse_payload["requirements"],
                       browse_payload["constraints"], mode)
    need(tool_execution["task_scope_digest"] == scope, "tool execution task scope drift")
    views = {"before": _coverage_view(browse_payload),
             "after": _coverage_view(after_payload)}
    transition = make_evidence_transition(
        question_id=browse_payload["question_id"],
        rollout_id=browse_payload["rollout_id"],
        browse_record_id=before["record_id"], tool_execution=tool_execution,
        policy=browse_payload["policy"], token_digest=tool_execution["token_digest"],
        task_scope_digest=scope,
        before_snapshot={"step": before["evidence_snapshot"]["step"],
                         "digest": shared_digest(canonical_evidence(views["before"]["evidence"]))},
        after_snapshot={"step": after["evidence_snapshot"]["step"],
                        "digest": shared_digest(canonical_evidence(views["after"]["evidence"]))},
        before_evidence=views["before"]["evidence"],
        after_evidence=views["after"]["evidence"],
        before_source_headers=views["before"]["source_headers"],
        after_source_headers=views["after"]["source_headers"])
    # The delta request is deliberately planned only after the immutable
    # before result has passed validation.  This prevents the model from
    # replacing or silently downgrading retained coverage.
    before_task = _task(frozen["scorer"], config, views["before"])
    inherited_result = None
    if prior_coverage_receipt is not None:
        inherited_result = validate_coverage_receipt(
            prior_coverage_receipt, scorer=frozen["scorer"],
            task_scope_digest=scope, coverage_key=before_task["key"])
        _normalized_coverage_result(views["before"], inherited_result)
    tasks = {} if inherited_result is not None else {"before": before_task}
    return {
        "version": BRIDGE_VERSION, "profile": frozen,
        "task_modes": copy.deepcopy(task_modes or {}), "task_scope_digest": scope,
        "config": copy.deepcopy(config), "browse_payload": copy.deepcopy(browse_payload),
        "after_payload": copy.deepcopy(after_payload),
        "browse_binding": copy.deepcopy(browse_binding),
        "after_binding": copy.deepcopy(after_binding),
        "tool_execution_id": tool_execution_id,
        "tool_execution": copy.deepcopy(tool_execution),
        "before_key": before_task["key"],
        "prior_coverage_receipt": copy.deepcopy(prior_coverage_receipt),
        "evidence_transition": transition,
        "selected_candidate_id": selected, "new_evidence_ids": new_ids,
        "views": views, "tasks": tasks, "training_ready": False,
    }


def plan_delta_task(planned, before_result=None):
    expected = plan_gain(planned["browse_payload"], planned["after_payload"],
                         planned["browse_binding"], planned["after_binding"],
                         planned["tool_execution_id"], planned["config"],
                         lambda key: copy.deepcopy(planned["tool_execution"])
                         if key == planned["tool_execution_id"] else None,
                         planned["task_modes"],
                         planned["prior_coverage_receipt"])
    need(planned == expected, "gain plan/config/source changed")
    if planned["prior_coverage_receipt"] is not None:
        need(before_result is None,
             "inherited coverage cannot be replaced by a fresh result")
        before_result = validate_coverage_receipt(
            planned["prior_coverage_receipt"],
            scorer=planned["profile"]["scorer"],
            task_scope_digest=planned["task_scope_digest"],
            coverage_key=planned["before_key"])
    else:
        need(isinstance(before_result, dict), "fresh before result required")
    delta_view = _delta_view(planned["views"]["before"],
                             planned["views"]["after"], before_result)
    task = _delta_task(planned["profile"]["scorer"],
                       planned["config"], delta_view)
    return {"view": delta_view, "task": task, "training_ready": False}


def record_no_state_browse(browse_payload, browse_binding, tool_execution_id,
                           config, collector_lookup=None, task_modes=None):
    """Record a terminal Browse that legitimately produced no State record.

    This path is intentionally limited to a failed read with no evidence or a
    successful empty/duplicate read.  A successful read that appends evidence
    still needs an assessed after-view and must use ``plan_gain``.
    """
    before, capture = _verify_source_binding(browse_payload, browse_binding)
    need(before["kind"] == "browse", "no-State outcome requires Browse")
    need(callable(collector_lookup), "trusted collector resolver required")
    execution = collector_lookup(tool_execution_id)
    need(isinstance(execution, dict) and
         execution.get("tool_execution_id") == tool_execution_id,
         "trusted tool execution absent")
    action, returned = validate_tool_execution(execution)
    need(execution["question_id"] == browse_payload["question_id"] and
         execution["rollout_id"] == browse_payload["rollout_id"] and
         execution["record_id"] == before["record_id"] and
         execution["policy"] == browse_payload["policy"] and
         execution["capture"]["capture_id"] == browse_binding["capture_id"] and
         execution["capture"]["capture_sha256"] == browse_binding["capture_sha256"] and
         execution["capture"]["raw_completion"] == capture["raw_completion"],
         "tool execution/capture binding mismatch")
    need(action == canonical_action(before["raw_completion"]),
         "tool execution/canonical action mismatch")
    candidates = indexed(before["visible_context"]["candidates"])
    need(action["arguments"]["candidate_id"] in candidates,
         "selected candidate was not visible")
    frozen = profile(config, task_modes)
    mode = (task_modes or {}).get(browse_payload["question_id"], {}).get(
        "mode", "evidence_grounded")
    need(mode == "evidence_grounded",
         "Browse gain is not defined for self-contained tasks")
    scope = task_scope(browse_payload["question"], browse_payload["requirements"],
                       browse_payload["constraints"], mode)
    need(execution["task_scope_digest"] == scope,
         "tool execution task scope drift")
    view = _coverage_view(browse_payload)
    old = {row["source_id"]: row for row in canonical_evidence(view["evidence"])}
    returned_by_id = {row["source_id"]: row for row in returned}
    need(all(source_id in old and old[source_id] == row
             for source_id, row in returned_by_id.items()),
         "successful new evidence requires an assessed after-view")
    need(execution["status"] == "succeeded" or not returned,
         "failed Browse returned evidence")
    before_snapshot = {"step": before["evidence_snapshot"]["step"],
                       "digest": shared_digest(canonical_evidence(view["evidence"]))}
    after_snapshot = {"step": before_snapshot["step"] + 1,
                      "digest": before_snapshot["digest"]}
    transition = make_evidence_transition(
        question_id=browse_payload["question_id"],
        rollout_id=browse_payload["rollout_id"],
        browse_record_id=before["record_id"], tool_execution=execution,
        policy=browse_payload["policy"], token_digest=execution["token_digest"],
        task_scope_digest=scope, before_snapshot=before_snapshot,
        after_snapshot=after_snapshot, before_evidence=view["evidence"],
        after_evidence=view["evidence"],
        before_source_headers=view["source_headers"],
        after_source_headers=view["source_headers"])
    status = ("observed_zero_no_change" if execution["status"] == "succeeded"
              else "failed_no_evidence")
    disposition = make_gain_disposition(
        status=status, artifact_key=None,
        question_id=browse_payload["question_id"],
        rollout_id=browse_payload["rollout_id"],
        browse_record_id=before["record_id"], after_record_id=None,
        tool_execution_id=tool_execution_id, policy=browse_payload["policy"],
        token_digest=execution["token_digest"], task_scope_digest=scope)
    return {"profile": frozen, "evidence_transition": transition,
            "gain_disposition": disposition, "artifact": None,
            "event": None, "reward_export_authorized": False,
            "training_ready": False}


def finish_gain(planned, results, *, event_id, reward_config,
                collector_lookup=None, task_modes=None):
    expected = plan_gain(planned["browse_payload"], planned["after_payload"],
                         planned["browse_binding"], planned["after_binding"],
                         planned["tool_execution_id"], planned["config"],
                         collector_lookup, task_modes,
                         planned["prior_coverage_receipt"])
    need(planned == expected, "gain plan/config/source changed")
    need(isinstance(event_id, str) and event_id.strip(), "event ID")
    values = reward_config.get("task", {}).get("coverage_values")
    need(isinstance(values, dict) and set(values) == {"unknown", "partial", "direct"}
         and values["unknown"] == 0 and values["direct"] == 1 and
         type(values["partial"]) in (int, float) and 0 <= values["partial"] <= 1,
         "coverage reward mapping")
    inherited = planned["prior_coverage_receipt"] is not None
    need(set(results) == ({"delta"} if inherited else {"before", "delta"}),
         "coverage result set")
    if inherited:
        prior_receipt = copy.deepcopy(planned["prior_coverage_receipt"])
        before_result = validate_coverage_receipt(
            prior_receipt, scorer=planned["profile"]["scorer"],
            task_scope_digest=planned["task_scope_digest"],
            coverage_key=planned["before_key"])
        normalized_results = {"before": _normalized_coverage_result(
            planned["views"]["before"], before_result)}
        delta_plan = plan_delta_task(planned)
    else:
        normalized_results = {"before": _normalized_coverage_result(
            planned["views"]["before"], results["before"])}
        prior_receipt = make_coverage_receipt(
            scorer=planned["profile"]["scorer"],
            task_scope_digest=planned["task_scope_digest"],
            coverage_key=planned["before_key"],
            result=normalized_results["before"])
        delta_plan = plan_delta_task(planned, results["before"])
    validate_coverage_delta_result(delta_plan["view"], results["delta"])
    normalized_results["after"], contradictions = merge_coverage_delta(
        delta_plan["view"], results["delta"])
    old, a = coverage_rows(planned["views"]["before"], normalized_results["before"], values)
    new, b = coverage_rows(planned["views"]["after"], normalized_results["after"], values)
    need(all(COVERAGE_RANK[new[rid]["status"]] >=
             COVERAGE_RANK[old[rid]["status"]] for rid in old),
         "incremental merge downgraded retained coverage")
    weights = {row["id"]: row.get("weight", 1)
               for row in planned["views"]["before"]["requirements"]}
    total_weight = sum(weights.values())
    transitions, positive, decreased = [], 0.0, False
    for rid in sorted(old):
        av = None if old[rid]["status"] == "unobservable" else values[old[rid]["status"]]
        bv = None if new[rid]["status"] == "unobservable" else values[new[rid]["status"]]
        change = None if av is None or bv is None else bv - av
        refs = sorted(set(new[rid]["evidence_ids"]) & set(planned["new_evidence_ids"]))
        if change is not None and change > 0:
            need(refs, "coverage increase lacks newly opened evidence")
            positive += weights[rid] * change / total_weight
        decreased |= change is not None and change < 0
        transitions.append(dict(id=rid, before=old[rid]["status"],
                                after=new[rid]["status"], delta=change,
                                new_evidence_ids=refs))
    if a is None or b is None:
        disposition, exported_delta = "unobservable", None
    elif decreased:
        disposition, exported_delta = "needs_review", None
    elif positive > 0:
        disposition, exported_delta = "observed_positive", positive
    else:
        disposition, exported_delta = "observed_zero", 0.0
    authorized = disposition == "observed_positive"
    binding = {
        "question_id": planned["browse_payload"]["question_id"],
        "rollout_id": planned["browse_payload"]["rollout_id"],
        "browse_record_id": planned["browse_payload"]["records"][0]["record_id"],
        "after_record_id": planned["after_payload"]["records"][0]["record_id"],
        "policy": planned["browse_payload"]["policy"],
        "before_snapshot": {"step": planned["browse_payload"]["records"][0]["evidence_snapshot"]["step"],
                            "digest": shared_digest(canonical_evidence(planned["views"]["before"]["evidence"]))},
        "after_snapshot": {"step": planned["after_payload"]["records"][0]["evidence_snapshot"]["step"],
                           "digest": shared_digest(canonical_evidence(planned["views"]["after"]["evidence"]))},
        "selected_candidate_id": planned["selected_candidate_id"],
        "new_evidence_ids": copy.deepcopy(planned["new_evidence_ids"]),
        "tool_execution_id": planned["tool_execution"]["tool_execution_id"],
        "task_scope_digest": planned["task_scope_digest"],
    }
    after_key = coverage_key(planned["profile"]["scorer"],
                             planned["views"]["after"])
    coverage_receipts = {
        planned["before_key"]: prior_receipt,
        after_key: make_coverage_receipt(
            scorer=planned["profile"]["scorer"],
            task_scope_digest=planned["task_scope_digest"],
            coverage_key=after_key, result=normalized_results["after"]),
    }
    delta_receipt = make_coverage_delta_receipt(
        scorer=planned["profile"]["scorer"],
        task_scope_digest=planned["task_scope_digest"],
        delta_key=delta_plan["task"]["key"],
        previous_receipt_id=prior_receipt["receipt_id"],
        view=delta_plan["view"], result=results["delta"])
    artifact = {
        "version": BRIDGE_VERSION, "scorer": planned["profile"]["scorer"],
        "reward_config": digest(reward_config),
        "task_modes_digest": digest(planned["task_modes"]),
        "task_scope_digest": planned["task_scope_digest"],
        "tool_execution": copy.deepcopy(planned["tool_execution"]),
        "binding": binding, "before_view": copy.deepcopy(planned["views"]["before"]),
        "after_view": copy.deepcopy(planned["views"]["after"]),
        "before_result": copy.deepcopy(normalized_results["before"]),
        "after_result": copy.deepcopy(normalized_results["after"]),
        "before_key": planned["before_key"],
        "after_key": after_key,
        "before_score": a, "after_score": b,
        "transitions": transitions, "disposition": disposition,
        "delta": exported_delta, "delta_receipt": delta_receipt,
        "contradictions": contradictions,
    }
    artifact_key = digest(artifact)
    assessment = validate_gain_assessment(
        artifact, scorer=planned["profile"]["scorer"],
        reward_config_digest=digest(reward_config), coverage_values=values,
        coverage_receipt_lookup=lambda key: copy.deepcopy(
            coverage_receipts.get(key)),
        evidence_transition=planned["evidence_transition"])
    need(assessment["status"] == disposition and
         assessment["value"] == exported_delta,
         "shared gain assessment verifier disagreement")
    gain_disposition = make_gain_disposition(
        status=disposition, artifact_key=artifact_key,
        question_id=binding["question_id"], rollout_id=binding["rollout_id"],
        browse_record_id=binding["browse_record_id"],
        after_record_id=binding["after_record_id"],
        tool_execution_id=binding["tool_execution_id"],
        policy=binding["policy"],
        token_digest=planned["tool_execution"]["token_digest"],
        task_scope_digest=binding["task_scope_digest"])
    event = None
    if authorized:
        event = {"id": event_id, "kind": "evidence_gain", "value": positive,
                 "tool_execution_id": planned["tool_execution"]["tool_execution_id"],
                 "proof": {"kind": "private_evidence_gain_v5",
                           "artifact_key": artifact_key,
                           "tool_execution_id": planned["tool_execution"]["tool_execution_id"],
                           "scorer": planned["profile"]["scorer"],
                           "reward_config": digest(reward_config)}}
    return {"artifact_key": artifact_key, "artifact": artifact,
            "coverage_receipts": coverage_receipts,
            "coverage_delta_receipt": delta_receipt,
            "evidence_transition": copy.deepcopy(planned["evidence_transition"]),
            "gain_disposition": gain_disposition, "event": event,
            "reward_export_authorized": authorized,
            "needs_review": disposition == "needs_review",
            "training_ready": False}
