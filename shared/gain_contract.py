"""Shared fail-closed contracts for tool execution and post-Browse evidence gain.

This module is imported by both the Judge bridge and the training compiler.  It
contains no network, model, or optimizer code.  All digests are over canonical
JSON, and event display IDs are deliberately excluded from business identity.
"""
import copy
import hashlib
import json
import math
import re


TOOL_EXECUTION_VERSION = "trusted_tool_execution_v1"
LEGACY_GAIN_VERSION = "browse_evidence_gain_v4"
GAIN_VERSION = "browse_evidence_gain_v5"
PARSER_VERSION = "json_tool_action_v1"
XML_PARSER_VERSION = "xml_call_tool_v1"
PARSER_VERSIONS = {PARSER_VERSION, XML_PARSER_VERSION}
TERMINATION_VERSION = "trusted_termination_event_v1"
EXECUTION_MANIFEST_VERSION = "trusted_execution_manifest_v1"
VIOLATION_VERSION = "trusted_policy_violation_v1"
COVERAGE_RECEIPT_VERSION = "trusted_coverage_receipt_v2"
COVERAGE_DELTA_RECEIPT_VERSION = "trusted_coverage_delta_receipt_v2"
COVERAGE_MERGE_POLICY_VERSION = "monotonic_incremental_coverage_v2"
GAIN_DISPOSITION_VERSION = "trusted_gain_disposition_v1"
EVIDENCE_TRANSITION_VERSION = "trusted_evidence_transition_v2"
STATUS_VALUES = {"unknown", "partial", "direct", "unobservable"}
HEADER_FIELDS = {"title", "url", "source_type", "published_at"}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def finite(value):
    need(not isinstance(value, bool) and isinstance(value, (int, float)) and
         math.isfinite(value), "finite number required")
    return float(value)


def sha256_text(value):
    return (isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value))


def _json_action(raw):
    need(isinstance(raw, str) and raw.strip(), "captured action text required")
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("unsupported captured action encoding") from exc
    need(isinstance(value, dict) and set(value) == {"tool", "arguments"},
         "canonical action fields")
    need(value["tool"] in {"search", "browse"} and
         isinstance(value["arguments"], dict), "supported tool action required")
    expected = {"query"} if value["tool"] == "search" else {"candidate_id", "focus"}
    arguments = value["arguments"]
    need(set(arguments) == expected and
         all(isinstance(v, str) for v in arguments.values()),
         "canonical tool arguments")
    need(arguments["query"].strip() if value["tool"] == "search" else
         arguments["candidate_id"].strip(), "empty canonical tool target")
    return {"tool": value["tool"], "arguments": {
        key: value["arguments"][key] for key in sorted(expected)}}


def _xml_action(raw):
    """Strictly project the XML action format accepted by the frozen runtime."""
    need(isinstance(raw, str) and raw.strip(), "captured action text required")
    public = re.sub(r"<think>.*?</think>", "", raw,
                    flags=re.IGNORECASE | re.DOTALL).strip()
    match = re.fullmatch(
        r'<call_tool\s+name="([a-z_]+)"([^<>]*)>([^<>]+)</call_tool>',
        public, flags=re.IGNORECASE | re.DOTALL)
    need(match is not None, "unsupported captured XML action encoding")
    name, raw_attributes, body = match.groups()
    body = body.strip()
    need(body, "empty XML tool action body")
    attributes = {}
    parts = list(re.finditer(r'\s+([a-z_]+)="([^"<>]*)"', raw_attributes,
                             flags=re.IGNORECASE))
    need("".join(part.group(0) for part in parts).strip() ==
         raw_attributes.strip(), "unsupported XML tool attributes")
    for part in parts:
        key, value = part.group(1).lower(), part.group(2).strip()
        need(key not in attributes, "duplicate XML tool attribute")
        attributes[key] = value
    name = name.lower()
    if name in {"pubmed_search", "medical_web_search"}:
        need(not attributes, "search attributes require a newer parser policy")
        return {"tool": "search", "arguments": {"query": body}}
    need(name in {"browse_document", "browse_webpage"} and
         set(attributes) <= {"query"},
         "canonical Browse XML fields")
    return {"tool": "browse", "arguments": {
        "candidate_id": body, "focus": attributes.get("query", "")}}


def canonical_action(raw, parser_version=PARSER_VERSION):
    """Project captured action bytes without rewriting the original completion."""
    need(parser_version in PARSER_VERSIONS, "unknown action parser")
    return (_json_action(raw) if parser_version == PARSER_VERSION
            else _xml_action(raw))


def canonical_evidence(rows):
    allowed = {"source_id", "text", "title", "url", "source_type", "published_at"}
    need(isinstance(rows, list), "tool response evidence list")
    result, seen = [], set()
    for row in rows:
        need(isinstance(row, dict) and {"source_id", "text"} <= set(row) <= allowed,
             "tool response evidence fields")
        need(all(isinstance(v, str) for v in row.values()),
             "tool response evidence strings")
        need(row["source_id"].strip() and row["text"].strip() and
             row["source_id"] not in seen, "tool response evidence identity")
        seen.add(row["source_id"])
        result.append({key: copy.deepcopy(row[key]) for key in sorted(row)})
    return sorted(result, key=lambda row: row["source_id"])


def canonical_source_headers(headers, evidence_ids):
    """Canonical metadata for sources already visible in one coverage view."""
    need(isinstance(headers, dict), "source headers object required")
    result = {}
    for source_id, metadata in headers.items():
        need(source_id in evidence_ids and isinstance(metadata, dict) and
             set(metadata) <= HEADER_FIELDS, "source header identity/fields")
        need(all(isinstance(value, str) for value in metadata.values()),
             "source header strings")
        normalized = {key: metadata[key] for key in sorted(metadata)}
        # An explicit empty source entry carries no information.  Dropping it
        # makes {} and {"E1": {}} the same coverage input instead of allowing
        # a no-op representation change to mint a new Judge identity.
        if normalized:
            result[source_id] = normalized
    return {key: result[key] for key in sorted(result)}


def canonical_coverage_view(view):
    """One shared normalization for Judge requests, keys, receipts and training."""
    need(isinstance(view, dict) and set(view) == {
        "question", "requirements", "constraints", "evidence", "source_headers"},
        "coverage view fields")
    evidence = canonical_evidence(view["evidence"])
    headers = canonical_source_headers(
        view["source_headers"], {row["source_id"] for row in evidence})
    return {
        "question": copy.deepcopy(view["question"]),
        "requirements": copy.deepcopy(view["requirements"]),
        "constraints": copy.deepcopy(view["constraints"]),
        "evidence": evidence,
        "source_headers": headers,
    }


def coverage_key(scorer, view):
    need(isinstance(scorer, str) and scorer, "coverage scorer required")
    return digest({"scorer": scorer, "task": "coverage",
                   "view": canonical_coverage_view(view)})


def canonical_coverage_delta_view(view):
    """Canonical incremental view: prior state is context, only new rows are judged."""
    required = {"question", "requirements", "constraints", "prior_coverage",
                "retained_evidence", "new_evidence", "source_headers"}
    need(isinstance(view, dict) and set(view) == required,
         "coverage delta view fields")
    retained = canonical_evidence(view["retained_evidence"])
    added = canonical_evidence(view["new_evidence"])
    retained_ids = {row["source_id"] for row in retained}
    added_ids = {row["source_id"] for row in added}
    need(retained_ids.isdisjoint(added_ids), "coverage delta evidence overlap")
    requirements = {row.get("id"): row for row in view["requirements"]
                    if isinstance(row, dict)}
    need(None not in requirements and len(requirements) == len(view["requirements"]),
         "coverage delta requirements")
    prior = {}
    need(isinstance(view["prior_coverage"], list),
         "coverage delta prior list")
    for row in canonical_coverage_result({"coverage": view["prior_coverage"]})["coverage"]:
        need(isinstance(row, dict) and set(row) == {
            "id", "status", "evidence_ids", "reason"} and
            row["id"] in requirements and row["id"] not in prior and
            row["status"] in STATUS_VALUES - {"unobservable"} and
            isinstance(row["reason"], str) and row["reason"].strip(),
            "coverage delta prior row")
        refs = row["evidence_ids"]
        need(isinstance(refs, list) and len(refs) == len(set(refs)) and
             set(refs) <= retained_ids, "coverage delta prior references")
        need((row["status"] in {"partial", "direct"}) == bool(refs),
             "coverage delta prior support binding")
        prior[row["id"]] = copy.deepcopy(row)
    need(set(prior) == set(requirements), "coverage delta prior denominator")
    headers = canonical_source_headers(
        view["source_headers"], retained_ids | added_ids)
    return {
        "question": copy.deepcopy(view["question"]),
        "requirements": copy.deepcopy(view["requirements"]),
        "constraints": copy.deepcopy(view["constraints"]),
        "prior_coverage": [prior[row["id"]] for row in view["requirements"]],
        "retained_evidence": retained,
        "new_evidence": added,
        "source_headers": headers,
    }


def coverage_delta_key(scorer, view):
    need(isinstance(scorer, str) and scorer, "coverage delta scorer required")
    return digest({"scorer": scorer, "task": "coverage_delta",
                   "view": canonical_coverage_delta_view(view)})


def validate_coverage_delta_result(view, result):
    normalized = canonical_coverage_delta_view(view)
    requirement_ids = {row["id"] for row in normalized["requirements"]}
    new_ids = {row["source_id"] for row in normalized["new_evidence"]}
    need(isinstance(result, dict) and set(result) == {"coverage"} and
         isinstance(result["coverage"], list), "coverage delta result fields")
    rows = {}
    for row in result["coverage"]:
        need(isinstance(row, dict) and set(row) == {
            "id", "any_requested_option_or_member_present",
            "requested_outcome_support", "population_applicable", "combined_requirement_complete",
            "contradiction", "basis_new_evidence_ids", "reason"} and
            row["id"] in requirement_ids and row["id"] not in rows,
            "coverage delta row fields")
        atoms = [row["any_requested_option_or_member_present"],
                 row["requested_outcome_support"], row["population_applicable"]]
        flags = atoms + [
                 row["combined_requirement_complete"], row["contradiction"]]
        need(all(type(value) is bool for value in flags),
             "coverage delta flags must be boolean")
        need(not row["combined_requirement_complete"] or
             all(atoms),
             "complete delta requires material new support")
        refs = row["basis_new_evidence_ids"]
        need(isinstance(refs, list) and len(refs) == len(set(refs)) and
             set(refs) <= new_ids, "coverage delta evidence reference")
        need(bool(refs) == (all(atoms) or row["contradiction"]),
             "coverage delta true flags require new evidence and zero delta cannot cite")
        need(isinstance(row["reason"], str) and row["reason"].strip(),
             "coverage delta reason")
        rows[row["id"]] = copy.deepcopy(row)
        # This derived field is internal only; the Judge must not supply it.
        rows[row["id"]]["new_material_support"] = all(atoms)
    need(set(rows) == requirement_ids, "coverage delta denominator")
    return {"coverage": rows}


def merge_coverage_delta(view, result):
    """Merge a validated delta without permitting ordinary evidence downgrades."""
    normalized = canonical_coverage_delta_view(view)
    delta = validate_coverage_delta_result(normalized, result)["coverage"]
    prior = {row["id"]: row for row in normalized["prior_coverage"]}
    merged, contradictions = [], []
    for requirement in normalized["requirements"]:
        rid = requirement["id"]
        old, change = prior[rid], delta[rid]
        if old["status"] == "direct" or change["combined_requirement_complete"]:
            status = "direct"
        elif old["status"] == "partial" or change["new_material_support"]:
            status = "partial"
        else:
            status = "unknown"
        refs = list(old["evidence_ids"])
        if change["new_material_support"]:
            refs = sorted(set(refs) | set(change["basis_new_evidence_ids"]))
        reason = old["reason"]
        if change["new_material_support"] or change["combined_requirement_complete"]:
            reason = reason + " Incremental evidence: " + change["reason"]
        merged.append({"id": rid, "status": status,
                       "evidence_ids": refs, "reason": reason})
        if change["contradiction"]:
            contradictions.append({"id": rid,
                                   "evidence_ids": list(change["basis_new_evidence_ids"]),
                                   "reason": change["reason"]})
    return canonical_coverage_result({"coverage": merged}), contradictions


def coverage_basis_digest(view):
    """Step-independent identity of the evidence actually judged for coverage."""
    normalized = canonical_coverage_view(view)
    return digest({"evidence": normalized["evidence"],
                   "source_headers": normalized["source_headers"]})


def record_target_digest(*, question_id, rollout_id, record, channel,
                         task_scope_digest):
    """Bind a private stage score to the exact sampled target and token span."""
    need(isinstance(record, dict) and channel in record.get("channels", {}),
         "scored channel missing from record")
    row = record["channels"][channel]
    target = {
        "question_id": question_id,
        "rollout_id": rollout_id,
        "record_id": record.get("id"),
        "stage": record.get("stage"),
        "channel": channel,
        "policy": record.get("policy"),
        "input_ids": record.get("input_ids"),
        "output_ids": record.get("output_ids"),
        "token_digest": record.get("token_digest"),
        "channel_indices": row.get("indices"),
        "evidence_snapshot": record.get("evidence_snapshot"),
        "task_scope_digest": task_scope_digest,
    }
    need(all(isinstance(target[key], str) and target[key] for key in
             ("question_id", "rollout_id", "record_id", "stage", "channel",
              "policy", "token_digest", "task_scope_digest")),
         "stage target identity")
    need(isinstance(target["input_ids"], list) and
         isinstance(target["output_ids"], list) and
         isinstance(target["channel_indices"], list), "stage target tokens")
    return digest(target)


def canonical_stop_context(value):
    required = {"remaining_budget", "voluntary", "available_candidates",
                "failed_reads", "task_constraints"}
    need(isinstance(value, dict) and set(value) == required,
         "termination stop context fields")
    need(type(value["remaining_budget"]) is int and
         value["remaining_budget"] >= 0 and type(value["voluntary"]) is bool,
         "termination budget/voluntary fields")
    candidates, seen = [], set()
    for row in value["available_candidates"]:
        allowed = {"id", "title", "snippet", "url", "source_type"}
        need(isinstance(row, dict) and {"id", "title"} <= set(row) <= allowed and
             all(isinstance(item, str) for item in row.values()),
             "termination candidate fields")
        need(row["id"].strip() and row["title"].strip() and row["id"] not in seen,
             "termination candidate identity")
        seen.add(row["id"])
        candidates.append({key: row[key] for key in sorted(row)})
    failures = []
    for row in value["failed_reads"]:
        need(isinstance(row, dict) and set(row) == {"candidate_id", "error_type"}
             and all(isinstance(item, str) and item.strip()
                     for item in row.values()), "termination failed-read fields")
        failures.append({"candidate_id": row["candidate_id"],
                         "error_type": row["error_type"]})
    need(isinstance(value["task_constraints"], list) and
         all(isinstance(item, str) and item.strip()
             for item in value["task_constraints"]),
         "termination task constraints")
    return {"remaining_budget": value["remaining_budget"],
            "voluntary": value["voluntary"],
            "available_candidates": candidates,
            "failed_reads": failures,
            "task_constraints": list(value["task_constraints"])}


def validate_termination_event(event):
    required = {"version", "event_id", "type", "binding", "evidence",
                "source_headers", "stop_context", "capture"}
    need(isinstance(event, dict) and set(event) == required and
         event["version"] == TERMINATION_VERSION,
         "termination event fields")
    need(event["type"] in {"active", "quota_exhausted", "forced", "error",
                           "cancelled"}, "termination type")
    binding = event["binding"]
    binding_fields = {"question_id", "rollout_id", "record_id", "policy",
                      "token_digest", "task_scope_digest", "evidence_snapshot"}
    need(isinstance(binding, dict) and set(binding) == binding_fields and
         all(isinstance(binding[key], str) and binding[key] for key in
             binding_fields - {"evidence_snapshot"}),
         "termination binding fields")
    evidence = canonical_evidence(event["evidence"])
    need(event["evidence"] == evidence, "noncanonical termination evidence")
    headers = canonical_source_headers(event["source_headers"],
                                       {row["source_id"] for row in evidence})
    need(event["source_headers"] == headers, "noncanonical termination headers")
    snapshot = binding["evidence_snapshot"]
    need(isinstance(snapshot, dict) and set(snapshot) == {"step", "digest"} and
         type(snapshot["step"]) is int and snapshot["step"] >= 0 and
         snapshot["digest"] == digest(evidence), "termination snapshot")
    context = canonical_stop_context(event["stop_context"])
    need(event["stop_context"] == context, "noncanonical termination context")
    need(context["voluntary"] == (event["type"] == "active"),
         "termination voluntary/type mismatch")
    need(not (event["type"] == "active" and context["remaining_budget"] == 0),
         "zero-budget Stop is not active")
    need(not (event["type"] == "quota_exhausted" and
              context["remaining_budget"] != 0), "quota/budget mismatch")
    capture = event["capture"]
    capture_fields = {"capture_id", "capture_sha256", "question_id", "rollout_id",
                      "record_id", "policy", "input_ids", "output_ids",
                      "raw_completion"}
    need(isinstance(capture, dict) and set(capture) == capture_fields and
         all(isinstance(capture[key], str) and capture[key] for key in
             ("capture_id", "capture_sha256", "question_id", "rollout_id",
              "record_id", "policy")) and
         isinstance(capture["raw_completion"], str), "termination capture fields")
    need(sha256_text(capture["capture_sha256"]),
         "termination capture sha256")
    need(isinstance(capture["input_ids"], list) and capture["input_ids"] and
         isinstance(capture["output_ids"], list) and capture["output_ids"] and
         all(type(token) is int and token >= 0 for token in
             capture["input_ids"] + capture["output_ids"]),
         "termination capture tokens")
    for key in ("question_id", "rollout_id", "record_id", "policy"):
        need(capture[key] == binding[key], "termination capture identity")
    need(digest([capture["input_ids"], capture["output_ids"]]) ==
         binding["token_digest"], "termination capture token binding")
    core = copy.deepcopy(event)
    core.pop("event_id")
    need(event["event_id"] == digest(core), "termination business identity")
    return event["type"]


def validate_execution_manifest(manifest, *, question_id, rollout_id):
    required = {"version", "manifest_id", "question_id", "rollout_id",
                "execution_ids"}
    need(isinstance(manifest, dict) and set(manifest) == required and
         manifest["version"] == EXECUTION_MANIFEST_VERSION and
         manifest["question_id"] == question_id and
         manifest["rollout_id"] == rollout_id,
         "execution manifest fields")
    ids = manifest["execution_ids"]
    need(isinstance(ids, list) and len(ids) == len(set(ids)) and
         all(isinstance(item, str) and item for item in ids),
         "execution manifest identities")
    core = copy.deepcopy(manifest)
    core.pop("manifest_id")
    need(manifest["manifest_id"] == digest(core),
         "execution manifest business identity")
    return set(ids)


def canonical_coverage_result(result):
    """Canonical decision state; raw explanations belong to response audits.

    Do not accept atomic or extra fields here: callers must first validate and
    map atomic judgments. Only explanation wording and set ordering are
    representation differences. Status and evidence references remain exact.
    """
    need(isinstance(result, dict) and set(result) == {"coverage"},
         "normalized coverage result fields")
    rows = result["coverage"]
    need(isinstance(rows, list), "normalized coverage rows")
    normalized, seen = [], set()
    for row in rows:
        need(isinstance(row, dict) and set(row) ==
             {"id", "status", "evidence_ids", "reason"},
             "normalized coverage decision fields")
        rid, status, refs = row["id"], row["status"], row["evidence_ids"]
        need(isinstance(rid, str) and rid and rid not in seen,
             "normalized coverage requirement identity")
        seen.add(rid)
        need(isinstance(status, str) and status in STATUS_VALUES,
             "normalized coverage status")
        need(isinstance(refs, list) and all(isinstance(ref, str) and ref for ref in refs)
             and len(refs) == len(set(refs)), "normalized coverage references")
        need(isinstance(row["reason"], str) and row["reason"].strip(),
             "normalized coverage explanation")
        decision = {"id": rid, "status": status, "evidence_ids": sorted(refs)}
        normalized.append({**decision, "reason": "Verified coverage decision " + digest(decision)})
    return {"coverage": sorted(normalized, key=lambda row: row["id"])}


def validate_coverage_receipt(receipt, *, scorer, task_scope_digest,
                              coverage_key):
    required = {"version", "receipt_id", "scorer", "task_scope_digest",
                "coverage_key", "result"}
    need(isinstance(receipt, dict) and set(receipt) == required and
         receipt["version"] == COVERAGE_RECEIPT_VERSION,
         "coverage receipt fields")
    need(receipt["scorer"] == scorer and
         receipt["task_scope_digest"] == task_scope_digest and
         receipt["coverage_key"] == coverage_key,
         "coverage receipt binding mismatch")
    core = copy.deepcopy(receipt)
    core.pop("receipt_id")
    need(receipt["receipt_id"] == digest(core),
         "coverage receipt business identity")
    need(receipt["result"] == canonical_coverage_result(receipt["result"]),
         "coverage receipt decision state is not canonical")
    return copy.deepcopy(receipt["result"])


def make_coverage_receipt(*, scorer, task_scope_digest, coverage_key, result):
    core = {"version": COVERAGE_RECEIPT_VERSION, "scorer": scorer,
            "task_scope_digest": task_scope_digest,
            "coverage_key": coverage_key, "result": canonical_coverage_result(result)}
    return {"receipt_id": digest(core), **core}


def make_coverage_delta_receipt(*, scorer, task_scope_digest, delta_key,
                                previous_receipt_id, view, result):
    normalized_view = canonical_coverage_delta_view(view)
    validate_coverage_delta_result(normalized_view, result)
    core = {
        "version": COVERAGE_DELTA_RECEIPT_VERSION,
        "scorer": scorer,
        "task_scope_digest": task_scope_digest,
        "delta_key": delta_key,
        "previous_receipt_id": previous_receipt_id,
        "merge_policy_version": COVERAGE_MERGE_POLICY_VERSION,
        "view": normalized_view,
        "result": copy.deepcopy(result),
    }
    return {"receipt_id": digest(core), **core}


def validate_coverage_delta_receipt(receipt, *, scorer, task_scope_digest,
                                    delta_key, previous_receipt_id):
    required = {"version", "receipt_id", "scorer", "task_scope_digest",
                "delta_key", "previous_receipt_id", "merge_policy_version",
                "view", "result"}
    need(isinstance(receipt, dict) and set(receipt) == required and
         receipt["version"] == COVERAGE_DELTA_RECEIPT_VERSION and
         receipt["merge_policy_version"] == COVERAGE_MERGE_POLICY_VERSION,
         "coverage delta receipt fields")
    need(receipt["scorer"] == scorer and
         receipt["task_scope_digest"] == task_scope_digest and
         receipt["delta_key"] == delta_key and
         receipt["previous_receipt_id"] == previous_receipt_id,
         "coverage delta receipt binding mismatch")
    need(delta_key == coverage_delta_key(scorer, receipt["view"]),
         "coverage delta scorer/input mismatch")
    validate_coverage_delta_result(receipt["view"], receipt["result"])
    core = copy.deepcopy(receipt)
    core.pop("receipt_id")
    need(receipt["receipt_id"] == digest(core),
         "coverage delta receipt business identity")
    return {"view": copy.deepcopy(receipt["view"]),
            "result": copy.deepcopy(receipt["result"])}


def validate_gain_disposition(disposition, expected):
    required = {"version", "disposition_id", "status", "question_id",
                "rollout_id", "browse_record_id", "after_record_id",
                "tool_execution_id", "policy", "token_digest",
                "task_scope_digest", "artifact_key"}
    statuses = {"observed_positive", "observed_zero", "unobservable",
                "observed_zero_no_change", "needs_review", "pending",
                "failed_no_evidence"}
    need(isinstance(disposition, dict) and set(disposition) == required and
         disposition["version"] == GAIN_DISPOSITION_VERSION and
         disposition["status"] in statuses,
         "gain disposition fields")
    need(all(disposition.get(key) == value for key, value in expected.items()),
         "gain disposition binding mismatch")
    completed = {"observed_positive", "observed_zero", "unobservable",
                 "needs_review"}
    if disposition["status"] in completed:
        need(sha256_text(disposition["artifact_key"]),
             "completed gain disposition requires assessment artifact")
    else:
        need(disposition["artifact_key"] is None,
             "pending/failed gain disposition cannot claim assessment")
    core = copy.deepcopy(disposition)
    core.pop("disposition_id")
    need(disposition["disposition_id"] == digest(core),
         "gain disposition business identity")
    return disposition["status"]


def make_gain_disposition(*, status, artifact_key, question_id, rollout_id,
                          browse_record_id, after_record_id,
                          tool_execution_id, policy, token_digest,
                          task_scope_digest):
    core = {"version": GAIN_DISPOSITION_VERSION, "status": status,
            "question_id": question_id, "rollout_id": rollout_id,
            "browse_record_id": browse_record_id,
            "after_record_id": after_record_id,
            "tool_execution_id": tool_execution_id, "policy": policy,
            "token_digest": token_digest,
            "task_scope_digest": task_scope_digest,
            "artifact_key": artifact_key}
    return {"disposition_id": digest(core), **core}


def make_evidence_transition(*, question_id, rollout_id, browse_record_id,
                             tool_execution, policy, token_digest,
                             task_scope_digest, before_snapshot, after_snapshot,
                             before_evidence, after_evidence,
                             before_source_headers=None,
                             after_source_headers=None):
    """Create one collector-authoritative Browse ledger transition.

    This ledger is independent of whether the policy emits a following State.
    It records evidence acquired by the tool, not the evidence subset retained
    in the model-visible context window.
    """
    before = canonical_evidence(before_evidence)
    after = canonical_evidence(after_evidence)
    before_headers = canonical_source_headers(
        before_source_headers or {}, {row["source_id"] for row in before})
    after_headers = canonical_source_headers(
        after_source_headers or {}, {row["source_id"] for row in after})
    old = {row["source_id"]: row for row in before}
    new = {row["source_id"]: row for row in after}
    need(set(old) <= set(new) and all(old[key] == new[key] for key in old),
         "evidence ledger deleted or modified")
    added = sorted(set(new) - set(old))
    status = tool_execution.get("status") if isinstance(tool_execution, dict) else None
    if status == "succeeded":
        change = "append" if added else "no_change"
    elif status == "failed":
        need(not added, "failed Browse cannot append evidence")
        change = "failed_no_change"
    else:
        raise ValueError("evidence transition requires terminal tool execution")
    core = {
        "version": EVIDENCE_TRANSITION_VERSION,
        "question_id": question_id,
        "rollout_id": rollout_id,
        "browse_record_id": browse_record_id,
        "tool_execution_id": tool_execution.get("tool_execution_id"),
        "policy": policy,
        "token_digest": token_digest,
        "task_scope_digest": task_scope_digest,
        "before_snapshot": copy.deepcopy(before_snapshot),
        "after_snapshot": copy.deepcopy(after_snapshot),
        "before_evidence": before,
        "after_evidence": after,
        "before_source_headers": before_headers,
        "after_source_headers": after_headers,
        "change": change,
    }
    transition = {"transition_id": digest(core), **core}
    validate_evidence_transition(transition, tool_execution=tool_execution)
    return transition


def validate_evidence_transition(transition, *, tool_execution,
                                 expected=None):
    required = {"version", "transition_id", "question_id", "rollout_id",
                "browse_record_id", "tool_execution_id", "policy",
                "token_digest", "task_scope_digest", "before_snapshot",
                "after_snapshot", "before_evidence", "after_evidence",
                "before_source_headers", "after_source_headers", "change"}
    need(isinstance(transition, dict) and set(transition) == required and
         transition["version"] == EVIDENCE_TRANSITION_VERSION,
         "evidence transition fields")
    need(isinstance(tool_execution, dict),
         "trusted tool execution required for evidence transition")
    action, response_evidence = validate_tool_execution(tool_execution)
    need(action["tool"] == "browse", "evidence transition requires Browse")
    identity = {
        "question_id": tool_execution["question_id"],
        "rollout_id": tool_execution["rollout_id"],
        "browse_record_id": tool_execution["record_id"],
        "tool_execution_id": tool_execution["tool_execution_id"],
        "policy": tool_execution["policy"],
        "token_digest": tool_execution["token_digest"],
        "task_scope_digest": tool_execution["task_scope_digest"],
    }
    need(all(transition.get(key) == value for key, value in identity.items()),
         "evidence transition/execution binding mismatch")
    if expected is not None:
        need(all(transition.get(key) == value for key, value in expected.items()),
             "evidence transition/batch binding mismatch")
    before = canonical_evidence(transition["before_evidence"])
    after = canonical_evidence(transition["after_evidence"])
    need(transition["before_evidence"] == before and
         transition["after_evidence"] == after,
         "noncanonical evidence transition")
    before_headers = canonical_source_headers(
        transition["before_source_headers"],
        {row["source_id"] for row in before})
    after_headers = canonical_source_headers(
        transition["after_source_headers"],
        {row["source_id"] for row in after})
    need(transition["before_source_headers"] == before_headers and
         transition["after_source_headers"] == after_headers,
         "noncanonical evidence-transition source headers")
    for name, evidence in (("before", before), ("after", after)):
        snapshot = transition[name + "_snapshot"]
        need(isinstance(snapshot, dict) and set(snapshot) == {"step", "digest"}
             and type(snapshot["step"]) is int and snapshot["step"] >= 0 and
             snapshot["digest"] == digest(evidence),
             "evidence transition snapshot mismatch")
    need(transition["after_snapshot"]["step"] ==
         transition["before_snapshot"]["step"] + 1,
         "evidence transition steps must be adjacent")
    old = {row["source_id"]: row for row in before}
    new = {row["source_id"]: row for row in after}
    need(set(old) <= set(new) and all(old[key] == new[key] for key in old),
         "evidence ledger deleted or modified")
    added = sorted(set(new) - set(old))
    need(set(before_headers) <= set(after_headers) and
         all(before_headers[key] == after_headers[key]
             for key in before_headers),
         "source header removed or modified across evidence transition")
    need(set(after_headers) - set(before_headers) <= set(added),
         "source metadata update lacks newly appended evidence")
    returned = {row["source_id"]: row for row in response_evidence}
    if tool_execution["status"] == "succeeded":
        need(all(key in new and new[key] == value
                 for key, value in returned.items()),
             "successful Browse response absent from evidence ledger")
        need(all(key in returned and returned[key] == new[key] for key in added),
             "new ledger evidence absent from Browse response")
        expected_change = "append" if added else "no_change"
    else:
        need(not response_evidence and not added and before == after,
             "failed Browse must preserve evidence ledger")
        expected_change = "failed_no_change"
    if expected_change in {"no_change", "failed_no_change"}:
        need(before_headers == after_headers,
             "no-change Browse must preserve coverage metadata")
    need(transition["change"] == expected_change,
         "evidence transition change mismatch")
    core = copy.deepcopy(transition)
    core.pop("transition_id")
    need(transition["transition_id"] == digest(core),
         "evidence transition business identity")
    return {"change": expected_change, "new_evidence_ids": added,
            "before_snapshot": copy.deepcopy(transition["before_snapshot"]),
            "after_snapshot": copy.deepcopy(transition["after_snapshot"]),
            "before_evidence": before, "after_evidence": after,
            "before_source_headers": before_headers,
            "after_source_headers": after_headers}


def validate_violation_event(event, expected):
    required = {"version", "violation_event_id", "question_id", "rollout_id",
                "record_id", "tool_execution_id", "policy", "token_digest",
                "task_scope_digest", "violation_type"}
    need(isinstance(event, dict) and set(event) == required and
         event["version"] == VIOLATION_VERSION,
         "policy violation fields")
    need(all(event.get(key) == value for key, value in expected.items()),
         "policy violation binding mismatch")
    need(isinstance(event["violation_type"], str) and
         event["violation_type"].strip(), "policy violation type")
    core = copy.deepcopy(event)
    core.pop("violation_event_id")
    need(event["violation_event_id"] == digest(core),
         "policy violation business identity")
    return event


def task_scope(question, requirements, constraints, task_mode):
    need(isinstance(question, str) and question.strip(), "question required")
    need(task_mode in {"evidence_grounded", "self_contained"}, "task mode")
    need(isinstance(requirements, list) and requirements, "requirements required")
    ids = [row.get("id") for row in requirements if isinstance(row, dict)]
    need(len(ids) == len(requirements) and len(ids) == len(set(ids)) and
         all(isinstance(x, str) and x for x in ids), "requirement identities")
    return digest({"question": question, "requirements": requirements,
                   "constraints": constraints, "task_mode": task_mode})


def validate_tool_execution(execution):
    required = {"version", "tool_execution_id", "question_id", "rollout_id",
                "record_id", "policy", "token_digest", "task_scope_digest",
                "capture", "response", "status"}
    need(isinstance(execution, dict) and set(execution) == required and
         execution["version"] == TOOL_EXECUTION_VERSION,
         "tool execution fields")
    for key in ("question_id", "rollout_id", "record_id", "policy",
                "token_digest", "task_scope_digest"):
        need(isinstance(execution[key], str) and execution[key],
             "tool execution identity")
    need(execution["status"] in {"succeeded", "failed"}, "tool execution status")
    capture = execution["capture"]
    need(isinstance(capture, dict) and set(capture) == {
        "capture_id", "capture_sha256", "parser_version", "raw_completion",
        "action", "action_digest"}, "capture projection fields")
    need(capture["parser_version"] in PARSER_VERSIONS, "unknown action parser")
    action = canonical_action(capture["raw_completion"],
                              capture["parser_version"])
    need(capture["action"] == action and capture["action_digest"] == digest(action),
         "capture/action projection mismatch")
    response = execution["response"]
    need(isinstance(response, dict) and set(response) == {
        "payload", "payload_digest"}, "tool response fields")
    payload = response["payload"]
    need(isinstance(payload, dict) and set(payload) == {"tool", "evidence"} and
         payload["tool"] == action["tool"], "tool response/action mismatch")
    evidence = canonical_evidence(payload["evidence"])
    need(payload["evidence"] == evidence and response["payload_digest"] == digest(payload),
         "tool response digest/evidence mismatch")
    core = copy.deepcopy(execution)
    core.pop("tool_execution_id")
    need(execution["tool_execution_id"] == digest(core),
         "tool execution business identity mismatch")
    return action, evidence


def coverage_rows(view, result, coverage_values):
    normalized = canonical_coverage_view(view)
    need(view == normalized, "noncanonical coverage view")
    canonical = normalized["evidence"]
    evidence = {row["source_id"]: row for row in canonical}
    need(len(evidence) == len(view["evidence"]), "duplicate coverage evidence")
    requirements = {row["id"]: row for row in view["requirements"]}
    need(requirements and len(requirements) == len(view["requirements"]),
         "frozen coverage denominator")
    need(isinstance(result, dict) and set(result) == {"coverage"} and
         isinstance(result["coverage"], list), "coverage result fields")
    rows = {row.get("id"): row for row in result["coverage"]
            if isinstance(row, dict)}
    need(set(rows) == set(requirements) and len(rows) == len(result["coverage"]),
         "coverage denominator drift")
    total, weight, unobservable = 0.0, 0.0, False
    for rid, row in rows.items():
        need(set(row) == {"id", "status", "evidence_ids", "reason"} and
             row["status"] in STATUS_VALUES and
             isinstance(row["reason"], str) and row["reason"].strip(),
             "coverage row fields")
        refs = row["evidence_ids"]
        need(isinstance(refs, list) and len(refs) == len(set(refs)) and
             set(refs) <= set(evidence), "coverage evidence reference")
        if row["status"] in {"partial", "direct"}:
            need(bool(refs), "supported coverage needs evidence")
        else:
            need(not refs, "unknown coverage cannot cite evidence")
        if row["status"] == "unobservable":
            unobservable = True
            continue
        w = finite(requirements[rid].get("weight", 1))
        need(w > 0, "positive requirement weight")
        total += w * finite(coverage_values[row["status"]])
        weight += w
    return rows, (None if unobservable or weight == 0 else total / weight)


def validate_gain_assessment(artifact, *, scorer, reward_config_digest,
                             coverage_values, expected_binding=None,
                             coverage_receipt_lookup=None,
                             evidence_transition=None):
    """Recompute a completed Browse assessment, including zero/review states."""
    common = {"version", "scorer", "reward_config", "task_modes_digest",
              "task_scope_digest", "tool_execution", "binding", "before_view",
              "after_view", "before_result", "after_result", "before_key",
              "after_key", "before_score", "after_score", "transitions",
              "disposition", "delta"}
    version = artifact.get("version") if isinstance(artifact, dict) else None
    required = (common if version == LEGACY_GAIN_VERSION else
                common | {"delta_receipt", "contradictions"})
    need(isinstance(artifact, dict) and set(artifact) == required and
         version in {LEGACY_GAIN_VERSION, GAIN_VERSION},
         "evidence gain artifact fields")
    need(artifact["scorer"] == scorer and
         artifact["reward_config"] == reward_config_digest,
         "evidence gain scorer/reward drift")
    action, response_evidence = validate_tool_execution(artifact["tool_execution"])
    need(artifact["tool_execution"]["status"] == "succeeded",
         "failed tool execution cannot export normal evidence gain")
    need(action["tool"] == "browse" and artifact["task_scope_digest"] ==
         artifact["tool_execution"]["task_scope_digest"],
         "Browse gain execution/scope mismatch")
    binding = artifact["binding"]
    binding_fields = {"question_id", "rollout_id", "browse_record_id",
                      "after_record_id", "policy", "before_snapshot",
                      "after_snapshot", "selected_candidate_id",
                      "new_evidence_ids", "tool_execution_id",
                      "task_scope_digest"}
    need(isinstance(binding, dict) and set(binding) == binding_fields,
         "evidence gain binding fields")
    if expected_binding is not None:
        need(all(binding.get(key) == value for key, value in expected_binding.items()),
             "evidence gain batch binding mismatch")
    execution = artifact["tool_execution"]
    need(binding["tool_execution_id"] == execution["tool_execution_id"] and
         binding["question_id"] == execution["question_id"] and
         binding["rollout_id"] == execution["rollout_id"] and
         binding["browse_record_id"] == execution["record_id"] and
         binding["policy"] == execution["policy"] and
         binding["task_scope_digest"] == execution["task_scope_digest"],
         "gain/tool execution identity mismatch")
    need(binding["selected_candidate_id"] == action["arguments"]["candidate_id"],
         "gain selected candidate mismatch")
    before, after = artifact["before_view"], artifact["after_view"]
    for key in ("question", "requirements", "constraints"):
        need(before.get(key) == after.get(key), "evidence gain task drift")
    # task_scope_digest is externally frozen and compared to the batch registry;
    # the view itself is always evidence-grounded coverage.
    need(artifact["task_scope_digest"] == task_scope(
        before["question"], before["requirements"], before["constraints"],
        "evidence_grounded"), "gain task scope mismatch")
    old = {row["source_id"]: row for row in canonical_evidence(before["evidence"])}
    new = {row["source_id"]: row for row in canonical_evidence(after["evidence"])}
    need(set(old) <= set(new) and all(old[key] == new[key] for key in old),
         "evidence deleted or modified")
    old_headers = canonical_source_headers(before["source_headers"], set(old))
    new_headers = canonical_source_headers(after["source_headers"], set(new))
    need(set(old_headers) <= set(new_headers) and
         all(old_headers[key] == new_headers[key] for key in old_headers),
         "source header removed or modified")
    new_ids = sorted(set(new) - set(old))
    need(binding["new_evidence_ids"] == new_ids,
         "evidence gain new-evidence binding mismatch")
    response = {row["source_id"]: row for row in response_evidence}
    need(all(key in response and response[key] == new[key] for key in new_ids),
         "new evidence not present in bound tool response")
    if evidence_transition is not None:
        verified = validate_evidence_transition(
            evidence_transition, tool_execution=artifact["tool_execution"],
            expected={"question_id": binding["question_id"],
                      "rollout_id": binding["rollout_id"],
                      "browse_record_id": binding["browse_record_id"],
                      "tool_execution_id": binding["tool_execution_id"],
                      "policy": binding["policy"],
                      "task_scope_digest": binding["task_scope_digest"]})
        need(verified["before_snapshot"] == binding["before_snapshot"] and
             verified["after_snapshot"] == binding["after_snapshot"] and
             verified["before_evidence"] == canonical_evidence(before["evidence"]) and
             verified["after_evidence"] == canonical_evidence(after["evidence"]) and
             verified["before_source_headers"] ==
             canonical_source_headers(before["source_headers"], set(old)) and
             verified["after_source_headers"] ==
             canonical_source_headers(after["source_headers"], set(new)),
             "gain assessment/evidence ledger mismatch")
    need(binding["before_snapshot"] == {
        "step": binding["before_snapshot"].get("step"),
        "digest": digest(canonical_evidence(before["evidence"]))} and
         binding["after_snapshot"] == {
        "step": binding["after_snapshot"].get("step"),
        "digest": digest(canonical_evidence(after["evidence"]))} and
         type(binding["before_snapshot"]["step"]) is int and
         binding["after_snapshot"]["step"] == binding["before_snapshot"]["step"] + 1,
         "gain snapshots must be rehashed and adjacent")
    need(before == canonical_coverage_view(before) and
         after == canonical_coverage_view(after),
         "noncanonical evidence-gain coverage view")
    need(artifact["before_key"] == coverage_key(scorer, before) and
         artifact["after_key"] == coverage_key(scorer, after),
         "coverage scorer/input mismatch")
    before_rows, before_score = coverage_rows(before, artifact["before_result"], coverage_values)
    after_rows, after_score = coverage_rows(after, artifact["after_result"], coverage_values)
    need(callable(coverage_receipt_lookup),
         "trusted coverage receipt resolver required")
    receipts = {}
    for key, result in ((artifact["before_key"], artifact["before_result"]),
                        (artifact["after_key"], artifact["after_result"])):
        receipt = coverage_receipt_lookup(key)
        canonical_result = validate_coverage_receipt(
            receipt, scorer=scorer,
            task_scope_digest=artifact["task_scope_digest"], coverage_key=key)
        need(canonical_result == canonical_coverage_result(result),
             "coverage result conflicts with canonical receipt")
        receipts[key] = receipt
    if version == GAIN_VERSION:
        delta_receipt = artifact["delta_receipt"]
        checked_delta = validate_coverage_delta_receipt(
            delta_receipt, scorer=scorer,
            task_scope_digest=artifact["task_scope_digest"],
            delta_key=delta_receipt.get("delta_key"),
            previous_receipt_id=receipts[artifact["before_key"]]["receipt_id"])
        delta_view = checked_delta["view"]
        need(canonical_coverage_result({"coverage": delta_view["prior_coverage"]}) ==
             canonical_coverage_result(artifact["before_result"]) and
             delta_view["retained_evidence"] == canonical_evidence(before["evidence"]) and
             delta_view["new_evidence"] ==
             canonical_evidence([new[key] for key in new_ids]),
             "coverage delta ledger binding mismatch")
        merged, contradictions = merge_coverage_delta(
            delta_view, checked_delta["result"])
        need(merged == canonical_coverage_result(artifact["after_result"]),
             "coverage delta merge/result mismatch")
        need(contradictions == artifact["contradictions"],
             "coverage contradiction ledger mismatch")
    if artifact["before_key"] == artifact["after_key"]:
        need(canonical_coverage_result(artifact["before_result"]) ==
             canonical_coverage_result(artifact["after_result"]),
             "same evidence received conflicting coverage")
    for name, computed in (("before_score", before_score), ("after_score", after_score)):
        saved = artifact[name]
        need((computed is None and saved is None) or
             (computed is not None and saved is not None and
              abs(computed - finite(saved)) <= 1e-12), "coverage score mismatch")
    transitions = []
    positive, decreased = 0.0, False
    weights = {row["id"]: finite(row.get("weight", 1))
               for row in before["requirements"]}
    total_weight = sum(weights.values())
    need(total_weight > 0, "coverage weights")
    for rid in sorted(before_rows):
        a, b = before_rows[rid], after_rows[rid]
        av = None if a["status"] == "unobservable" else finite(coverage_values[a["status"]])
        bv = None if b["status"] == "unobservable" else finite(coverage_values[b["status"]])
        change = None if av is None or bv is None else bv - av
        added_refs = sorted(set(b["evidence_ids"]) & set(new_ids))
        if change is not None and change > 1e-12:
            need(added_refs, "coverage increase lacks newly opened evidence")
            positive += weights[rid] * change / total_weight
        decreased |= change is not None and change < -1e-12
        transitions.append({"id": rid, "before": a["status"], "after": b["status"],
                            "delta": change, "new_evidence_ids": added_refs})
    need(artifact["transitions"] == transitions, "coverage transition drift")
    if before_score is None or after_score is None:
        status, expected_delta = "unobservable", None
    elif decreased:
        status, expected_delta = "needs_review", None
    elif positive > 1e-12:
        need(abs((after_score - before_score) - positive) <= 1e-12,
             "evidence gain delta mismatch")
        status, expected_delta = "observed_positive", positive
    else:
        need(abs(after_score - before_score) <= 1e-12,
             "zero evidence gain score mismatch")
        status, expected_delta = "observed_zero", 0.0
    need(artifact["disposition"] == status,
         "gain assessment disposition mismatch")
    saved_delta = artifact["delta"]
    need((expected_delta is None and saved_delta is None) or
         (expected_delta is not None and saved_delta is not None and
          abs(finite(saved_delta) - expected_delta) <= 1e-12),
         "gain assessment delta mismatch")
    return {"status": status, "value": expected_delta,
            "before_key": artifact["before_key"],
            "after_key": artifact["after_key"]}


def validate_gain_artifact(artifact, *, scorer, reward_config_digest,
                           coverage_values, expected_binding=None,
                           coverage_receipt_lookup=None,
                           evidence_transition=None):
    """Recompute an exported positive gain; returns its authorized scalar."""
    assessment = validate_gain_assessment(
        artifact, scorer=scorer, reward_config_digest=reward_config_digest,
        coverage_values=coverage_values, expected_binding=expected_binding,
        coverage_receipt_lookup=coverage_receipt_lookup,
        evidence_transition=evidence_transition)
    need(assessment["status"] == "observed_positive",
         "only observed-positive assessment can export gain")
    return assessment["value"]
