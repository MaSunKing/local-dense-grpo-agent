"""Verify legacy Search executions against original collector tool logs.

Search returns candidates, not opened evidence. Its evidence projection is
therefore empty; the complete unmodified tool log remains separately audited.
This does not create coverage gains or semantic Search scores.
"""
import copy

from .common import atomic_json, file_sha256, read_json, safe_relative
from .pipeline import release_imports


def verified_search_executions(package, run_dir):
    fixture = package / "fixture"
    items = fixture / "direct_records/records"
    index = read_json(fixture / "provenance/source_binding_index.json")
    grouped = {}
    for entry in index["bindings"]:
        binding = read_json(items / entry["item"] / "source_binding.json")
        if binding["canonical_kind"] in {"search", "browse"}:
            grouped.setdefault(binding["rollout_id"], []).append((binding, entry))
    executions, audit = {}, []
    with release_imports(package / "release"):
        from gain_contract import XML_PARSER_VERSION, canonical_action, digest, task_scope, validate_tool_execution
        for rollout_id, rows in sorted(grouped.items()):
            rows.sort(key=lambda pair: pair[0]["stage_row_index"])
            number = rollout_id.rsplit("rollout-", 1)[1]
            path = fixture / "collector_tool_calls" / f"rollout-{number}.json"
            calls = read_json(path)
            if len(calls) != len(rows):
                raise ValueError("collector tool log/decision count mismatch")
            for tool_index, (call, (binding, entry)) in enumerate(zip(calls, rows)):
                ref = next(row for row in entry["referenced_files"] if row["field"] == "capture_file")
                capture_path = safe_relative(fixture, ref["bundle_path"])
                if file_sha256(capture_path) != ref["sha256"]:
                    raise ValueError("Search capture hash mismatch")
                capture = read_json(capture_path)
                action = canonical_action(capture["raw_completion"], XML_PARSER_VERSION)
                if action["tool"] != binding["canonical_kind"] or call["operation"] != action["tool"]:
                    raise ValueError("collector tool operation/capture mismatch")
                if action["tool"] != "search":
                    continue
                if call["arguments"].get("query") != action["arguments"]["query"]:
                    raise ValueError("collector executed Search query/capture mismatch")
                payload = read_json(items / entry["item"] / "canonical.payload.json")
                if call["arguments"].get("question") != payload["question"]:
                    raise ValueError("collector executed Search question mismatch")
                if call["response"].get("status") != "ok":
                    raise ValueError("Search failure requires a separately defined execution branch")
                response_projection = {"tool": "search", "evidence": []}
                core = {
                    "version": "trusted_tool_execution_v1", "question_id": binding["question_id"],
                    "rollout_id": rollout_id, "record_id": binding["record_id"],
                    "policy": capture["policy"], "token_digest": capture["token_digest"],
                    "task_scope_digest": task_scope(payload["question"], payload["requirements"],
                                                     payload["constraints"], "evidence_grounded"),
                    "capture": {"capture_id": capture["capture_id"], "capture_sha256": ref["sha256"],
                                "parser_version": XML_PARSER_VERSION,
                                "raw_completion": capture["raw_completion"], "action": action,
                                "action_digest": digest(action)},
                    "response": {"payload": response_projection,
                                 "payload_digest": digest(response_projection)}, "status": "succeeded"}
                execution = {**core, "tool_execution_id": digest(core)}
                validate_tool_execution(execution)
                executions[execution["tool_execution_id"]] = execution
                audit.append({"item": entry["item"], "tool_execution_id": execution["tool_execution_id"],
                              "collector_tool_log_sha256": file_sha256(path), "tool_row_index": tool_index,
                              "collector_response_digest": digest(call["response"]),
                              "evidence_projection": "Search candidates are not opened evidence"})
    if len(executions) != 5:
        raise ValueError("real four-rollout fixture must have five verified Search executions")
    atomic_json(run_dir / "VERIFIED_SEARCH_EXECUTIONS.json", {"executions": executions, "audit": audit})
    return copy.deepcopy(executions)
