
import json
import math

def parse_response(text):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def bad_constant(value):
        raise ValueError("nonfinite JSON constant")
    return json.loads(text, object_pairs_hook=unique_pairs,
                      parse_constant=bad_constant)

def validate_response(payload, result):
    if not isinstance(result, dict) or set(result) != {"results"}:
        raise ValueError("invalid top-level schema")
    rows = result["results"]
    wanted = {r["record_id"]: r for r in payload["records"]}
    if not isinstance(rows, list) or len(rows) != len(wanted):
        raise ValueError("record count mismatch")
    seen = set()
    requirement_ids = {r["id"] for r in payload["requirements"]}

    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "record_id", "scores", "coverage", "reason"
        }:
            raise ValueError("invalid result fields")
        rid = row["record_id"]
        if not isinstance(rid, str) or rid not in wanted or rid in seen:
            raise ValueError("unknown or duplicate record")
        seen.add(rid)
        record = wanted[rid]
        scores = row["scores"]
        if not isinstance(scores, dict) or set(scores) != set(
            record["requested_dimensions"]
        ):
            raise ValueError("score dimensions mismatch")
        for value in scores.values():
            if value is not None and (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError("invalid score")
        if not isinstance(row["reason"], str):
            raise ValueError("invalid reason")
        coverage = row["coverage"]
        if not isinstance(coverage, list):
            raise ValueError("invalid coverage")
        if not record["coverage_requested"]:
            if coverage:
                raise ValueError("unexpected coverage")
            continue
        if len(coverage) != len(requirement_ids):
            raise ValueError("coverage denominator mismatch")
        available = {c["source_id"] for c in record["evidence"]}
        covered = set()
        for item in coverage:
            if not isinstance(item, dict) or set(item) != {
                "id", "status", "evidence_ids"
            }:
                raise ValueError("invalid coverage row")
            qid = item["id"]
            if not isinstance(qid, str) or qid not in requirement_ids or qid in covered:
                raise ValueError("unknown or duplicate requirement")
            covered.add(qid)
            status = item["status"]
            if status not in (None, "unknown", "partial", "direct"):
                raise ValueError("invalid coverage status")
            refs = item["evidence_ids"]
            if not isinstance(refs, list) or any(
                not isinstance(s, str) or s not in available for s in refs
            ):
                raise ValueError("citation outside this step's opened evidence")
            if len(set(refs)) != len(refs):
                raise ValueError("duplicate citation")
            if status in ("direct", "partial") and not refs:
                raise ValueError("supported coverage without evidence")
            if status in (None, "unknown") and refs:
                raise ValueError("unknown coverage with asserted support")
    return result
