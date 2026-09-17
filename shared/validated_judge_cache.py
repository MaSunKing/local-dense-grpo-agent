"""Exact validated-result cache for post-trajectory Judge requests."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

from contract_retry import run_contract_retry

CACHE_VERSION = "validated_judge_result_cache_v1"
RECEIPT_VERSION = "validated_judge_run_receipt_v2"


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def identity(*, scorer, task, request):
    if not isinstance(scorer, str) or not scorer.strip():
        raise ValueError("cache scorer required")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("cache task required")
    if not isinstance(request, dict):
        raise ValueError("cache request required")
    core = {"version": CACHE_VERSION, "scorer": scorer, "task": task,
            "request_digest": _digest(request)}
    return _digest(core), core


def _path(cache_dir, cache_key):
    return Path(cache_dir).resolve() / cache_key[:2] / (cache_key + ".json")


def _atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    body=_canonical(value)+b"\n"
    fd,temporary=tempfile.mkstemp(prefix=".writing-",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as stream:
            stream.write(body);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def _write_run_receipt(*,audit_dir,scorer,task,request,outcome):
    cache_key,core=identity(scorer=scorer,task=task,request=request)
    attempts=copy.deepcopy(outcome.get("attempts",[]))
    rejected=[row for row in attempts if row.get("status")=="contract_rejected"]
    validated=[row for row in attempts if row.get("status")=="validated"]
    receipt={
        "version":RECEIPT_VERSION,"status":outcome["status"],"scorer":scorer,
        "task":task,"request_identity":core,
        "cache_key":outcome.get("cache_key") or cache_key,
        "cache_hit":bool(outcome.get("cache_hit")),
        "attempt_count":len(attempts),"attempts":attempts,
        "first_invalid_response_sha256":(
            rejected[0].get("response_sha256") if rejected else None),
        "validator_error":(rejected[0].get("validator_error") if rejected else None),
        "final_valid_response_sha256":(
            validated[-1].get("response_sha256") if validated else None),
        "result_digest":(_digest(outcome["result"])
                         if outcome.get("result") is not None else None),
        "training_ready":False,
    }
    _atomic_json(Path(audit_dir)/"final-receipt.json",receipt)


def load_validated(*, cache_dir, scorer, task, request, validate):
    if not callable(validate):
        raise ValueError("cache validator required")
    cache_key, core = identity(scorer=scorer, task=task, request=request)
    path = _path(cache_dir, cache_key)
    if not path.is_file():
        return None
    raw = path.read_bytes()
    entry = json.loads(raw)
    required = {"version", "cache_key", "identity", "result",
                "result_digest", "validated"}
    if not isinstance(entry, dict) or set(entry) != required:
        raise ValueError("cached Judge entry fields")
    if (entry["version"] != CACHE_VERSION or entry["cache_key"] != cache_key
            or entry["identity"] != core or entry["validated"] is not True
            or entry["result_digest"] != _digest(entry["result"])):
        raise ValueError("cached Judge entry binding mismatch")
    validate(copy.deepcopy(entry["result"]))
    return {"status": "validated", "result": copy.deepcopy(entry["result"]),
            "cache_hit": True, "cache_key": cache_key,
            "api_calls": 0, "contract_retries": 0}


def store_validated(*, cache_dir, scorer, task, request, result, validate):
    if not callable(validate):
        raise ValueError("cache validator required")
    validate(copy.deepcopy(result))
    cache_key, core = identity(scorer=scorer, task=task, request=request)
    entry = {"version": CACHE_VERSION, "cache_key": cache_key,
             "identity": core, "result": copy.deepcopy(result),
             "result_digest": _digest(result), "validated": True}
    body = _canonical(entry) + b"\n"
    path = _path(cache_dir, cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        existing = load_validated(cache_dir=cache_dir, scorer=scorer,
                                  task=task, request=request, validate=validate)
        if _digest(existing["result"]) != _digest(result):
            raise ValueError("validated cache conflict for identical Judge input")
        return existing
    fd, temporary = tempfile.mkstemp(prefix=cache_key + ".", suffix=".tmp",
                                     dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(body); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"status": "validated", "result": copy.deepcopy(result),
            "cache_hit": False, "cache_key": cache_key,
            "api_calls": 0, "contract_retries": 0}


def run_cached(*, cache_dir, scorer, task, request, call, validate, audit_dir):
    hit = load_validated(cache_dir=cache_dir, scorer=scorer, task=task,
                         request=request, validate=validate)
    if hit is not None:
        _write_run_receipt(audit_dir=audit_dir,scorer=scorer,task=task,
                           request=request,outcome=hit)
        return hit
    outcome = run_contract_retry(request=request, call=call,
                                 validate=validate, audit_dir=audit_dir)
    if outcome["status"] != "validated":
        completed=outcome | {"cache_hit": False, "cache_key": None}
        _write_run_receipt(audit_dir=audit_dir,scorer=scorer,task=task,
                           request=request,outcome=completed)
        return completed
    stored = store_validated(cache_dir=cache_dir, scorer=scorer, task=task,
                             request=request, result=outcome["result"],
                             validate=validate)
    completed=outcome | {"cache_hit": False,"cache_key":stored["cache_key"]}
    _write_run_receipt(audit_dir=audit_dir,scorer=scorer,task=task,
                       request=request,outcome=completed)
    return completed
