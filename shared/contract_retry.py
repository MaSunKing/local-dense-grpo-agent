"""One bounded retry for malformed Judge output; never for semantic scores."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

CONTRACT_RETRY_VERSION = "judge_contract_retry_once_v2"


def _atomic_text(path, text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=".writing-",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as stream:
            stream.write(text);stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def contract_error_code(error):
    if isinstance(error,json.JSONDecodeError):return "invalid_json"
    text=str(error).lower()
    rules=(
        (("field","schema","top-level","row count","denominator"),"invalid_fields"),
        (("enum","label","grade","status","boolean or null"),"invalid_value"),
        (("reference","evidence","item binding"),"invalid_reference"),
        (("cannot exist","requires","conflict"),"logical_conflict"),
    )
    for needles,code in rules:
        if any(needle in text for needle in needles):return code
    return "contract_rejected"


def _error_detail(error):
    text=" ".join(str(error).split()).strip()
    return (text or type(error).__name__)[:500]


def correction_message(code, detail):
    return (
        "The previous response was rejected by the deterministic output validator. "
        f"Error category: {code}. Validator reason: {detail}. "
        "Return one corrected JSON object only, using exactly "
        "the schema and allowed IDs stated in the system message. Do not add fields, "
        "guess missing evidence, or change the substantive judgment merely to obtain "
        "a higher score."
    )


def run_contract_retry(*,request,call,validate,audit_dir):
    """Call once, retry once only after a returned response fails validation."""
    base=copy.deepcopy(request);audit=Path(audit_dir);attempts=[]
    for number in (1,2):
        current=copy.deepcopy(base)
        if number==2:
            code=attempts[-1]["error_code"];detail=attempts[-1]["validator_error"]
            current["messages"]=copy.deepcopy(base["messages"])+[
                {"role":"user","content":correction_message(code,detail)}]
        request_digest=hashlib.sha256(json.dumps(
            current,sort_keys=True,ensure_ascii=False,separators=(",",":"),
        ).encode()).hexdigest()
        try:
            raw=call(current)
        except (OSError,TimeoutError) as error:
            row={"attempt":number,"status":"transport_failed",
                 "error_code":type(error).__name__,"validator_error":_error_detail(error),
                 "request_digest":request_digest}
            attempts.append(row)
            _atomic_text(audit/f"attempt-{number}"/"receipt.json",
                         json.dumps(row,ensure_ascii=False,indent=2)+"\n")
            return {"status":"pending","result":None,"attempts":attempts,
                    "contract_retries":number-1,"training_ready":False}
        except (ValueError,TypeError,KeyError) as error:
            code=contract_error_code(error)
            row={"attempt":number,"status":"contract_rejected","error_code":code,
                 "validator_error":_error_detail(error),"request_digest":request_digest}
            attempts.append(row)
            _atomic_text(audit/f"attempt-{number}"/"receipt.json",
                         json.dumps(row,ensure_ascii=False,indent=2)+"\n")
            if number==1:continue
            return {"status":"pending","result":None,"attempts":attempts,
                    "contract_retries":1,"training_ready":False}
        if not isinstance(raw,str):raw_type=type(raw).__name__;raw="";error=TypeError("response text required: "+raw_type)
        else:error=None
        _atomic_text(audit/f"attempt-{number}"/"response.raw",raw)
        response_digest=hashlib.sha256(raw.encode()).hexdigest()
        try:
            if error is not None:raise error
            result=json.loads(raw)
            validate(result)
        except (ValueError,TypeError,KeyError) as exc:
            code=contract_error_code(exc)
            row={"attempt":number,"status":"contract_rejected","error_code":code,
                 "validator_error":_error_detail(exc),"request_digest":request_digest,
                 "response_sha256":response_digest}
            attempts.append(row)
            _atomic_text(audit/f"attempt-{number}"/"receipt.json",
                         json.dumps(row,ensure_ascii=False,indent=2)+"\n")
            if number==1:continue
            return {"status":"pending","result":None,"attempts":attempts,
                    "contract_retries":1,"training_ready":False}
        row={"attempt":number,"status":"validated","request_digest":request_digest,
             "response_sha256":response_digest}
        attempts.append(row)
        _atomic_text(audit/f"attempt-{number}"/"receipt.json",
                     json.dumps(row,ensure_ascii=False,indent=2)+"\n")
        return {"status":"validated","result":result,"attempts":attempts,
                "contract_retries":number-1,"training_ready":False}
    raise AssertionError("unreachable")
