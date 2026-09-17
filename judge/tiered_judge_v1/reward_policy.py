"""Stage score policy only. No collector bypass, synthetic labels, or optimizer."""
import hashlib
import json
import math
from pathlib import Path

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
POLICY=json.loads(Path(__file__).with_name('policy.json').read_text())

def scalar(stage,core,auxiliary=None,*,material_core_error=False,excluded=(),contract_valid=True,policy=POLICY):
    weights=policy['stages'][stage]['core_weights'];labels=policy['auxiliary_labels'];cap=policy['auxiliary_adjustment_max']
    if set(core)!=set(weights):raise ValueError('exact core dimensions required')
    if type(cap) not in (int,float) or not math.isfinite(cap) or not 0<=cap<=.05 or any(type(w) not in (int,float) or not math.isfinite(w) or w<=0 for w in weights.values()):raise ValueError('invalid frozen weights')
    if labels!={'poor':-1,'adequate':0,'good':1}:raise ValueError('invalid auxiliary scale')
    if abs(sum(weights.values())-1)>1e-9:raise ValueError('weights must sum to one')
    excluded=set(excluded)
    # Exclusion is a frozen task-mode decision, never inferred from Judge nulls.
    if excluded and not (stage=='final' and excluded=={'citation_support'}):raise ValueError('unsupported structural exclusion')
    for value in core.values():
        if value is not None and (type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1):raise ValueError('core score range')
    active={k:w for k,w in weights.items() if k not in excluded}
    missing=[k for k in active if core[k] is None]
    warnings=[];adjust=0.;aux_status='not_requested'
    if policy['stages'][stage]['auxiliary'] is not None:
        if isinstance(auxiliary,str) and auxiliary in labels:
            adjust=cap*labels[auxiliary];aux_status='observed_coarse'
        else:
            aux_status='omitted';warnings.append('auxiliary_unavailable_no_adjustment')
    if material_core_error and adjust>0:adjust=0.;warnings.append('positive_auxiliary_blocked_by_core_error')
    value=None;core_value=None
    if contract_valid and not missing:
        core_value=sum(core[k]*w for k,w in active.items())/sum(active.values())
        value=max(0.,min(1.,core_value+adjust))
    return dict(stage=stage,policy_digest=digest(policy),core=core,structurally_excluded=sorted(excluded),
        core_score=core_value,auxiliary_status=aux_status,auxiliary_adjustment=adjust,
        score=value,status='observed' if value is not None else 'pending_core',
        missing_core=missing,warnings=warnings,contract_valid=contract_valid,
        automatic_extra_limitation_required=False,training_ready=False)
