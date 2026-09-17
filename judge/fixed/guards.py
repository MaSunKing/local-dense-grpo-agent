"""Conservative attribution boundaries; not a GRPO batch compiler."""
import hashlib
import json
import math
import re


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def task_scope_digest(payload):
    return digest({k: payload[k] for k in
                   ("question", "requirements", "constraints")})


def evidence_key(payload, scorer):
    """Equal evidence/rubric scope shares an identity, independent of stage output."""
    record = payload['records'][0]
    chunks = record['evidence']
    ids = [c['source_id'] for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate evidence ID')
    return digest(dict(question=payload['question'],
        requirements=payload['requirements'], constraints=payload['constraints'],
        scorer=scorer, evidence=sorted(chunks, key=lambda c:c['source_id']),
        source_headers=record['source_headers']))


def _chunks(chunks):
    if not isinstance(chunks, list):
        raise ValueError('evidence list required')
    found = {}
    for c in chunks:
        if not isinstance(c, dict) or not isinstance(c.get('source_id'), str) or not c['source_id'] or not isinstance(c.get('text'), str):
            raise ValueError('invalid evidence chunk')
        if c['source_id'] in found:
            raise ValueError('duplicate evidence ID')
        found[c['source_id']] = digest(c)
    return found


def verify_stop(payload, collector_lookup=None):
    """collector_lookup is a trusted integration dependency, NEVER a Judge field.

    It resolves an immutable event from the collector ledger and its bound capture.
    This library does not establish authenticity of caller-supplied registries.
    No resolver => no provenance; self-declared verified flags cannot enable reward.
    """
    record = payload['records'][0]
    meta = record.get('runtime_termination', {})
    if meta == {'verified': False, 'type': 'unverified'} or not meta:
        return None
    if collector_lookup is None or not callable(collector_lookup):
        raise ValueError('trusted collector resolver required')
    if set(meta) != {'event_id', 'event_digest', 'type'}:
        raise ValueError('Stop requires ledger reference, not verified/type flags')
    event, capture = collector_lookup(meta['event_id'])
    if meta['type'] != event['type']:
        raise ValueError('termination type not bound to ledger')
    if digest(event) != meta['event_digest'] or event['event_id'] != meta['event_id']:
        raise ValueError('collector event mismatch')
    binding = event['binding']
    if set(binding) != {'question_id','rollout_id','record_id','token_digest','policy','evidence_snapshot','task_scope_digest'}:
        raise ValueError('complete Stop binding required')
    if binding['task_scope_digest'] != task_scope_digest(payload):
        raise ValueError('Stop full task scope mismatch')
    for key in ('question_id','rollout_id','policy'):
        if not isinstance(payload.get(key),str) or not payload[key] or binding[key] != payload[key]:
            raise ValueError('Stop identity mismatch: '+key)
    if binding['record_id'] != record['record_id']:
        raise ValueError('Stop record mismatch')
    for key in ('input_ids','output_ids'):
        if not isinstance(capture.get(key),list) or not capture[key] or any(type(x) is not int or x<0 for x in capture[key]):
            raise ValueError('actual captured tokens required')
    if digest([capture['input_ids'],capture['output_ids']]) != binding['token_digest']:
        raise ValueError('Stop token mismatch')
    for key in ('question_id','rollout_id','record_id','policy'):
        if capture.get(key) != binding[key]:
            raise ValueError('capture identity mismatch')
    for obj, hash_key in ((capture, 'source_sha256'), (record, 'capture_sha256')):
        if not isinstance(obj.get('capture_id'), str) or not obj['capture_id'].strip():
            raise ValueError('nonempty capture_id required')
        if not isinstance(obj.get('raw_completion'), str):
            raise ValueError('raw_completion must be a string')
        value = obj.get(hash_key)
        if not isinstance(value, str) or re.fullmatch(r'[0-9a-fA-F]{64}', value) is None:
            raise ValueError('valid capture SHA256 required')
    if capture.get('capture_id') != record['capture_id'] or capture.get('raw_completion') != record['raw_completion']:
        raise ValueError('actual output binding mismatch')
    # capture_sha256 is an externally verified raw-file hash supplied by the resolver.
    if capture.get('source_sha256') != record['capture_sha256']:
        raise ValueError('capture file binding mismatch')
    snap = record['evidence_snapshot']
    _chunks(record['evidence'])
    if type(snap.get('step')) is not int or snap['step']<0 or snap.get('digest') != digest(record['evidence']):
        raise ValueError('invalid Stop snapshot')
    if binding['evidence_snapshot'] != snap or event['evidence'] != record['evidence']:
        raise ValueError('Stop evidence snapshot mismatch')
    if event['source_headers'] != record['source_headers']:
        raise ValueError('Stop source metadata mismatch')
    ctx = event['stop_context']
    required = {'remaining_budget','available_candidates','failed_reads','task_constraints','voluntary'}
    if set(ctx) != required or type(ctx['remaining_budget']) is not int or ctx['remaining_budget']<0:
        raise ValueError('complete Stop opportunity context required')
    if any(not isinstance(ctx[k],list) for k in required-{'remaining_budget','voluntary'}):
        raise ValueError('invalid Stop opportunity lists')
    if type(ctx['voluntary']) is not bool:
        raise ValueError('invalid voluntary flag')
    if event['type'] not in {'active','quota_exhausted','forced','error','cancelled'}:
        raise ValueError('invalid termination type')
    if ctx['voluntary'] != (event['type']=='active'):
        raise ValueError('termination type contradicts collector')
    if event['type']=='active' and ctx['remaining_budget']==0:
        raise ValueError('zero-budget stop is not voluntary timing')
    if event['type']=='quota_exhausted' and ctx['remaining_budget']!=0:
        raise ValueError('quota termination contradicts remaining budget')
    if record.get('stop_context') != ctx:
        raise ValueError('Judge context does not match collector')
    return event['type']


def stop_local_score(payload, result, collector_lookup=None):
    kind = verify_stop(payload, collector_lookup)
    if kind != 'active':
        if result['scores'].get('stop_appropriateness') is not None:
            raise ValueError('unverified/forced stop cannot receive timing score')
        return None
    value = result['scores']['stop_appropriateness']
    if value is not None and (type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1):
        raise ValueError('invalid timing score')
    return value


def validate_result(payload, result, upstream, collector_lookup=None, task_modes=None):
    mode = payload.get('evaluation_task_mode','evidence_grounded')
    if mode not in {'self_contained','evidence_grounded'}:
        raise ValueError('invalid task mode')
    if mode == 'self_contained':
        frozen = (task_modes or {}).get(payload['question_id'])
        expected = dict(mode=mode,requirements_digest=digest(payload['requirements']),
                        question_digest=digest(payload['question']),
                        constraints_digest=digest(payload['constraints']))
        if frozen != expected:
            raise ValueError('self-contained exception needs frozen question configuration')
    rec = payload['records'][0]
    if rec['kind']=='final' and mode=='self_contained':
        if rec.get('coverage_requested') is not False:
            raise ValueError('self-contained Final must disable external evidence coverage')
        # Existing validator may hardcode coverage for every non-checklist task.
        # This narrow branch validates the full unchanged four-dimensional contract.
        if not isinstance(result,dict) or set(result)!={'results'} or not isinstance(result['results'],list) or len(result['results'])!=1:
            raise ValueError('invalid result envelope')
        r=result['results'][0]
        if not isinstance(r,dict) or set(r)!={'record_id','scores','coverage','reason'} or r['record_id']!=rec['record_id']:
            raise ValueError('invalid Final result identity')
        dims={'completeness','fidelity','citation_support','uncertainty'}
        if set(rec['requested_dimensions'])!=dims or not isinstance(r['scores'],dict) or set(r['scores'])!=dims:
            raise ValueError('invalid Final dimensions')
        if r['coverage']!=[] or not isinstance(r['reason'],str):
            raise ValueError('invalid self-contained coverage/reason')
        for v in r['scores'].values():
            if v is not None and (type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1):
                raise ValueError('invalid score')
    else:
        if rec['kind']=='final' and rec.get('coverage_requested') is not True:
            raise ValueError('cannot disable grounded Final coverage')
        upstream(payload, result)
    row = result['results'][0]
    if rec['kind'] == 'stop':
        stop_local_score(payload, row, collector_lookup)
    return result


def coverage_gain(before, after, new_evidence_ids):
    """Only canonical evidence evaluations, never answer/state-score differences.

    Caller must supply separately reviewed, same-scorer canonical coverage. A nonzero
    result is still NOT credited to an action until the real tool ledger binds it.
    """
    manifests = []
    for row in (before, after):
        if row['scope'] != 'evidence' or row.get('canonical_reviewed') is not True:
            raise ValueError('canonical evidence coverage required; not Final coverage')
        manifest = _chunks(row['evidence'])
        if not isinstance(row.get('source_headers'),dict):
            raise ValueError('canonical source headers must be keyed by header ID')
        manifests.append(manifest)
        if len(row['evidence_ids']) != len(set(row['evidence_ids'])) or set(row['evidence_ids']) != set(manifest):
            raise ValueError('evidence IDs do not match actual chunks')
        if row['evidence_key'] != digest({'chunks':manifest,'source_headers':row['source_headers']}):
            raise ValueError('snapshot identity mismatch')
    for key in ('scorer', 'requirements_digest', 'question_id'):
        if before[key] != after[key]:
            raise ValueError('mixed scoring identities')
    values = {'unknown': 0., 'partial': .5, 'direct': 1.}
    for row in (before, after):
        if any(v is not None and v not in values for v in row['coverage'].values()):
            raise ValueError('invalid coverage label')
    if before['coverage'].keys() != after['coverage'].keys():
        raise ValueError('requirements changed')
    old, new = manifests
    if not set(old) <= set(new):
        raise ValueError('removed evidence')
    if any(old[k] != new[k] for k in old):
        raise ValueError('existing evidence content/metadata changed')
    if any(k not in after['source_headers'] or after['source_headers'][k] != v for k,v in before['source_headers'].items()):
        raise ValueError('source header removed or modified')
    if not isinstance(new_evidence_ids,list) or len(new_evidence_ids)!=len(set(new_evidence_ids)) or set(new_evidence_ids) != set(new)-set(old):
        raise ValueError('unbound additions')
    if before['evidence_key'] == after['evidence_key']:
        if before['coverage'] != after['coverage']:
            raise ValueError('conflicting evaluations of identical evidence')
        return 0.
    if not new_evidence_ids:
        return None  # reinterpretation is not retrieval gain
    if any(v is None for r in (before, after) for v in r['coverage'].values()):
        return None
    return sum(values[after['coverage'][q]] - values[before['coverage'][q]]
               for q in before['coverage']) / max(1, len(before['coverage']))
