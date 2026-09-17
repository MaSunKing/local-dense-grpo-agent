"""Explicit semantic fields and caller-owned frozen task modes. No field inference."""
import copy
from guards import task_scope_digest

METADATA = {'title', 'url', 'source_type', 'published_at'}

def need(ok, message):
    if not ok:
        raise ValueError(message)


SOURCE_CONTEXT_STRINGS = {
    'authority_type', 'fetch_method', 'title', 'url',
    'content_level', 'evidence_scope', 'journal',
    'publication_date', 'source_format', 'year'
}
SOURCE_CONTEXT_BOOLS = {
    'is_human', 'abstract_only', 'full_text_available'
}

def canonical_source_context(value):
    allowed = SOURCE_CONTEXT_STRINGS | SOURCE_CONTEXT_BOOLS | {'publication_types'}
    need(isinstance(value, dict) and set(value) <= allowed,
         'unknown source context field')
    result = {}
    for key, item in value.items():
        if key in SOURCE_CONTEXT_STRINGS:
            need(isinstance(item, str), 'source context string required')
        elif key in SOURCE_CONTEXT_BOOLS:
            need(type(item) is bool, 'source context boolean required')
        else:
            need(isinstance(item, list) and
                 all(isinstance(x, str) for x in item),
                 'publication_types must be a string list')
        result[key] = copy.deepcopy(item)
    return result


def validate_inputs(payload):
    for r in payload['requirements']:
        need(isinstance(r, dict) and set(r) <= {'id', 'description', 'weight'}, 'unknown requirement field')
    constraints = payload['constraints']
    need(constraints == {} or isinstance(constraints, list) and all(isinstance(x, str) for x in constraints), 'constraints must be text list or empty object')
    for rec in payload['records']:
        evidence = rec['evidence']
        need(isinstance(evidence, list), 'evidence list required')
        ids = set()
        for e in evidence:
            need(isinstance(e, dict) and {'source_id', 'text'} <= set(e) <= {'source_id', 'text', 'source_context'} | METADATA, 'unknown evidence field')
            need(all(isinstance(v, str) for k, v in e.items() if k != 'source_context'),
                 'evidence values must be strings')
            if 'source_context' in e:
                need(rec['kind'] == 'final', 'source context currently supported for Final only')
                canonical_source_context(e['source_context'])
            need(e['source_id'].strip() and e['text'].strip() and e['source_id'] not in ids, 'invalid evidence identity/text')
            ids.add(e['source_id'])
        headers = rec['source_headers']
        need(isinstance(headers, dict), 'source headers object required')
        for sid, meta in headers.items():
            need(sid in ids and isinstance(meta, dict) and set(meta) <= METADATA, 'unknown header source/field')
            need(all(isinstance(v, str) for v in meta.values()), 'header values must be strings')

def task_mode(payload, task_modes=None):
    declared = payload.get('evaluation_task_mode', 'evidence_grounded')
    need(declared in {'evidence_grounded', 'self_contained'}, 'invalid declared mode')
    modes = task_modes or {}
    need(isinstance(modes, dict), 'caller frozen mode registry required')
    entry = modes.get(payload.get('question_id'))
    if entry is None:
        need(declared != 'self_contained', 'self-contained mode needs caller frozen registry')
        return 'evidence_grounded'
    need(isinstance(entry, dict) and set(entry) == {'mode', 'task_scope_digest'}, 'frozen mode entry fields')
    need(entry['mode'] in {'evidence_grounded', 'self_contained'}, 'invalid frozen mode')
    need(entry['task_scope_digest'] == task_scope_digest(payload), 'frozen task scope drift')
    if 'evaluation_task_mode' in payload:
        need(declared == entry['mode'], 'declared/frozen mode conflict')
    return entry['mode']

def semantic_evidence(rec):
    # Original dictionaries remain in the bound payload; only explicit fields enter views.
    return sorted([{k: copy.deepcopy(v) for k, v in e.items() if k in {'source_id', 'text', 'source_context'} | METADATA}
                   for e in rec['evidence']], key=lambda e: e['source_id'])


def canonical_stop_context(ctx):
    required = {'remaining_budget','voluntary','available_candidates','failed_reads','task_constraints'}
    need(isinstance(ctx,dict) and set(ctx)==required, 'Stop context fields')
    need(type(ctx['remaining_budget']) is int and ctx['remaining_budget']>=0, 'Stop budget')
    need(type(ctx['voluntary']) is bool, 'Stop voluntary flag')
    for key in ('available_candidates','failed_reads','task_constraints'):
        need(isinstance(ctx[key],list), 'Stop list required')

    candidates=[]
    seen=set()
    for row in ctx['available_candidates']:
        allowed={'id','title','snippet','url','source_type'}
        need(isinstance(row,dict) and {'id','title'}<=set(row)<=allowed, 'Stop candidate fields')
        need(all(isinstance(v,str) for v in row.values()), 'Stop candidate strings')
        need(row['id'].strip() and row['title'].strip() and row['id'] not in seen, 'Stop candidate identity')
        seen.add(row['id'])
        candidates.append({k:row[k] for k in sorted(allowed) if k in row})

    failures=[]
    for row in ctx['failed_reads']:
        need(isinstance(row,dict) and set(row)=={'candidate_id','error_type'}, 'Stop failed-read fields')
        need(all(isinstance(v,str) and v.strip() for v in row.values()), 'Stop failed-read strings')
        failures.append(dict(candidate_id=row['candidate_id'],error_type=row['error_type']))

    need(all(isinstance(v,str) and v.strip() for v in ctx['task_constraints']), 'Stop constraint strings')
    return dict(
        remaining_budget=ctx['remaining_budget'],
        voluntary=ctx['voluntary'],
        available_candidates=candidates,
        failed_reads=failures,
        task_constraints=list(ctx['task_constraints']))
