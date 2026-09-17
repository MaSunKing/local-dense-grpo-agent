"""Code-owned provenance and observations; no action prohibitions."""
import copy
import re

def search_environment_failure(output):
    kind = output.get('failure_type')
    if kind == 'search_no_results' or output.get('error') == 'no_supported_medical_web_results':
        return False
    return bool(output.get('failed') or output.get('error'))

def normalized(text):
    return re.sub(r'\s+', ' ', str(text)).strip()

def browse_observation(opened, evidence, source_id):
    previous = {normalized(c.get('text', '')) for e in opened
                if e.get('source_id') == source_id
                for c in (e.get('evidence') or {}).get('chunks', [])}
    current = {normalized(c.get('text', '')) for c in evidence.get('chunks', []) if normalized(c.get('text', ''))}
    return dict(source_id=source_id, new_text_chunks=len(current-previous),
                unchanged_text_chunks=len(current & previous),
                no_new_text=bool(current) and not (current-previous),
                comparison='whitespace_normalized_exact_text_not_semantic_gain')

def wrap_pack(original):
    def pack(opened, max_docs=6, per_doc_chars=2400):
        events = copy.deepcopy(opened or [])
        seen_ids, headers, conflicts = {}, {}, []
        for event in reversed(events):
            ev = event.get('evidence') or {}
            if ev.get('failed'):
                continue
            kept = []
            for c in ev.get('chunks', []):
                sid = c.get('source_id')
                if not sid or not normalized(c.get('text', '')):
                    continue
                if sid in seen_ids:
                    if normalized(c['text']) != seen_ids[sid]:
                        conflicts.append(dict(source_id=sid, reason='older_same_id_text_version_omitted', selected=False))
                    continue
                seen_ids[sid] = normalized(c['text'])
                m = ev.get('document_metadata') or {}
                headers[sid] = dict(document_id=sid.split('#',1)[0], citation_id=sid,
                    source_type=m.get('source_type'), year=m.get('year') or m.get('publication_year'),
                    content_level=m.get('content_level'), abstract_only=m.get('abstract_only'),
                    full_text_available=m.get('full_text_available'), evidence_scope=m.get('evidence_scope'),
                    model_view='selected_passages_not_entire_document', metadata_owner='tool_code')
                kept.append(c)
            ev['chunks'] = kept
        packed, audit = original(events, max_docs=max_docs, per_doc_chars=per_doc_chars)
        for doc in packed:
            for c in doc['chunks']:
                c['source_header'] = headers[c['source_id']]
        return packed, audit + conflicts
    return pack
