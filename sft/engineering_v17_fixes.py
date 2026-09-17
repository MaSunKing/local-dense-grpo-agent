"""Inference-only evidence/window repairs; no model, reward or tool restrictions."""
import copy
import math
import re
from collections import Counter


def pack_opened(opened, max_docs=6, per_doc_chars=2400):
    from evidence_handoff_v3 import within_budget
    documents, seen = {}, set()
    for event in reversed(opened or []):
        evidence = event.get('evidence') or {}
        if evidence.get('failed'):
            continue
        for chunk in evidence.get('chunks') or []:
            sid = str(chunk.get('source_id') or '')
            text = str(chunk.get('text') or '')
            if not sid or not text.strip():
                continue
            doc = sid.split('#', 1)[0]
            key = (doc, re.sub(r'\s+', ' ', text).strip())
            if key in seen:
                continue
            seen.add(key)
            documents.setdefault(doc, []).append(chunk)
    output, audit = [], []
    for index, (doc, chunks) in enumerate(documents.items()):
        remaining = per_doc_chars if index < max_docs else 0
        kept = []
        for chunk in chunks:
            text = chunk['text']
            shown = within_budget(text, remaining, remaining, None) if remaining else ''
            reason = ('kept' if shown == text else 'sentence_prefix_budget' if shown
                      else 'document_limit' if index >= max_docs else 'no_sentence_fits_budget')
            audit.append(dict(source_id=chunk['source_id'], selected=bool(shown), reason=reason,
                              original_chars=len(text), returned_chars=len(shown),
                              chunk_relative_span=[0, len(shown)]))
            if shown:
                kept.append(dict(source_id=chunk['source_id'], title=chunk.get('title'), text=shown,
                                 evidence_truncated=shown != text,
                                 chunk_relative_span=[0, len(shown)]))
                remaining -= len(shown)
        if kept:
            output.append(dict(source_id=doc, chunks=kept))
    return output, audit


def checklist_focus(state, fallback):
    gaps = []
    for statuses in ({'unknown', 'missing'}, {'partial'}):
        gaps = [r['description'] for r in state.get('requirements', []) if r.get('status') in statuses]
        if gaps:
            break
    return dict(current_query=fallback, checklist_gaps=' '.join(gaps))


def window(self, query, original_question, limit, reread_registry):
    from candidate_window import terms
    current = query.get('current_query', '') if isinstance(query, dict) else query
    gaps = query.get('checklist_gaps', '') if isinstance(query, dict) else ''
    rows = list(self.rows.values())
    docs = [terms(r.get('title', '')) * 2 + terms(r.get('_native_preview_text', '')) for r in rows]
    df = Counter(t for d in docs for t in set(d))
    average = sum(map(len, docs)) / max(1, len(docs))
    def score(d, q):
        tf = Counter(d)
        return sum(math.log(1 + (len(docs)-df[t]+.5)/(df[t]+.5)) * tf[t]*2.2 /
                   (tf[t]+1.2*(.25+.75*len(d)/max(1, average))) for t in set(terms(q)) if tf[t])
    ranked = []
    for row, doc in zip(rows, docs):
        parts = dict(current_query=score(doc, current), checklist_gaps=score(doc, gaps),
                     original_question=score(doc, original_question))
        value = .60*parts['current_query'] + .25*parts['checklist_gaps'] + .15*parts['original_question']
        ranked.append((value, row, parts))
    selected = []
    for value, row, parts in sorted(ranked, key=lambda x: (-x[0], x[1]['source_id']))[:limit]:
        r = copy.deepcopy(row)
        r['_preview_query'] = current
        r['ranking_audit'] = dict(score=value, components=parts, current_query=current, checklist_gaps=gaps)
        r['previous_browse_queries'] = list(reread_registry.get(r['source_id'], {}).get('previous_browse_queries', []))
        r['requires_new_query'] = False
        selected.append(r)
    return selected


def bound_previews(rows):
    # Select directly from normalized native text ONCE, under the final budget.
    from candidate_window import excerpt
    result = copy.deepcopy(rows)
    share = 1200 // max(1, len(result))
    for row in result:
        raw = re.sub(r'\s+', ' ', str(row.get('_native_preview_text') or '')).strip()
        paper = str(row.get('source_id', '')).startswith(('PMID:', 'S2:'))
        cap = min(240 if paper else 160, share)
        shown, spans, truncated = excerpt(raw, str(row.get('_preview_query') or ''), cap) if cap else ('', [], bool(raw))
        row.update(search_preview=shown, preview_selected_spans=[list(s) for s in spans],
                   preview_truncated=truncated, preview_available=bool(raw),
                   preview_offsets_reference='whitespace_normalized_native_text',
                   preview_budget_chars=cap, preview_selection='single_pass_native_excerpt_v17')
        assert shown == ' '.join(raw[a:b] for a, b in spans)
        assert len(shown) <= cap
    return result

from source_context_v19 import wrap_pack
pack_opened = wrap_pack(pack_opened)
