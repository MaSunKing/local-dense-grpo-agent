"""Expand grouped chunk citations. No aliasing, source invention or text rewrite."""
import re

def install():
    """Opt-in parser adapter; immutable historical snapshots remain unchanged."""
    import evidence_handoff_v3 as module
    original=module.parse_final
    if getattr(original,'grouped_citations_v4_1',False):return
    def parse(raw,available_ids,hit_token_limit=False):
        groups=re.findall(r'<cite\s+id=[\"\']([^\"\']+)[\"\']',str(raw),flags=re.I)
        result=original(raw,set(available_ids)|set(groups),hit_token_limit)
        error=None
        try:ids=citation_ids(result.get('answer') or '')
        except ValueError as exc:ids=[];error=str(exc)
        unknown=sorted(set(ids)-set(available_ids))
        result.update(citation_ids=ids,citation_format_error=error,unknown_citation_ids=unknown)
        if error or unknown:
            result.update(strict_protocol_passed=False,usable_for_display=False)
        return result
    parse.grouped_citations_v4_1=True
    module.parse_final=parse

def citation_ids(text):
    text = str(text)
    tags = list(re.finditer(r'</?cite\b[^>]*>', text, flags=re.I))
    remainder = re.sub(r'</?cite\b[^>]*>', '', text, flags=re.I)
    if re.search(r'</?cite\b', remainder, flags=re.I):
        raise ValueError('incomplete citation tag')
    ids=[]
    opened = False
    for tag in tags:
        if re.fullmatch(r'</cite\s*>', tag.group(), flags=re.I):
            if not opened: raise ValueError('unexpected closing citation tag')
            opened = False
            continue
        match = re.fullmatch(r'<cite\s+id=([\"\'])([^<>\"\']*?)\1\s*>', tag.group(), flags=re.I)
        if not match: raise ValueError('invalid citation attributes')
        if opened: raise ValueError('nested citation tags')
        opened = True
        parts=[p.strip() for p in match.group(2).split(',')]
        if any(not p for p in parts): raise ValueError('empty citation ID in group')
        ids.extend(parts)
    if opened: raise ValueError('unclosed citation tag')
    return ids
