
import re

CITE=re.compile(r'<cite\s+id="([^"<>\s]+)">([^<>]*)</cite>')

def bind(raw, evidence):
    if not isinstance(raw,str):
        raise ValueError("Final text required")
    if raw.count("<answer>")!=1 or raw.count("</answer>")!=1:
        raise ValueError("exactly one answer boundary required")
    a=raw.index("<answer>")+len("<answer>")
    b=raw.index("</answer>")
    if b<a:
        raise ValueError("invalid answer boundary")
    body=raw[a:b]
    ids=[x["source_id"] for x in evidence]
    if len(ids)!=len(set(ids)):
        raise ValueError("duplicate evidence")
    allowed=set(ids)
    # 只接受当前已知cite格式，不猜测其他标签的含义。
    remaining=CITE.sub("",body)
    if "<" in remaining or ">" in remaining:
        raise ValueError("unsupported answer markup; explicit parser update required")
    units=[]
    attachments={}
    for m in re.finditer(r'\S.*?(?=\r?\n[ \t]*\r?\n|\Z)',body,re.S):
        start=m.start()
        paragraph=m.group().rstrip()
        end=start+len(paragraph)
        refs=[]
        for c in CITE.finditer(paragraph):
            citation_ids = c[1].split(',')
            if any(not sid or sid != sid.strip() for sid in citation_ids):
                raise ValueError("empty or malformed citation ID")
            for sid in citation_ids:
                if sid not in allowed:
                    raise ValueError("answer cites unopened evidence")
                if sid not in refs:
                    refs.append(sid)
        units.append({
            "span":{"start":start,"end":end,"quote":paragraph},
            "citation_ids":refs,
        })
        attachments[(start,end)]=refs
    if not units:
        raise ValueError("empty answer")
    covered=[False]*len(body)
    for unit in units:
        s=unit["span"]
        for i in range(s["start"],s["end"]):
            if covered[i]:
                raise ValueError("overlapping answer units")
            covered[i]=True
    if any(not ch.isspace() and not covered[i] for i,ch in enumerate(body)):
        raise ValueError("answer text omitted by parser")
    return body,units,attachments
