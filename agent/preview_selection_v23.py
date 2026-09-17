"""Source-exact query/question sentence selection; no first-sentence priority."""
import re
import preview_tokens_v22 as budget

def select(raw, query, target, ceiling):
    from candidate_window import terms
    raw=re.sub(r'\s+',' ',str(raw or '')).strip()
    if not raw or ceiling<=0:return '',[],bool(raw)
    if budget.count(raw)<=ceiling:return raw,[[0,len(raw)]],False
    original=query.get('question','') if isinstance(query,dict) else ''
    focus=query.get('focus','') if isinstance(query,dict) else str(query)
    oq=set(terms(original));fq=set(terms(focus));spans=[];start=0
    for m in re.finditer(r'[.!?。！？](?=\s|$)',raw):
        if re.search(r'\b(?:e\.g|i\.e|vs|al|Dr|Fig)\.$',raw[:m.end()],re.I):continue
        a=start
        while a<m.end() and raw[a].isspace():a+=1
        spans.append((a,m.end()));start=m.end()
    if raw[start:].strip():
        while raw[start].isspace():start+=1
        spans.append((start,len(raw)))
    def relevance(s):
        ts=set(terms(raw[s[0]:s[1]]))
        return .7*len(ts & oq)/max(1,len(oq))+.3*len(ts & fq)/max(1,len(fq)) if oq else len(ts & fq)
    ranked=sorted(spans,key=lambda s:(-relevance(s),s[0]))
    # Start with the relevant sentence, not the document introduction. Exact
    # negation, populations and qualifiers inside selected sentences are retained.
    a,b=ranked[0]
    if budget.count(raw[a:b])>ceiling:
        lo=a;hi=b
        while lo<hi:
            mid=(lo+hi+1)//2
            if budget.count(raw[a:mid])<=ceiling:lo=mid
            else:hi=mid-1
        end=lo;boundary=raw.rfind(' ',a,end)
        if boundary>a:end=boundary
        # Do not expose half a parenthesized estimate/CI if a complete earlier
        # parenthesized clause fits. Offsets still refer to unchanged source text.
        prefix=raw[a:end]
        if prefix.count('(')>prefix.count(')'):
            last_close=raw.rfind(')',a,end)
            if last_close>a:end=last_close+1
        return raw[a:end],[[a,end]],True
    chosen=[ranked[0]]
    for span in ranked[1:]:
        if budget.count(' '.join(raw[a:b] for a,b in sorted(chosen)))>=target:break
        trial=sorted(chosen+[span])
        if budget.count(' '.join(raw[a:b] for a,b in trial))<=ceiling:chosen=trial
    chosen.sort()
    return ' '.join(raw[a:b] for a,b in chosen),[list(s) for s in chosen],False

def install(collection):
    budget.select=select
    previous=collection.policy_prompt
    def prompt(**kwargs):
        if kwargs.get('role') in {'decision','browse','search'}:
            candidates=[]
            for row in kwargs.get('candidates',[]):
                c=dict(row)
                c['_preview_query']={'question':kwargs.get('question',''),
                                     'focus':row.get('_preview_query',row.get('origin_query',''))}
                candidates.append(c)
            kwargs['candidates']=candidates
        return previous(**kwargs)
    collection.policy_prompt=prompt
