"""ID-only Final audit. Structural binding is not semantic acceptance."""
import copy
import hashlib
import json
import math
import re

VERSION = 'final_span_contract_v6'
TYPES = {'population_scope','subgroup_unavailable','comparator_mismatch','outcome_mismatch',
         'followup_limitation','study_design_limitation','imprecision','confounding',
         'selection_bias','missing_data','indirectness','conflicting_evidence','regulatory_or_temporal_scope'}
CONTENT = {'full':1.,'partial':.5,'missing':0.,'unobservable':None}
SUPPORT = {'supported':1.,'partial':.5,'unsupported':0.,'contradicted':0.,'overclaimed':0.,'unobservable':None}
CITATION = {'supported':1.,'partial':.5,'mismatch':0.,'missing':0.,'contradicted':0.,'unobservable':None}

def need(ok, msg):
    if not ok: raise ValueError(msg)

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def fields(value, keys, message):
    need(isinstance(value,dict) and set(value)==set(keys),message)

def reason(value):
    need(isinstance(value,str) and bool(value.strip()),'nonempty reason required')

def ids(value, allowed):
    need(isinstance(value,list) and all(isinstance(x,str) for x in value),'ID list required')
    need(len(value)==len(set(value)) and set(value)<=set(allowed),'unknown or duplicate ID')

def indexed(text, prefix, source_id, kind):
    """Deterministic surface spans; never claim linguistic/semantic atomization."""
    need(isinstance(text,str) and bool(text.strip()),'source text required')
    boundaries=[0]+[m.end() for m in re.finditer(r'(?<=[.!?])\s+(?=[A-Z])|\n\s*\n',text)]+[len(text)]
    spans=[]
    for a,b in zip(boundaries,boundaries[1:]):
        while a<b and text[a].isspace(): a+=1
        while b>a and text[b-1].isspace(): b-=1
        if a==b: continue
        value=dict(span_id=f'{prefix}{len(spans)+1:04d}',source_id=source_id,kind=kind,start=a,end=b,text=text[a:b])
        value['digest']=digest(value);spans.append(value)
    covered=set()
    for s in spans: covered.update(range(s['start'],s['end']))
    need(all(ch.isspace() or i in covered for i,ch in enumerate(text)),'source text omitted')
    return spans

def task_context(payload):
    need(len(payload['records'])==1 and payload['records'][0]['kind']=='final','one Final required')
    rec=payload['records'][0]; reqs=payload['requirements']; evs=rec['evidence']
    need(isinstance(reqs,list) and bool(reqs),'requirements required')
    need(all(isinstance(r.get('id'),str) for r in reqs) and len({r['id'] for r in reqs})==len(reqs),'requirement IDs')
    need(all(isinstance(e.get('source_id'),str) for e in evs) and len({e['source_id'] for e in evs})==len(evs),'evidence IDs')
    for r in reqs:
        w=r.get('weight',1)
        need(type(w) in (int,float) and math.isfinite(w) and w>0,'positive frozen weight')
    spans=indexed(payload['question'],'Q','question','question')
    for i,r in enumerate(reqs):spans+=indexed(r['description'],f'R{i+1}_',r['id'],'requirement')
    for i,e in enumerate(evs):spans+=indexed(e['text'],f'E{i+1}_',e['source_id'],'evidence')
    # Raw answer, trajectory identities and old scores never enter this allowlist.
    context=copy.deepcopy(dict(version=VERSION,question=payload['question'],requirements=reqs,
        constraints=payload['constraints'],evidence=evs,source_headers=rec.get('source_headers',{})))
    context['basis_spans']=spans
    return context

def verify_context(context):
    payload={k:context[k] for k in ('question','requirements','constraints')}
    payload['records']=[dict(kind='final',evidence=context['evidence'],source_headers=context['source_headers'])]
    need(context==task_context(payload),'source/span context changed')

def freeze(context, draft):
    verify_context(context)
    fields(draft,{'requirements'},'rubric top-level fields')
    rs=context['requirements'];rows=draft['requirements'];spans={s['span_id']:s for s in context['basis_spans']}
    need(isinstance(rows,list) and len(rows)==len(rs),'rubric row count')
    seen=set(); all_lids=set()
    for row in rows:
        fields(row,{'id','applicability','limitations','reason'},'rubric row fields')
        rid=row['id'];need(isinstance(rid,str) and rid in {r['id'] for r in rs} and rid not in seen,'rubric requirement ID');seen.add(rid)
        reason(row['reason']);a=row['applicability'];need(a in ('required','not_required','unobservable'),'applicability')
        ls=row['limitations'];need(isinstance(ls,list),'limitations list')
        need(bool(ls)==(a=='required'),'applicability/list mismatch')
        signatures=set()
        for l in ls:
            fields(l,{'limitation_id','type','basis_span_ids','reason'},'limitation fields')
            lid=l['limitation_id'];need(isinstance(lid,str) and lid.strip() and lid not in all_lids,'limitation ID');all_lids.add(lid)
            need(isinstance(l['type'],str) and l['type'] in TYPES,'limitation type')
            reason(l['reason']);ids(l['basis_span_ids'],spans);need(bool(l['basis_span_ids']),'limitation basis required')
            for sid in l['basis_span_ids']:
                s=spans[sid]
                need(s['kind']!='requirement' or s['source_id']==rid,'cross-requirement basis')
            signature=digest([l['type'],sorted(l['basis_span_ids'])]);need(signature not in signatures,'duplicate limitation');signatures.add(signature)
    body=dict(version=VERSION,context_digest=digest(context),requirements=copy.deepcopy(rows))
    return body|{'digest':digest(body),'semantic_review':'PENDING'}

def prepare(payload, parser, frozen):
    context=task_context(payload)
    need(frozen==freeze(context,{'requirements':frozen['requirements']}),'frozen rubric changed')
    body,units,_=parser(payload['records'][0]['raw_completion'],context['evidence'])
    answer_spans=[]
    for i,u in enumerate(units):
        # Keep paragraph citation scope; do not guess sentence-to-citation relations.
        value=dict(span_id=f'A{i+1:04d}',start=u['span']['start'],end=u['span']['end'],
            text=u['span']['quote'],attached_citation_ids=u['citation_ids'])
        value['digest']=digest(value);answer_spans.append(value)
    return dict(version=VERSION,context=context,frozen_limitations=copy.deepcopy(frozen),
                answer_body=body,answer_spans=answer_spans,
                answer_digest=digest([body,answer_spans]))

def weighted_mean(values):
    if not values or any(v is None for v,w in values):return None
    return sum(v*w for v,w in values)/sum(w for v,w in values)

def validate(packet,result,parser):
    context=packet['context'];verify_context(context);frozen=packet['frozen_limitations']
    need(frozen==freeze(context,{'requirements':frozen['requirements']}),'rubric changed')
    payload={k:context[k] for k in ('question','requirements','constraints')}
    payload['records']=[dict(kind='final',raw_completion='<answer>'+packet['answer_body']+'</answer>',
        evidence=context['evidence'],source_headers=context['source_headers'])]
    need(packet==prepare(payload,parser,frozen),'answer binding changed')
    fields(result,{'units','content','limitations'},'result fields')
    answers={s['span_id']:s for s in packet['answer_spans']};basis={s['span_id']:s for s in context['basis_spans']}
    reqs={r['id']:r for r in context['requirements']}
    limits={l['limitation_id']:(r,l) for r in frozen['requirements'] for l in r['limitations']}
    def rows(xs,key,allowed,keys):
        need(isinstance(xs,list) and len(xs)==len(allowed),'result row count');seen=set()
        for x in xs:
            fields(x,keys,'result row fields');rid=x[key]
            need(isinstance(rid,str) and rid in allowed and rid not in seen,'result ID');seen.add(rid);reason(x['reason'])
        return xs
    fs=[];cs=[];content=[];ls={};unit_verdicts=[]
    for u in rows(result['units'],'answer_span_id',answers,{'answer_span_id','factual_support','citation_support','factual_basis_span_ids','citation_basis_span_ids','reason'}):
        actual=answers[u['answer_span_id']]['attached_citation_ids'];f=u['factual_support'];c=u['citation_support']
        need(isinstance(f,str) and f in {*SUPPORT,'not_required'},'factual label')
        need(isinstance(c,str) and c in {*CITATION,'not_required'},'citation label')
        ids(u['factual_basis_span_ids'],basis);ids(u['citation_basis_span_ids'],basis)
        for sid in u['citation_basis_span_ids']:
            need(basis[sid]['kind']=='evidence' and basis[sid]['source_id'] in actual,'basis outside actual citation attachment')
        if f=='not_required':need(not actual and c=='not_required' and not u['factual_basis_span_ids'],'nonfactual exemption')
        else:
            if f in ('supported','partial','contradicted','overclaimed'):need(bool(u['factual_basis_span_ids']),'factual basis required')
            fs.append((SUPPORT[f],1))
        if c=='not_required':need(not actual and not u['citation_basis_span_ids'],'citation exemption')
        else:
            if c in ('supported','partial','contradicted'):need(bool(actual) and bool(u['citation_basis_span_ids']),'attached citation basis required')
            if c=='mismatch':need(bool(actual),'mismatch needs actual attachment')
            if c=='missing':need(not actual,'missing with attachment')
            cs.append((CITATION[c],1))
        unit_verdicts.append(dict(answer_span_id=u['answer_span_id'],attached_citation_ids=actual,factual_support=f,citation_support=c))
    for r in rows(result['content'],'requirement_id',reqs,{'requirement_id','status','answer_span_ids','reason'}):
        need(isinstance(r['status'],str) and r['status'] in CONTENT,'content status');ids(r['answer_span_ids'],answers)
        if r['status'] in ('full','partial'):need(bool(r['answer_span_ids']),'content answer required')
        if r['status']=='missing':need(not r['answer_span_ids'],'missing content references')
        content.append((CONTENT[r['status']],reqs[r['requirement_id']].get('weight',1)))
    for l in rows(result['limitations'],'limitation_id',limits,{'limitation_id','status','acknowledged_answer_span_ids','reason'}):
        need(l['status'] in ('assessed','unobservable'),'limitation assessment status')
        ids(l['acknowledged_answer_span_ids'],answers)
        if l['status']=='unobservable':need(not l['acknowledged_answer_span_ids'],'unobservable acknowledgment')
        ls[l['limitation_id']]=None if l['status']=='unobservable' else float(bool(l['acknowledged_answer_span_ids']))
    qualifications=[];applicability={}
    for r in frozen['requirements']:
        applicability[r['id']]=r['applicability']
        if r['applicability']=='unobservable':qualifications.append((None,reqs[r['id']].get('weight',1)))
        elif r['applicability']=='required':
            qualifications.append((weighted_mean([(ls[l['limitation_id']],1) for l in r['limitations']]),reqs[r['id']].get('weight',1)))
    return dict(version=VERSION,diagnostic_scores=dict(content_completeness=weighted_mean(content),
        fidelity=weighted_mean(fs),citation_support=weighted_mean(cs),limitation_coverage=weighted_mean(qualifications)),
        limitation_applicability=applicability,unit_verdicts=unit_verdicts,
        packet_digest=digest(packet),rubric_digest=frozen['digest'],result_digest=digest(result),
        overall_reward=None,semantic_review='PENDING',training_ready=False,reward_ready=False)
