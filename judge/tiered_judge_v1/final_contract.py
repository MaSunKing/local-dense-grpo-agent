"""Reduced Final contract: frozen requirements + factual/citation support only."""
import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from general_final_v6.contract import task_context,need,fields,ids,weighted_mean,digest
from reward_policy import scalar

VERSION='tiered_final_v1'
SUPPORT={'supported':1.,'partial':.5,'unsupported':0.,'contradicted':0.,'overclaimed':0.,'unobservable':None}
CITATION={'supported':1.,'partial':.5,'mismatch':0.,'missing':0.,'contradicted':0.,'unobservable':None}
CONTENT={'full':1.,'partial':.5,'missing':0.,'unobservable':None}

def normalize(result):
    need(isinstance(result,dict),'result object')
    value=copy.deepcopy(result)
    for alias,key in (('core_units','units'),('core_content','content')):
        if alias in value:
            need(key not in value,'ambiguous alias')
            value[key]=value.pop(alias)
    return value

def prepare(payload,parser,task_mode='evidence_grounded'):
    need(task_mode in ('evidence_grounded','self_contained'),'frozen task mode')
    context=task_context(payload);raw=payload['records'][0]['raw_completion']
    body,units,_=parser(raw,context['evidence'])
    if task_mode=='self_contained':need(not context['evidence'] and all(not u['citation_ids'] for u in units),'self-contained mode requires no external evidence or citations')
    spans=[dict(id=f'A{i+1:04d}',**u) for i,u in enumerate(units)]
    packet=dict(version=VERSION,task_mode=task_mode,context=context,raw_completion=raw,answer_spans=spans)
    return packet|{'binding_digest':digest(packet)}

def validate(packet,result,parser):
    result=normalize(result)
    ctx=packet['context'];payload={k:ctx[k] for k in ('question','requirements','constraints')}
    payload['records']=[dict(kind='final',raw_completion=packet['raw_completion'],evidence=ctx['evidence'],source_headers=ctx['source_headers'])]
    need(packet==prepare(payload,parser,packet['task_mode']),'packet binding changed')
    need(isinstance(result,dict) and {'units','content'}<=set(result) and set(result)<={'units','content','auxiliary'},'result fields')
    units={u['id']:u for u in packet['answer_spans']};rs={r['id']:r for r in ctx['requirements']};basis={b['span_id']:b for b in ctx['basis_spans']}
    def rows(xs,key,allowed,keys):
        need(isinstance(xs,list) and len(xs)==len(allowed),'row count');seen=set()
        for x in xs:
            fields(x,keys,'row fields');rid=x[key]
            need(isinstance(rid,str) and rid in allowed and rid not in seen,'unknown/duplicate record');seen.add(rid)
            need(isinstance(x['reason'],str) and bool(x['reason'].strip()),'reason')
        return xs
    fidelity=[];citation=[];complete=[];material=False
    for u in rows(result['units'],'answer_id',units,{'answer_id','factual_support','citation_support','factual_basis_ids','citation_basis_ids','reason'}):
        actual=units[u['answer_id']]['citation_ids'];f=u['factual_support'];c=u['citation_support']
        need(isinstance(f,str) and f in {*SUPPORT,'not_required'},'factual label')
        need(isinstance(c,str) and c in {*CITATION,'not_required'},'citation label')
        ids(u['factual_basis_ids'],basis);ids(u['citation_basis_ids'],basis)
        for sid in u['citation_basis_ids']:
            need(basis[sid]['kind']=='evidence' and basis[sid]['source_id'] in actual,'citation outside actual attachment')
        if f=='not_required':need(not actual and c=='not_required' and not u['factual_basis_ids'],'nonfactual exemption')
        else:
            if f in ('supported','partial','contradicted','overclaimed'):need(bool(u['factual_basis_ids']),'factual basis required')
            fidelity.append((SUPPORT[f],1))
        if c=='not_required':need(not actual and not u['citation_basis_ids'],'citation exemption')
        else:
            if c in ('supported','partial','contradicted'):need(bool(actual) and bool(u['citation_basis_ids']),'citation basis required')
            if c=='mismatch':need(bool(actual),'mismatch requires actual attachment')
            if c=='missing':need(not actual,'missing with attached citation')
            citation.append((CITATION[c],1))
        material|=f in ('unsupported','contradicted','overclaimed') or c in ('mismatch','missing','contradicted')
    for r in rows(result['content'],'requirement_id',rs,{'requirement_id','status','answer_ids','reason'}):
        need(isinstance(r['status'],str) and r['status'] in CONTENT,'content status');ids(r['answer_ids'],units)
        if r['status'] in ('full','partial'):need(bool(r['answer_ids']),'content binding')
        if r['status']=='missing':need(not r['answer_ids'],'missing binding')
        material|=r['status']=='missing'
        complete.append((CONTENT[r['status']],rs[r['requirement_id']].get('weight',1)))
    # Auxiliary schema defects cannot discard valid core judgments. Record them.
    aux=result.get('auxiliary');label=None
    if isinstance(aux,dict) and set(aux)=={'grade','reason'} and isinstance(aux.get('reason'),str):label=aux.get('grade')
    core=dict(completeness=weighted_mean(complete),fidelity=weighted_mean(fidelity),citation_support=weighted_mean(citation))
    report=scalar('final',core,label,material_core_error=material,excluded=('citation_support',) if packet['task_mode']=='self_contained' else ())
    return report|dict(version=VERSION,binding_digest=packet['binding_digest'],result_digest=digest(result),
        explicit_requirement_limits_retained=True,auto_limitations='not_requested_not_in_reward',
        semantic_review='PENDING',collector_export_authorized=False)

def wire(packet):
    ctx=packet['context']
    return dict(task_mode=packet['task_mode'],question=ctx['question'],requirements=ctx['requirements'],constraints=ctx['constraints'],
        source_headers=ctx['source_headers'],basis_spans=[{k:s[k] for k in ('span_id','source_id','kind','text')} for s in ctx['basis_spans']],
        answer_spans=[dict(id=u['id'],text=u['span']['quote'],actual_citation_ids=u['citation_ids']) for u in packet['answer_spans']])
