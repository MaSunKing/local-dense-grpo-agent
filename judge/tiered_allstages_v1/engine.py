"""Private post-trajectory judging. Not an Agent tool or a training authorization."""
import copy,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tiered_judge_v1'))
from reward_policy import scalar,digest as score_digest
sys.path.insert(0,str(ROOT/'fixed'))
from response_validator import parse_response
from guards import verify_stop,digest,task_scope_digest
sys.path.insert(0,str(ROOT/'tiered_judge_v2'))
import isolated as final_engine
sys.path.insert(0,str(Path(__file__).resolve().parent))
from input_contract import validate_inputs, task_mode, semantic_evidence, canonical_stop_context

GRADES={'correct':1.,'partial':.5,'incorrect':0.,'unobservable':None}
COVER={'direct':1.,'partial':.5,'unknown':0.,'unobservable':None}
FULL={'full':1.,'partial':.5,'missing':0.,'unobservable':None}
ALIASES={'checklist_init':'checklist','state_update':'state'}
DIMENSIONS={'search':('gap_relevance','scope_fidelity'),'browse':('source_relevance','scope_fidelity'),'stop':('stop_appropriateness',)}

def need(ok,msg):
    if not ok:raise ValueError(msg)
def avg(xs):return None if not xs or any(x is None for x in xs) else sum(xs)/len(xs)
def weighted(xs):return None if not xs or any(x is None for x,w in xs) else sum(x*w for x,w in xs)/sum(w for x,w in xs)
def indexed(rows,key='id'):
    need(isinstance(rows,list),'list required');result={}
    for r in rows:
        need(isinstance(r,dict) and isinstance(r.get(key),str) and r[key].strip() and r[key] not in result,'duplicate/invalid ID')
        result[r[key]]=r
    return result
def refs(xs,allowed):need(isinstance(xs,list) and all(isinstance(x,str) for x in xs) and len(xs)==len(set(xs)) and set(xs)<=set(allowed),'unknown/duplicate reference')
def fields(row,keys):need(isinstance(row,dict) and set(row)==set(keys),'exact core fields required')
def reason(x):need(isinstance(x,str) and bool(x.strip()),'nonempty reason required')
def grade(x):
    fields(x,{'verdict','reason'});reason(x['reason']);need(isinstance(x['verdict'],str) and x['verdict'] in GRADES,'coarse grade required');return GRADES[x['verdict']]
def scope_signal(x):
    """Map observable/helpful/complete facts to a fixed semantic grade."""
    fields(x,{'material_help','complete_scope','reason'});reason(x['reason'])
    helpful=x['material_help'];complete=x['complete_scope']
    need(helpful is None or type(helpful) is bool,'material_help must be boolean or null')
    if helpful is None:
        need(complete is None,'unobservable scope requires complete_scope null')
        return None,'unobservable'
    need(type(complete) is bool,'complete_scope must be boolean')
    if not helpful:
        need(not complete,'complete scope cannot exist without material help')
        return 0.,'unknown'
    return (1.,'direct') if complete else (.5,'partial')
ATOMIC_SUPPORT_FIELDS={
    'any_requested_option_or_member_present','requested_outcome_support',
    'population_applicable','complete_requirement_support','reason'
}
def atomic_support_signal(x):
    """Derive help/completeness from simpler observable facts."""
    fields(x,ATOMIC_SUPPORT_FIELDS);reason(x['reason'])
    names=('any_requested_option_or_member_present','requested_outcome_support','population_applicable','complete_requirement_support')
    values=[x[name] for name in names]
    need(all(value is None or type(value) is bool for value in values),'support atoms must be boolean or null')
    if any(value is None for value in values):
        need(all(value is None for value in values),'unobservable support requires all atoms null')
        return None,'unobservable'
    material=all(values[:3]);complete=values[3]
    need(not complete or material,'complete support requires option, outcome, and population support')
    if not material:return 0.,'unknown'
    return (1.,'direct') if complete else (.5,'partial')
def auxiliary(result):
    x=result.get('auxiliary')
    return x.get('grade') if isinstance(x,dict) and set(x)=={'grade','reason'} and isinstance(x['reason'],str) else None

def prepare(payload,collector_lookup=None,task_modes=None):
    p=copy.deepcopy(payload)
    need(isinstance(p.get('question'),str) and p['question'].strip(),'question required')
    req=indexed(p['requirements']);need(bool(req),'frozen requirements required')
    for r in req.values():
        need(isinstance(r.get('description'),str) and r['description'].strip(),'requirement text')
        w=r.get('weight',1);need(type(w) in (int,float) and math.isfinite(w) and w>0,'positive fixed weight')
    need(len(p['records'])==1,'one captured step per packet');rec=p['records'][0]
    kind=ALIASES.get(rec['kind'],rec['kind'])
    validate_inputs(p)
    mode=task_mode(p,task_modes)
    if kind=='final':return dict(kind='final',final=final_engine.packets(p,mode),payload=p)
    need(kind in {'checklist','search','browse','state','stop','evidence'},'unsupported stage')
    need(isinstance(rec.get('raw_completion'),str),'actual raw output required')
    evidence=rec['evidence'];ev=indexed(evidence,'source_id')
    need(all(isinstance(e.get('text'),str) and e['text'].strip() for e in evidence),'evidence text')
    need(isinstance(rec.get('source_headers'),dict),'source headers required')
    snap=rec.get('evidence_snapshot',{})
    need(type(snap.get('step')) is int and snap['step']>=0 and snap.get('digest')==digest(evidence),'step evidence digest')
    base={k:copy.deepcopy(p[k]) for k in ('question','requirements','constraints')}
    semantic=base|dict(evidence=semantic_evidence(rec),source_headers=copy.deepcopy(rec['source_headers']))
    views={};skip=None;decoded=None
    if kind in {'checklist','search','browse','state'}:decoded=parse_response(rec['raw_completion'])
    if kind=='checklist':
        fields(decoded,{'items'});items=indexed(decoded['items']);need(bool(items),'empty model checklist')
        for r in items.values():fields(r,{'id','description'});need(isinstance(r['description'],str) and r['description'].strip(),'actual item text')
        views['checklist']=base|dict(model_items=decoded['items'])
    if kind in {'search','browse'}:
        fields(decoded,{'tool','arguments'});need(decoded['tool']==kind and isinstance(decoded['arguments'],dict),'actual action shape')
        visible=rec['visible_context'];need(isinstance(visible.get('candidates'),list),'visible candidate list')
        candidates=indexed(visible['candidates'])
        for c in candidates.values():need(set(c)<={'id','title','snippet','url','source_type'},'candidate contains future body or hidden judgment')
        need(all(isinstance(c.get('title'),str) for c in candidates.values()),'candidate title')
        for f in ('title','snippet','url','source_type'):
            need(all(f not in c or isinstance(c[f],str) for c in candidates.values()),'candidate text fields')
        view=semantic|dict(actual_action=decoded,candidates=visible['candidates'])
        if kind=='browse':
            fields(decoded['arguments'],{'candidate_id','focus'})
            need(decoded['arguments']['candidate_id'] in candidates,'browse source not visible at decision')
            need(isinstance(decoded['arguments']['focus'],str),'browse focus')
        else:
            fields(decoded['arguments'],{'query'})
            need(isinstance(decoded['arguments']['query'],str) and decoded['arguments']['query'].strip(),'query text')
        # Deliberately omit later tool outputs, Final, model-claimed coverage and private scores.
        if kind=='browse':view['decision_contract']='atomic_component_outcome_population_v2'
        views[kind]=view
    if kind=='state':
        items=indexed(rec['visible_context']['items']);need(bool(items),'fixed policy items')
        for r in items.values():fields(r,{'id','description'});need(isinstance(r['description'],str),'item description')
        fields(decoded,{'updates'});updates=indexed(decoded['updates']);need(set(updates)==set(items),'state cannot omit or create items')
        for u in updates.values():
            fields(u,{'id','status','evidence_ids'});need(u['status'] in {'direct','partial','unknown'},'state status');refs(u['evidence_ids'],ev)
            if u['status']=='unknown':need(not u['evidence_ids'],'unknown cannot assert support')
        views['state_truth']=semantic|dict(policy_items=rec['visible_context']['items'],decision_contract='atomic_component_outcome_population_v2')
        citation_items=[dict(id=u['id'],description=items[u['id']]['description'],claimed_status=u['status'],evidence=[ev[e] for e in u['evidence_ids']]) for u in updates.values() if u['status']!='unknown']
        if citation_items:views['state_citation']=base|dict(items=citation_items,decision_contract='atomic_component_outcome_population_v2')
    if kind=='evidence':views['coverage']=semantic
    if kind=='stop':
        termination=verify_stop(p,collector_lookup)
        clean_context=canonical_stop_context(rec['stop_context']) if termination is not None else None
        if termination!='active':skip='not_voluntary_or_unverified'
        else:views['stop']=semantic|dict(stop_context=clean_context)
    return dict(kind=kind,payload=p,views=views,decoded=decoded,skip=skip,binding_digest=digest(p))

def validate(task,view,result):
    if task in {'checklist','search','browse'}:
        need(isinstance(result,dict) and set(result)<=({'coverage','scope','auxiliary'} if task=='checklist' else {'judgments','target_requirement_ids','auxiliary'}),'result fields')
    if task=='checklist':
        need({'coverage','scope'}<=set(result),'missing checklist core');rows=indexed(result['coverage']);wanted=indexed(view['requirements']);need(set(rows)==set(wanted),'frozen checklist denominator')
        parts=[]
        for q,r in rows.items():
            fields(r,{'id','verdict','item_ids','reason'});reason(r['reason']);need(r['verdict'] in FULL,'coverage grade');refs(r['item_ids'],indexed(view['model_items']))
            if r['verdict'] in {'full','partial'}:need(bool(r['item_ids']),'model item binding')
            if r['verdict']=='missing':need(not r['item_ids'],'missing item binding')
            parts.append((FULL[r['verdict']],wanted[q].get('weight',1)))
        core=dict(coverage=weighted(parts),scope_fidelity=grade(result['scope']))
    elif task in DIMENSIONS:
        if task=='stop':fields(result,{'judgments'})
        fields(result['judgments'],DIMENSIONS[task])
        core={d:(atomic_support_signal(result['judgments'][d])[0] if task=='browse' and view.get('decision_contract')=='atomic_component_outcome_population_v2' else scope_signal(result['judgments'][d])[0] if task=='browse' else grade(result['judgments'][d])) for d in DIMENSIONS[task]}
        if task!='stop':
            refs(result['target_requirement_ids'],indexed(view['requirements']))
            if core[DIMENSIONS[task][0]] is not None and core[DIMENSIONS[task][0]]>0:need(bool(result['target_requirement_ids']),'relevant action needs frozen task target')
    elif task in {'coverage','state_truth'}:
        fields(result,{'coverage'});target='requirements' if task=='coverage' else 'policy_items'
        rows=indexed(result['coverage']);wanted=indexed(view[target]);need(set(rows)==set(wanted),'evidence/state denominator')
        ev=indexed(view['evidence'],'source_id');values=[];normalized={}
        for q,r in rows.items():
            if task=='coverage' or view.get('decision_contract')=='atomic_component_outcome_population_v2':
                fields(r,{'id','any_requested_option_or_member_present','requested_outcome_support','population_applicable','complete_requirement_support','evidence_ids','reason'})
                value,status=atomic_support_signal({k:r[k] for k in ATOMIC_SUPPORT_FIELDS})
                if task=='coverage':
                    need(status!='unobservable',
                         'materialized coverage snapshot requires true/false atoms; use false for absent support')
            elif task=='state_truth':
                fields(r,{'id','material_help','complete_scope','evidence_ids','reason'})
                value,status=scope_signal({k:r[k] for k in ('material_help','complete_scope','reason')})
            else:
                fields(r,{'id','status','evidence_ids','reason'});reason(r['reason']);need(r['status'] in COVER,'evidence status')
                value,status=COVER[r['status']],r['status']
            refs(r['evidence_ids'],ev)
            if status in {'direct','partial'}:need(bool(r['evidence_ids']),'supported status needs actual evidence')
            if status in {'unknown','unobservable'}:need(not r['evidence_ids'],'unknown reference')
            values.append((value,wanted[q].get('weight',1)))
            normalized[q]=dict(id=q,status=status,evidence_ids=r['evidence_ids'],reason=r['reason'])
        return dict(score=weighted(values),coverage=normalized)
    elif task=='state_citation':
        fields(result,{'items'});rows=indexed(result['items']);targets=indexed(view['items']);need(set(rows)==set(targets),'state citation denominator')
        values={}
        for k,r in rows.items():
            allowed=indexed(targets[k]['evidence'],'source_id')
            if view.get('decision_contract')=='atomic_component_outcome_population_v2':
                fields(r,{'id','any_requested_option_or_member_present','requested_outcome_support','population_applicable','complete_requirement_support','evidence_ids','reason'})
                signal,status=atomic_support_signal({name:r[name] for name in ATOMIC_SUPPORT_FIELDS})
                refs(r['evidence_ids'],allowed)
                if status in {'direct','partial'}:need(bool(r['evidence_ids']),'support evidence required')
                if status in {'unknown','unobservable'}:need(not r['evidence_ids'],'unsupported citation cannot bind evidence')
                claimed=targets[k]['claimed_status']
                need(claimed in {'direct','partial'},'unsupported claimed citation status')
                values[k]=(None if signal is None else
                           1. if claimed=='partial' and signal>0 else
                           1. if claimed=='direct' and signal==1 else
                           .5 if claimed=='direct' and signal==.5 else 0.)
            else:
                fields(r,{'id','verdict','evidence_ids','reason'});reason(r['reason']);need(r['verdict'] in GRADES,'citation grade');refs(r['evidence_ids'],allowed)
                if r['verdict'] in {'correct','partial'}:need(bool(r['evidence_ids']),'support evidence required')
                values[k]=GRADES[r['verdict']]
        return dict(items=values)
    else:raise ValueError('unknown task')
    return scalar(task,core,auxiliary(result),material_core_error=any(v==0 for v in core.values()))

def judgment_key(scorer,task,view):return digest(dict(scorer=scorer,task=task,view=view))
def aggregate(bundle,judgments,scorer,collector_lookup=None,task_modes=None):
    need(bundle==prepare(bundle['payload'],collector_lookup,task_modes),'packet changed since preparation')
    kind=bundle['kind']
    if kind=='final':
        report=final_engine.aggregate(bundle['final'],judgments,scorer)
        return report|dict(binding_digest=bundle['final']['bound']['binding_digest'],
                           scorer=scorer,compiler_connected=False,training_ready=False)
    if bundle['skip']:return dict(stage=kind,score=None,status='not_eligible',reason=bundle['skip'],training_ready=False)
    results={}
    need(set(judgments)==set(bundle['views']),'all and only requested judgments')
    for task,view in bundle['views'].items():
        j=judgments[task];need(j['key']==judgment_key(scorer,task,view),'scorer/input mismatch');results[task]=validate(task,view,j['result'])
    if kind=='state':
        expected=results['state_truth']['coverage'];pred=indexed(bundle['decoded']['updates']);status=[];citation=[];grounded=[]
        for k,p in pred.items():
            correct=None if expected[k]['status']=='unobservable' else float(p['status']==expected[k]['status'])
            cited=None if p['status']=='unknown' else results['state_citation']['items'][k]
            status.append(correct);citation.append(cited)
            # Fixed item denominator. Citation cannot rescue an incorrect status.
            # Correct unknown is a correct abstention, not a fabricated citation score.
            grounded.append(None if correct is None else 0. if correct==0 else correct if p['status']=='unknown' else cited)
        core=dict(status_accuracy=avg(status),grounded_status_accuracy=avg(grounded))
        report=scalar('state',core,material_core_error=any(v==0 for v in status+grounded))
        report.update(citation_diagnostics=dict(zip(pred,citation)),scope_contract_valid=True,
                      aggregation='fixed_item_status_75_grounded_status_25_v2')
    elif kind=='evidence':report=dict(stage='evidence',score=results['coverage']['score'],coverage=results['coverage']['coverage'],training_ready=False,reward_export_authorized=False)
    else:report=results[kind]
    return report|dict(binding_digest=bundle['binding_digest'],scorer=scorer,training_ready=False,compiler_connected=False)

def coverage_delta(before,after,new_ids):
    """Diagnostic only. Never converts State labels into an evidence-gain event."""
    for row in (before,after):
        need(row['bundle']['kind']=='evidence','only evidence snapshots');need(row['report']['binding_digest']==row['bundle']['binding_digest'],'coverage report binding')
    a,b=before['bundle'],after['bundle'];pa,pb=a['payload'],b['payload'];ra,rb=before['report'],after['report']
    need(ra['scorer']==rb['scorer'] and task_scope_digest(pa)==task_scope_digest(pb),'frozen rubric/scorer drift')
    old,new=(indexed(p['records'][0]['evidence'],'source_id') for p in (pa,pb))
    need(set(old)<=set(new) and all(old[k]==new[k] for k in old),'evidence deleted or modified')
    ha,hb=(p['records'][0]['source_headers'] for p in (pa,pb));need(all(k in hb and hb[k]==v for k,v in ha.items()),'header changed')
    refs(new_ids,new);need(set(new_ids)==set(new)-set(old),'new evidence binding')
    if a['views']==b['views']:
        need(ra['coverage']==rb['coverage'],'conflicting identical evidence judgments');return dict(delta=0.,reward_export_authorized=False)
    if not new_ids:return dict(delta=None,reason='metadata_only',reward_export_authorized=False)
    delta=None if ra['score'] is None or rb['score'] is None else rb['score']-ra['score']
    return dict(delta=delta,needs_review=delta is not None and delta<0,reward_export_authorized=False)

CHANNELS={'checklist':'checklist','search':'search_query','browse':'browse_source_focus','state':'state','stop':'stop_boundary_event','evidence':'evidence_gain_event','final':'final_reward_once'}
def mapping_preview(stage,report):
    need(stage in CHANNELS,'stage mapping')
    return dict(target=CHANNELS[stage],diagnostic_score=report.get('score'),requires_tool_ledger=stage in {'stop','evidence'},compiler_connected=False,training_ready=False)
