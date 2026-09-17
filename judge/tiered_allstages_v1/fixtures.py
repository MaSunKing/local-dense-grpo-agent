"""Authored transfer fixtures. Never used as hidden expected answers in prompts."""
import copy,json
from engine import digest,task_scope_digest

def pack(kind,question,requirements,evidence,output,visible=None):
    return copy.deepcopy(dict(question=question,question_id='fixture-task',rollout_id='fixture-rollout',policy='a'*64,requirements=requirements,constraints=[],records=[dict(kind=kind,record_id='step-1',raw_completion=json.dumps(output) if not isinstance(output,str) else output,evidence=evidence,source_headers={},evidence_snapshot=dict(step=1,digest=digest(evidence)),visible_context=visible or {})]))

Q='Under the archive policy, state deleted project file retention, who can recover files, and whether successful recovery is guaranteed.'
REQ=[dict(id='Q1',description='State the deleted project file retention period.',weight=1),dict(id='Q2',description='Identify who may recover deleted files.',weight=1),dict(id='Q3',description='State whether successful recovery is guaranteed.',weight=1)]
RET=dict(source_id='RET',text='Deleted project files are retained for 45 days.')
ALL=[RET,dict(source_id='PERM',text='Only workspace administrators may recover deleted project files. Retention does not guarantee successful recovery.')]
CANDIDATES=[dict(id='guide',title='Official deleted-file recovery permissions',snippet='This policy section describes recovery roles and limitations.',source_type='official policy'),dict(id='theme',title='Dashboard display themes',snippet='Switch between light and dark themes.',source_type='help page')]

def stop_packet(good=True,termination='active'):
    p=pack('stop',Q,REQ,copy.deepcopy(ALL if good else [RET]),'FINAL_READY')
    r=p['records'][0]
    ctx=dict(voluntary=termination=='active',remaining_budget=0 if termination=='quota_exhausted' else 2,available_candidates=[] if good else copy.deepcopy(CANDIDATES[:1]),failed_reads=[],task_constraints=[])
    r['stop_context']=ctx
    cap=dict(question_id=p['question_id'],rollout_id=p['rollout_id'],record_id=r['record_id'],policy=p['policy'],input_ids=[11,12],output_ids=[13],capture_id='fixture-capture',raw_completion=r['raw_completion'])
    cap['source_sha256']=digest(cap)
    r.update(capture_id=cap['capture_id'],capture_sha256=cap['source_sha256'])
    binding=dict(question_id=p['question_id'],rollout_id=p['rollout_id'],record_id=r['record_id'],policy=p['policy'],token_digest=digest([cap['input_ids'],cap['output_ids']]),evidence_snapshot=r['evidence_snapshot'],task_scope_digest=task_scope_digest(p))
    event=dict(event_id='fixture-stop',type=termination,binding=binding,evidence=r['evidence'],source_headers=r['source_headers'],stop_context=ctx)
    r['runtime_termination']=dict(event_id=event['event_id'],event_digest=digest(event),type=termination)
    def resolver(event_id):
        assert event_id==event['event_id']
        return copy.deepcopy(event),copy.deepcopy(cap)
    return p,resolver

SQ='For cold-chamber units, compare type A and B capacity at 24 hours and report safety incident counts for those units.'
SREQ=[dict(id='Q1',description='Compare cold-chamber type A and B capacity at 24 hours.',weight=1),dict(id='Q2',description='Report cold-chamber safety incident counts.',weight=1)]
SEV=[dict(source_id='POOL',text='The pooled indoor and cold-chamber test measured mean 24-hour capacity of 64 units for type A and 60 for type B. Cold-chamber capacity results were not reported separately.'),dict(source_id='SAFE',text='Among cold-chamber units, safety incidents numbered zero for type A and zero for type B.')]
ITEMS=[dict(id='P1',description=SREQ[0]['description']),dict(id='P2',description=SREQ[1]['description'])]

def case(kind,variant='good'):
    good=variant=='good'
    if kind=='stop':return stop_packet(good)
    if kind=='checklist':
        items=[dict(id=f'P{i+1}',description=r['description']) for i,r in enumerate(REQ)]
        return pack(kind,Q,REQ,[],dict(items=items if good else items[:1])),None
    if kind=='search':
        query='archive deleted project files recovery permissions authorized roles' if good else 'dashboard dark theme color settings'
        return pack(kind,Q,REQ,[RET],dict(tool='search',arguments=dict(query=query)),dict(candidates=[])),None
    if kind=='browse':
        action=dict(tool='browse',arguments=dict(candidate_id='guide' if good else 'theme',focus='Find recovery permissions and limitations.' if good else 'Find dark theme colors.'))
        return pack(kind,Q,REQ,[RET],action,dict(candidates=CANDIDATES)),None
    if kind=='state':
        updates=[dict(id='P1',status='partial' if good else 'direct',evidence_ids=['POOL']),dict(id='P2',status='direct',evidence_ids=['SAFE'])]
        return pack(kind,SQ,SREQ,SEV,dict(updates=updates),dict(items=ITEMS)),None
    if kind=='evidence':
        return pack(kind,SQ,SREQ,SEV if good else SEV[:1],''),None
    raise ValueError(kind)

def cases():
    return [(f'{kind}_{v}',*case(kind,v)) for kind in ('checklist','search','browse','state','stop','evidence') for v in ('good','bad')]
