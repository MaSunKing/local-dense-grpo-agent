"""Unified pure planning/validation entry. No network, collector invention or training."""
import copy,hashlib,json
from pathlib import Path
from engine import ROOT,prepare,aggregate,judgment_key,digest,final_engine,mapping_preview,need
from active_policy_v11 import PROMPTS,STATE_COMBINED_PROMPT
from state_guard import guarded_aggregate

def profile(config,task_modes=None):
    need(set(config)=={'model','temperature','max_tokens','response_format'},'unsupported or unbound effective model parameters')
    need(isinstance(config['model'],str) and config['model'].strip(),'model')
    need(type(config['temperature']) in (int,float) and config['temperature']==0,'frozen temperature')
    need(type(config['max_tokens']) is int and config['max_tokens']>0,'output cap')
    need(config['response_format']=={'type':'json_object'},'JSON response contract')
    paths=[Path(__file__)]+[Path(__file__).with_name(n) for n in ('engine.py','active_policy.py','active_policy_v2.py','active_policy_v3.py','active_policy_v4.py','active_policy_v5.py','active_policy_v6.py','active_policy_v7.py','active_policy_v8.py','active_policy_v9.py','active_policy_v10.py','active_policy_v11.py','state_guard.py','prompts.py','evidence_gain.py')]+[ROOT/x for x in ('tiered_judge_v2/isolated.py','tiered_judge_v1/reward_policy.py','tiered_judge_v1/policy.json','tiered_judge_v1/final_contract.py','general_final_v6/contract.py','fixed/guards.py','fixed/final_binding.py','fixed/response_validator.py')]
    paths.append(Path(__file__).with_name('input_contract.py'))
    paths.append(ROOT/'private_judge_addendum.md')
    paths.append(ROOT.parent/'shared/gain_contract.py')
    paths.append(ROOT.parent/'shared/contract_retry.py')
    package_root=ROOT.parent
    data=dict(config=config,frozen_task_modes=task_modes or {},stage_prompts=PROMPTS,final_prompts=final_engine.PROMPTS,source_hashes={p.relative_to(package_root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    return data|dict(scorer=digest(data))

def plan(payload,config,collector_lookup=None,task_modes=None):
    frozen=profile(config,task_modes);bundle=prepare(payload,collector_lookup,task_modes);scorer=frozen['scorer'];tasks={}
    views=bundle['final']['views'] if bundle['kind']=='final' else bundle['views']
    if bundle['kind']=='state' and set(views)=={'state_truth','state_citation'}:
        # The Judge never sees the claimed status. Both nested outputs remain
        # separately validated and separately aggregated after transport.
        citation_view=copy.deepcopy(views['state_citation'])
        for item in citation_view['items']:
            item.pop('claimed_status')
        public_view={'state_truth_view':copy.deepcopy(views['state_truth']),
                     'state_citation_view':citation_view}
        prompt=STATE_COMBINED_PROMPT+'\n\n'+(ROOT/'private_judge_addendum.md').read_text(encoding='utf-8')
        key=judgment_key(scorer,'state_combined',public_view)
        request=config|dict(messages=[dict(role='system',content=prompt),
                                     dict(role='user',content=json.dumps(public_view,ensure_ascii=False))])
        tasks['state_combined']=dict(key=key,request=request)
        return dict(profile=frozen,bundle=bundle,tasks=tasks,training_ready=False)
    for task,view in views.items():
        prompt=final_engine.PROMPTS[task] if bundle['kind']=='final' else PROMPTS[task]
        prompt=prompt+'\n\n'+(ROOT/'private_judge_addendum.md').read_text(encoding='utf-8')
        key=final_engine.digest(dict(scorer=scorer,dimension=task,view=view)) if bundle['kind']=='final' else judgment_key(scorer,task,view)
        request=config|dict(messages=[dict(role='system',content=prompt),dict(role='user',content=json.dumps(view,ensure_ascii=False))])
        tasks[task]=dict(key=key,request=request)
    return dict(profile=frozen,bundle=bundle,tasks=tasks,training_ready=False)

def finish(planned,results,collector_lookup=None,task_modes=None):
    expected=plan(planned['bundle']['payload'],planned['profile']['config'],collector_lookup,task_modes)
    need(planned==expected,'plan/config/source binding changed')
    need(set(results)==set(planned['tasks']),'exact result task set')
    if set(planned['tasks'])=={'state_combined'}:
        combined=results['state_combined']
        validate_task_result(planned,'state_combined',combined)
        views=planned['bundle']['views']
        judgments={task:dict(key=judgment_key(planned['profile']['scorer'],task,views[task]),
                             result=combined[task])
                   for task in ('state_truth','state_citation')}
    else:
        judgments={task:dict(key=planned['tasks'][task]['key'],result=result) for task,result in results.items()}
    report=guarded_aggregate(planned['bundle'],judgments,planned['profile']['scorer'],collector_lookup,task_modes)
    return dict(report=report,mapping_preview=mapping_preview(planned['bundle']['kind'],report),semantic_review='PENDING',training_ready=False,production_switched=False)

def validate_task_result(planned,task,result):
    """Validate one returned task before bounded contract retry or final assembly."""
    need(task in planned['tasks'],'unplanned Judge task')
    views=planned['bundle']['final']['views'] if planned['bundle']['kind']=='final' else planned['bundle']['views']
    if task=='state_combined':
        need(planned['bundle']['kind']=='state' and set(result)=={'state_truth','state_citation'},
             'exact combined State result fields required')
        engine=__import__('engine')
        return {
            'state_truth':engine.validate('state_truth',views['state_truth'],result['state_truth']),
            'state_citation':engine.validate('state_citation',views['state_citation'],result['state_citation']),
        }
    need(task in views,'Judge task/view mismatch')
    return final_engine.validate(task,views[task],result) if planned['bundle']['kind']=='final' else __import__('engine').validate(task,views[task],result)
