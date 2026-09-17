import copy
import json
from pathlib import Path
import tempfile
import unittest
import sys

from core import compile_batch,compile_batch_report,digest,validate,SCHEMA
from judge import PrivateJudge,coverage_score,scalar_score,validate_packet,stop_boundary_event,evidence_gain_event,observed_dimensions,response_content
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'shared'))
from gain_contract import (canonical_action,canonical_evidence,digest as shared_digest,
    make_coverage_receipt,make_evidence_transition,make_gain_disposition,
    record_target_digest,task_scope,
    validate_termination_event)

_config_path=Path(__file__).with_name('config.json')
if not _config_path.is_file():
    _config_path=Path(__file__).with_name('config.review_fixture.json')
CONFIG=json.loads(_config_path.read_text(encoding='utf-8'))


def compile_fixture(batch,config=CONFIG,authorities=None):
    authorities=authorities or batch
    return compile_batch(batch,config,collector_lookup=lambda key:
                         copy.deepcopy(authorities.get('private_tool_executions',{}).get(key)),
                         score_lookup=lambda key:
                         copy.deepcopy(authorities.get('private_stage_scores',{}).get(key)),
                         gain_lookup=lambda key:
                         copy.deepcopy(authorities.get('private_evidence_scores',{}).get(key)),
                         termination_lookup=lambda key:
                         copy.deepcopy(authorities.get('private_terminations',{}).get(key)),
                         execution_manifest_lookup=lambda rollout_id:
                          copy.deepcopy(authorities.get('private_execution_manifests',{}).get(rollout_id)),
                          violation_lookup=lambda key:
                          copy.deepcopy(authorities.get('private_violations',{}).get(key)),
                           gain_disposition_lookup=lambda key:
                           copy.deepcopy(authorities.get('private_gain_dispositions',{}).get(key)),
                           coverage_receipt_lookup=lambda key:
                           copy.deepcopy(authorities.get('private_coverage_receipts',{}).get(key)),
                           evidence_transition_lookup=lambda key:
                           copy.deepcopy(authorities.get('private_evidence_transitions',{}).get(key)))


def fixture():
    identity={k:'a'*64 for k in ('policy','runtime','tokenizer','scorer')}
    identity['reward_config']=digest(CONFIG)
    identity['task_modes']=digest({})
    requirements=[dict(id='Q1',description='Primary outcome?',weight=.7),
                  dict(id='Q2',description='Safety outcome?',weight=.3)]
    scope=task_scope('Question?',requirements,[],'evidence_grounded')
    rolls=[]
    for i,score in enumerate((1.,.7,.3,0.)):
        records=[]
        for stage,ch in (('checklist','checklist'),('decision','tool'),('state','state'),('stop','stop'),('final','final')):
            rec=dict(id=stage,stage=stage,policy=identity['policy'],input_ids=[1,2],output_ids=[3,4],
                     behavior_logps=[-1.,-2.],token_digest=digest([[1,2],[3,4]]),
                     sampling=dict(temperature=.8,top_p=1.,top_k=0,grammar=None,
                                   implementation='explicit_multinomial_v1',
                                   distribution='temperature_full_support_v1'),
                     channels={ch:dict(indices=[0,1],local_reward=score)},task_events=[])
            if stage=='decision':
                rec['channels'][ch]['local_reward']=None
                rec['channels'][ch]['indices']=[0]
                rec['channels']['browse_source_focus']=dict(indices=[1],local_reward=score)
                rec['task_events']=[]
            if stage=='stop':
                rec['channels'][ch]['local_reward']=None
            if stage=='final':
                rec['channels'][ch]['local_reward']=None
            records.append(rec)
        rolls.append(dict(question_id='q',rollout_id=f'r{i}',policy=identity['policy'],
                          records=records,final_reward=score,
                          task_scope_digest=scope))
    batch=dict(schema=SCHEMA,data_split='train',identity=identity,context_limit=8192,rollouts=rolls,
               task_scopes={'q':dict(question='Question?',requirements=requirements,
                   constraints=[],task_mode='evidence_grounded',task_scope_digest=scope)},
               private_stop_scores={},private_evidence_scores={},
               private_tool_executions={},private_stage_scores={},
               private_terminations={},private_execution_manifests={},
                private_violations={},private_gain_dispositions={},
                private_coverage_receipts={},private_evidence_transitions={})
    for roll,score in zip(rolls,(1.,.7,.3,0.)):
        attach_gain(batch,roll,score);attach_stop(batch,roll,.5)
        for rec in roll['records']:
            for channel,row in rec['channels'].items():
                if row.get('local_reward') is not None:
                    attach_stage_score(batch,roll,rec,channel,row['local_reward'])
        final=next(r for r in roll['records'] if r['stage']=='final')
        roll['final_reward_proof']=attach_stage_score(
            batch,roll,final,'final',roll['final_reward'],channel_row=False)
        attach_execution_manifest(batch,roll)
    return batch


def attach_stage_score(batch,roll,rec,channel,score,channel_row=True):
    stage={'search_query':'search','browse_source_focus':'browse'}.get(channel,channel)
    target=record_target_digest(question_id=roll['question_id'],rollout_id=roll['rollout_id'],
        record=rec,channel=channel,task_scope_digest=roll['task_scope_digest'])
    view_digest=digest([roll['rollout_id'],rec['id'],channel,'judge-view'])
    artifact=dict(version='private_stage_score_v2',question_id=roll['question_id'],
        rollout_id=roll['rollout_id'],record_id=rec['id'],channel=channel,score=score,
        scorer=batch['identity']['scorer'],reward_config=batch['identity']['reward_config'],
        task_scope_digest=roll['task_scope_digest'],policy=roll['policy'],
        token_digest=rec['token_digest'],target_digest=target,judge_view_digest=view_digest,
        report=dict(stage=stage,score=score,status='observed',contract_valid=True,
                    scorer=batch['identity']['scorer'],binding_digest=view_digest))
    key=digest(artifact);batch['private_stage_scores'][key]=artifact
    proof=dict(kind='private_stage_score_v2',artifact_key=key)
    if channel_row:rec['channels'][channel]['local_reward_proof']=proof
    return proof


def attach_gain(batch,roll,score,config=CONFIG):
    before=next(r for r in roll['records'] if r['stage']=='decision')
    after=next(r for r in roll['records'] if r['stage']=='state')
    before['evidence_snapshot']=dict(step=1,digest=digest([]))
    evidence_id='E-'+roll['rollout_id']
    evidence=[dict(source_id=evidence_id,text='Observed result for '+roll['rollout_id'])]
    after['evidence_snapshot']=dict(step=2,digest=digest(canonical_evidence(evidence)))
    requirements=batch['task_scopes'][roll['question_id']]['requirements']
    base=dict(question='Question?',requirements=requirements,constraints=[],source_headers={})
    before_view=base|dict(evidence=[]);after_view=base|dict(evidence=evidence)
    before_rows=[dict(id=r['id'],status='unknown',evidence_ids=[],reason='not opened') for r in requirements]
    after_rows=[]
    selected={1.:{0,1},.7:{0},.3:{1},0.:set()}[score]
    for index,r in enumerate(requirements):
        direct=index in selected
        after_rows.append(dict(id=r['id'],status='direct' if direct else 'unknown',
            evidence_ids=[evidence_id] if direct else [],reason='opened' if direct else 'not covered'))
    scorer=batch['identity']['scorer']
    raw=json.dumps({'tool':'browse','arguments':{'candidate_id':'candidate','focus':'outcomes'}})
    action=canonical_action(raw)
    execution_core=dict(version='trusted_tool_execution_v1',question_id=roll['question_id'],
        rollout_id=roll['rollout_id'],record_id=before['id'],policy=roll['policy'],
        token_digest=before['token_digest'],task_scope_digest=roll['task_scope_digest'],
        capture=dict(capture_id='cap-'+roll['rollout_id'],capture_sha256='c'*64,
            parser_version='json_tool_action_v1',raw_completion=raw,action=action,
            action_digest=digest(action)),
        response=dict(payload=dict(tool='browse',evidence=canonical_evidence(evidence)),
                      payload_digest=digest(dict(tool='browse',evidence=canonical_evidence(evidence)))),
        status='succeeded')
    execution=copy.deepcopy(execution_core);execution['tool_execution_id']=digest(execution_core)
    before['tool_execution_id']=execution['tool_execution_id']
    batch['private_tool_executions'][execution['tool_execution_id']]=execution
    transition=make_evidence_transition(
        question_id=roll['question_id'],rollout_id=roll['rollout_id'],
        browse_record_id=before['id'],tool_execution=execution,policy=roll['policy'],
        token_digest=before['token_digest'],task_scope_digest=roll['task_scope_digest'],
        before_snapshot=before['evidence_snapshot'],after_snapshot=after['evidence_snapshot'],
        before_evidence=[],after_evidence=evidence)
    batch['private_evidence_transitions'][execution['tool_execution_id']]=transition
    before['task_events'].append(dict(id='cost',kind='tool_cost',value=-.01,
        tool_execution_id=execution['tool_execution_id'],proof=dict(
            kind='trusted_tool_execution_v1',tool_execution_id=execution['tool_execution_id'])))
    transitions=[]
    for index,r in enumerate(requirements):
        direct=index in selected
        transitions.append(dict(id=r['id'],before='unknown',after='direct' if direct else 'unknown',
            delta=1. if direct else 0.,new_evidence_ids=[evidence_id] if direct else []))
    disposition='observed_positive' if score>0 else 'observed_zero'
    artifact=dict(version='browse_evidence_gain_v4',scorer=scorer,reward_config=digest(config),
        task_modes_digest=digest({}),task_scope_digest=roll['task_scope_digest'],
        tool_execution=execution,
        binding=dict(question_id=roll['question_id'],rollout_id=roll['rollout_id'],
            browse_record_id=before['id'],after_record_id=after['id'],policy=roll['policy'],
            before_snapshot=before['evidence_snapshot'],after_snapshot=after['evidence_snapshot'],
            selected_candidate_id='candidate',new_evidence_ids=[evidence_id],
            tool_execution_id=execution['tool_execution_id'],task_scope_digest=roll['task_scope_digest']),
        before_view=before_view,after_view=after_view,
        before_result=dict(coverage=before_rows),after_result=dict(coverage=after_rows),
        before_key=digest(dict(scorer=scorer,task='coverage',view=before_view)),
        after_key=digest(dict(scorer=scorer,task='coverage',view=after_view)),
        before_score=0.,after_score=score,transitions=transitions,
        disposition=disposition,delta=score)
    artifact_key=digest(artifact)
    for coverage_key,result in ((artifact['before_key'],artifact['before_result']),
                                (artifact['after_key'],artifact['after_result'])):
        batch['private_coverage_receipts'][coverage_key]=make_coverage_receipt(
            scorer=scorer,task_scope_digest=roll['task_scope_digest'],
            coverage_key=coverage_key,result=result)
    batch['private_evidence_scores'][artifact_key]=artifact
    batch['private_gain_dispositions'][execution['tool_execution_id']]=make_gain_disposition(
        status=disposition,artifact_key=artifact_key,question_id=roll['question_id'],
        rollout_id=roll['rollout_id'],browse_record_id=before['id'],
        after_record_id=after['id'],tool_execution_id=execution['tool_execution_id'],
        policy=roll['policy'],token_digest=before['token_digest'],
        task_scope_digest=roll['task_scope_digest'])
    event=None
    if score>0:
        event=evidence_gain_event(artifact,event_id='gain',config=config,scorer=scorer,
            coverage_receipt_lookup=lambda key:batch['private_coverage_receipts'].get(key))
        before['task_events'].insert(0,event)
    return event


def attach_stop(batch,roll,score,config=CONFIG):
    packet=JudgeTests().packet('stop');rec=next(r for r in roll['records'] if r['stage']=='stop')
    evidence=canonical_evidence(packet['evidence'])
    rec['evidence_snapshot']=dict(step=2,digest=shared_digest(evidence))
    binding=dict(question_id=roll['question_id'],rollout_id=roll['rollout_id'],record_id=rec['id'],
        token_digest=rec['token_digest'],policy=roll['policy'],
        task_scope_digest=roll['task_scope_digest'],evidence_snapshot=rec['evidence_snapshot'])
    termination_core=dict(version='trusted_termination_event_v1',type='active',binding=binding,
        evidence=evidence,source_headers={},stop_context=copy.deepcopy(packet['stop_context']),
        capture=dict(capture_id='stop-'+roll['rollout_id'],capture_sha256='d'*64,
            question_id=roll['question_id'],rollout_id=roll['rollout_id'],record_id=rec['id'],
            policy=roll['policy'],input_ids=rec['input_ids'],output_ids=rec['output_ids'],
            raw_completion='stop'))
    termination=copy.deepcopy(termination_core);termination['event_id']=shared_digest(termination_core)
    batch['private_terminations'][termination['event_id']]=termination
    scorer=batch['identity']['scorer']
    target=record_target_digest(question_id=roll['question_id'],rollout_id=roll['rollout_id'],
        record=rec,channel='stop',task_scope_digest=roll['task_scope_digest'])
    view_digest=digest([roll['rollout_id'],rec['id'],'stop','judge-view'])
    artifact=dict(version='private_stop_score_v2',question_id=roll['question_id'],
        rollout_id=roll['rollout_id'],record_id=rec['id'],score=score,scorer=scorer,
        reward_config=digest(config),task_scope_digest=roll['task_scope_digest'],
        policy=roll['policy'],token_digest=rec['token_digest'],target_digest=target,
        judge_view_digest=view_digest,termination_event_id=termination['event_id'],
        report=dict(stage='stop',score=score,status='observed',contract_valid=True,
                    scorer=scorer,binding_digest=view_digest))
    key=digest(artifact);batch['private_stage_scores'][key]=artifact
    expected=dict(question_id=roll['question_id'],rollout_id=roll['rollout_id'],
        record_id=rec['id'],policy=roll['policy'],token_digest=rec['token_digest'],
        task_scope_digest=roll['task_scope_digest'],target_digest=target)
    event=stop_boundary_event(artifact,termination,event_id='boundary',config=config,
        scorer=scorer,expected=expected)
    rec['task_events']=[event]
    return event


def attach_execution_manifest(batch,roll):
    core=dict(version='trusted_execution_manifest_v1',question_id=roll['question_id'],
        rollout_id=roll['rollout_id'],execution_ids=sorted(
            rec['tool_execution_id'] for rec in roll['records']
            if rec.get('tool_execution_id') is not None))
    manifest=copy.deepcopy(core);manifest['manifest_id']=shared_digest(core)
    batch['private_execution_manifests'][roll['rollout_id']]=manifest
    return manifest


class CreditTests(unittest.TestCase):
    def test_compile_report_exposes_group_gate(self):
        batch=fixture();authorities=batch
        kwargs=dict(collector_lookup=lambda key:copy.deepcopy(authorities['private_tool_executions'].get(key)),
            score_lookup=lambda key:copy.deepcopy(authorities['private_stage_scores'].get(key)),
            gain_lookup=lambda key:copy.deepcopy(authorities['private_evidence_scores'].get(key)),
            termination_lookup=lambda key:copy.deepcopy(authorities['private_terminations'].get(key)),
            execution_manifest_lookup=lambda key:copy.deepcopy(authorities['private_execution_manifests'].get(key)),
            violation_lookup=lambda key:copy.deepcopy(authorities['private_violations'].get(key)),
            gain_disposition_lookup=lambda key:copy.deepcopy(authorities['private_gain_dispositions'].get(key)),
            coverage_receipt_lookup=lambda key:copy.deepcopy(authorities['private_coverage_receipts'].get(key)),
            evidence_transition_lookup=lambda key:copy.deepcopy(authorities['private_evidence_transitions'].get(key)))
        ready=compile_batch_report(batch,CONFIG,**kwargs)
        self.assertEqual(ready['status'],'reward_compiled')
        self.assertTrue(ready['reward_export_authorized'])
        batch=fixture();batch['rollouts'][0]['final_reward']=None
        blocked=compile_batch_report(batch,CONFIG,**kwargs)
        self.assertEqual(blocked['status'],'needs_attention')
        self.assertFalse(blocked['reward_export_authorized'])
        self.assertIsNone(blocked['rows'])
    def test_all_four_all_stages(self):
        rows=compile_fixture(fixture(),CONFIG)
        self.assertEqual(len(rows),24)
        self.assertEqual({r['rollout_id'] for r in rows},{'r0','r1','r2','r3'})
        for name in ('checklist','tool','browse_source_focus','state','final'):
            channel=[r for r in rows if r['channel']==name]
            self.assertGreater(channel[0]['advantage'],0)
            self.assertLess(channel[-1]['advantage'],0)
        self.assertTrue(all(r['advantage']==0 for r in rows if r['channel']=='stop'))

    def test_final_once(self):
        rows=compile_fixture(fixture(),CONFIG)
        final=next(r for r in rows if r['channel']=='final')
        self.assertAlmostEqual(final['advantage'],1-(.7+.3+0)/3)

    def test_unknown_channels_independent(self):
        batch=fixture();batch['rollouts'][0]['final_reward']=None
        with self.assertRaisesRegex(ValueError,'required pending core'):
            compile_fixture(batch,CONFIG)
        batch=fixture();next(e for e in batch['rollouts'][0]['records'][1]['task_events'] if e['kind']=='tool_cost')['value']=None
        with self.assertRaises(ValueError):compile_fixture(batch,CONFIG)

    def test_required_local_core_blocks_whole_group(self):
        batch=fixture()
        state=next(r for r in batch['rollouts'][0]['records'] if r['stage']=='state')
        state['channels']['state']['local_reward']=None
        state['channels']['state'].pop('local_reward_proof')
        with self.assertRaisesRegex(ValueError,'required pending core'):
            compile_fixture(batch,CONFIG)

    def test_unscored_stop_remains_allowed(self):
        rows=compile_fixture(fixture(),CONFIG)
        self.assertEqual(len([r for r in rows if r['channel']=='stop']),4)

    def test_no_state_revenue(self):
        batch=fixture();event=copy.deepcopy(batch['rollouts'][0]['records'][1]['task_events'][0]);event['id']='state'
        batch['rollouts'][0]['records'][2]['task_events']=[event]
        with self.assertRaises(ValueError):validate(batch)

    def test_state_count_normalization_and_whole_rollout_baseline(self):
        batch=fixture();extra=copy.deepcopy(batch['rollouts'][0]['records'][2]);extra['id']='state2'
        extra['channels']['state']['local_reward']=0.
        attach_stage_score(batch,batch['rollouts'][0],extra,'state',0.)
        batch['rollouts'][0]['records'].insert(3,extra)
        rows=compile_fixture(batch,CONFIG)
        own=[r for r in rows if r['rollout_id']=='r0' and r['channel']=='state']
        self.assertAlmostEqual(sum(r['loss_weight'] for r in own),1.)
        self.assertAlmostEqual(own[0]['local_advantage'],1-(.7+.3)/3)

    def test_hard_lineage_failures(self):
        for mutation in ('identity','scope','event','tokens','grammar'):
            batch=fixture();rec=batch['rollouts'][0]['records'][1]
            if mutation=='identity':rec['policy']='b'*64
            elif mutation=='scope':rec['channels']['search_query']=dict(indices=[0],local_reward=0.)
            elif mutation=='event':rec['task_events'].append(copy.deepcopy(rec['task_events'][0]))
            elif mutation=='tokens':rec['output_ids'][0]=99
            else:rec['sampling']['grammar']={}
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):validate(batch)

    def test_business_event_cardinality_and_cost_completeness(self):
        batch=fixture();rec=batch['rollouts'][0]['records'][1]
        gain=copy.deepcopy(next(e for e in rec['task_events'] if e['kind']=='evidence_gain'))
        gain['id']='renamed-duplicate';rec['task_events'].append(gain)
        with self.assertRaises(ValueError):validate(batch)
        batch=fixture();rec=batch['rollouts'][0]['records'][1]
        cost=copy.deepcopy(next(e for e in rec['task_events'] if e['kind']=='tool_cost'))
        cost['id']='renamed-cost';rec['task_events'].append(cost)
        with self.assertRaises(ValueError):validate(batch)
        batch=fixture();rec=batch['rollouts'][0]['records'][1]
        rec['task_events']=[e for e in rec['task_events'] if e['kind']!='tool_cost']
        with self.assertRaises(ValueError):validate(batch)

    def test_group_scope_and_private_local_receipt_required(self):
        batch=fixture();batch['rollouts'][0]['task_scope_digest']='f'*64
        with self.assertRaises(ValueError):validate(batch)
        batch=fixture();channel=batch['rollouts'][0]['records'][0]['channels']['checklist']
        channel['local_reward']=123.;channel.pop('local_reward_proof')
        with self.assertRaises(ValueError):validate(batch)
        with self.assertRaises(ValueError):compile_batch(fixture(),CONFIG)

    def test_tool_execution_response_tamper_rejected(self):
        batch=fixture();rec=batch['rollouts'][0]['records'][1]
        execution=batch['private_tool_executions'][rec['tool_execution_id']]
        execution['response']['payload']['evidence'][0]['text']='fabricated'
        with self.assertRaises(ValueError):compile_fixture(batch,CONFIG)

    def test_v15_authority_bindings_reject_batch_only_rewrites(self):
        for mutation in ('checklist_tokens','state_input','final_tokens','state_snapshot','channel_indices'):
            batch=fixture();authorities=copy.deepcopy(batch)
            stage={'checklist_tokens':'checklist','state_input':'state',
                   'final_tokens':'final','state_snapshot':'state',
                   'channel_indices':'checklist'}[mutation]
            rec=next(row for row in batch['rollouts'][0]['records'] if row['stage']==stage)
            if mutation=='state_input':rec['input_ids']=[55,66]
            elif mutation in {'checklist_tokens','final_tokens'}:rec['output_ids']=[55,66]
            elif mutation=='state_snapshot':rec['evidence_snapshot']={'step':9,'digest':'b'*64}
            else:rec['channels']['checklist']['indices']=[1]
            if mutation in {'checklist_tokens','state_input','final_tokens'}:
                rec['token_digest']=digest([rec['input_ids'],rec['output_ids']])
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                compile_fixture(batch,CONFIG,authorities)

        batch=fixture();authorities=copy.deepcopy(batch)
        rec=next(row for row in batch['rollouts'][0]['records'] if row['stage']=='decision')
        del rec['channels']['tool'];rec.pop('tool_execution_id');rec['task_events']=[]
        with self.assertRaises(ValueError):compile_fixture(batch,CONFIG,authorities)

        batch=fixture();rec=next(row for row in batch['rollouts'][0]['records'] if row['stage']=='decision')
        rec['task_events'].append(dict(id='penalty',kind='policy_penalty',value=-.1,
            tool_execution_id=rec['tool_execution_id']))
        with self.assertRaises(ValueError):validate(batch)

    def test_v15_gain_and_stop_are_resolver_owned(self):
        batch=fixture();authorities=copy.deepcopy(batch)
        rec=next(row for row in batch['rollouts'][2]['records'] if row['stage']=='decision')
        index=next(i for i,event in enumerate(rec['task_events']) if event['kind']=='evidence_gain')
        old=rec['task_events'][index];artifact=copy.deepcopy(
            batch['private_evidence_scores'][old['proof']['artifact_key']])
        for row in artifact['after_result']['coverage']:
            row.update(status='direct',evidence_ids=['E'],reason='batch-only rewrite')
        for row in artifact['transitions']:
            row.update(after='direct',delta=1.,new_evidence_ids=['E'])
        artifact['after_score']=1.;artifact['delta']=1.
        with self.assertRaises(ValueError):
            evidence_gain_event(artifact,event_id=old['id'],config=CONFIG,
                scorer=batch['identity']['scorer'],coverage_receipt_lookup=lambda key:
                batch['private_coverage_receipts'].get(key))

        batch=fixture();authorities=copy.deepcopy(batch)
        stop=next(row for row in batch['rollouts'][0]['records'] if row['stage']=='stop')
        stop['task_events'][0]['value']=.1
        with self.assertRaises(ValueError):compile_fixture(batch,CONFIG,authorities)

    def test_v15_policy_penalty_requires_trusted_violation(self):
        batch=fixture();roll=batch['rollouts'][0]
        rec=next(row for row in roll['records'] if row['stage']=='decision')
        core=dict(version='trusted_policy_violation_v1',question_id=roll['question_id'],
            rollout_id=roll['rollout_id'],record_id=rec['id'],
            tool_execution_id=rec['tool_execution_id'],policy=roll['policy'],
            token_digest=rec['token_digest'],task_scope_digest=roll['task_scope_digest'],
            violation_type='invalid_tool_policy')
        violation=copy.deepcopy(core);violation['violation_event_id']=shared_digest(core)
        batch['private_violations'][violation['violation_event_id']]=violation
        rec['task_events'].append(dict(id='penalty',kind='policy_penalty',
            value=-CONFIG['task']['policy_penalty'],tool_execution_id=rec['tool_execution_id'],
            proof=dict(kind='trusted_policy_violation_v1',
                       violation_event_id=violation['violation_event_id'])))
        compile_fixture(batch,CONFIG)
        authorities=copy.deepcopy(batch)
        batch['private_violations'][violation['violation_event_id']]['violation_type']='rewritten'
        compile_fixture(batch,CONFIG,authorities)
        rec['task_events'][-1]['proof']['violation_event_id']='forged'
        with self.assertRaises(ValueError):compile_fixture(batch,CONFIG,authorities)


class JudgeTests(unittest.TestCase):
    def packet(self,kind='state'):
        req=[dict(id='R1',description='Question?',weight=1.)]
        packet=dict(kind=kind,question='Question?',requirements=req,requirements_digest=digest(req),
                    evidence=[dict(source_id='D#c1',text='Original evidence.')],output='Actual model output')
        if kind in {'state','stop'}:
            packet.update(policy_items=[dict(id='P1',description='Actual policy subquestion')],
                previous_state=[dict(id='P1',status='partial',evidence_ids=['D#c1'])],
                evidence_snapshot=dict(step=2,digest=digest(packet['evidence'])),new_evidence_ids=['D#c1'])
        if kind=='stop':packet['stop_context']=dict(voluntary=True,remaining_budget=3,
                         available_candidates=[],failed_reads=[],task_constraints=[])
        if kind=='stop':packet['binding']=dict(question_id='q',rollout_id='r0',record_id='stop',
            token_digest=digest([[1,2],[3,4]]),policy='a'*64,evidence_snapshot=packet['evidence_snapshot'])
        return packet

    def test_cache_and_private_prompt(self):
        calls=[]
        def transport(model,messages):
            calls.append(messages)
            return json.dumps(dict(scores=dict(status_accuracy=1.,citation_support=1.,scope_fidelity=1.),coverage=[],reason='ok'))
        with tempfile.TemporaryDirectory() as d:
            judge=PrivateJudge(model='test',cache=d,transport=transport)
            result=judge.score(self.packet());judge.score(self.packet())
            self.assertEqual(len(calls),1);self.assertEqual(scalar_score(result),1.)
            self.assertIn('JSON',calls[0][0]['content'])

    def test_failure_not_zero_or_success_cache(self):
        def transport(*args):raise TimeoutError('mock')
        with tempfile.TemporaryDirectory() as d:
            result=PrivateJudge(model='test',cache=d,transport=transport).score(self.packet())
            self.assertEqual(result['status'],'pending');self.assertIsNone(scalar_score(result))
            self.assertFalse(list(Path(d).glob('*.json')))

    def test_coverage_requires_real_refs(self):
        for ref in ('invented','D#c1'):
            def transport(*args):return json.dumps(dict(scores={},coverage=[dict(id='R1',status='direct',evidence_ids=[ref])],reason='ok'))
            with tempfile.TemporaryDirectory() as d:
                result=PrivateJudge(model='test',cache=d,transport=transport).score(self.packet('evidence'))
                self.assertEqual(coverage_score(self.packet()['requirements'],result,CONFIG),None if ref=='invented' else 1.)

    def test_all_null_retries_and_is_not_success_cache(self):
        calls=[]
        def transport(*args):
            calls.append(1)
            v=None if len(calls)<=2 else 1.
            return json.dumps(dict(scores=dict(status_accuracy=v,citation_support=v,scope_fidelity=v),coverage=[],reason='test'))
        with tempfile.TemporaryDirectory() as d:
            judge=PrivateJudge(model='test',cache=d,transport=transport)
            self.assertEqual(judge.score(self.packet())['status'],'pending')
            self.assertFalse(list(Path(d).glob('*.json')))
            self.assertEqual(judge.score(self.packet())['status'],'pending')
            self.assertEqual(judge.score(self.packet())['status'],'observed')
            self.assertEqual(len(calls),3)

    def test_partial_dimensions_remain_available_but_not_fake_scalar(self):
        def transport(*args):return json.dumps(dict(scores=dict(status_accuracy=None,citation_support=0.,scope_fidelity=1.),coverage=[],reason='test'))
        with tempfile.TemporaryDirectory() as d:
            result=PrivateJudge(model='test',cache=d,transport=transport).score(self.packet())
            self.assertEqual(result['status'],'partially_observed')
            self.assertIsNone(scalar_score(result))
            self.assertEqual(observed_dimensions(result),dict(citation_support=0.,scope_fidelity=1.))

    def test_state_requires_policy_mapping_and_snapshot(self):
        for key in ('policy_items','previous_state','evidence_snapshot'):
            packet=self.packet();packet.pop(key)
            with self.assertRaises(ValueError):validate_packet(packet)
        packet=self.packet();packet['previous_state'][0]['id']='R1'
        with self.assertRaises(ValueError):validate_packet(packet)

    def test_stop_boundary_changes_advantage_with_equal_final(self):
        batch=fixture();packet=self.packet('stop')
        for i,roll in enumerate(batch['rollouts']):
            score=(1.,.7,.3,0.)[i];roll['final_reward']=.5
            final=next(r for r in roll['records'] if r['stage']=='final')
            roll['final_reward_proof']=attach_stage_score(
                batch,roll,final,'final',.5,channel_row=False)
            event=attach_stop(batch,roll,score)
        rows=[r for r in compile_fixture(batch,CONFIG) if r['channel']=='stop']
        self.assertGreater(rows[0]['advantage'],0);self.assertLess(rows[-1]['advantage'],0)
        self.assertAlmostEqual(rows[0]['local_score'],.1)
        self.assertEqual(rows[0]['final_advantage'],0.)
        batch['rollouts'][0]['records'][3]['task_events'].append(copy.deepcopy(event))
        batch['rollouts'][0]['records'][3]['task_events'][-1]['id']='other'
        with self.assertRaises(ValueError):validate(batch)

    def test_stop_not_status_count_or_forced_termination(self):
        # Both labels can be submitted for evaluation; neither is an auto-reward.
        for label in ('partial','direct','unknown'):
            packet=self.packet('stop');packet['previous_state'][0]['status']=label
            validate_packet(packet)
        packet['stop_context']['voluntary']=False
        with self.assertRaises(ValueError):validate_packet(packet)

    def test_terminal_order_and_ignored_reward_rejected(self):
        batch=fixture();batch['rollouts'][0]['records'].reverse()
        with self.assertRaises(ValueError):validate(batch)
        batch=fixture();batch['rollouts'][0]['records'][3]['channels']['stop']['local_reward']=1.
        with self.assertRaises(ValueError):validate(batch)

    def test_stop_config_scorer_and_snapshot_drift_rejected(self):
        for mutation in ('scale','scorer','snapshot','rollout','value','receipt_key','receipt_result'):
            batch=fixture();event=batch['rollouts'][0]['records'][3]['task_events'][0]
            artifact=batch['private_stage_scores'][event['proof']['artifact_key']]
            termination=batch['private_terminations'][event['proof']['termination_event_id']]
            config=copy.deepcopy(CONFIG)
            if mutation=='scale':config['task']['stop_scale']=1.
            elif mutation=='scorer':event['proof']['scorer']='b'*64
            elif mutation=='snapshot':batch['rollouts'][0]['records'][3]['evidence_snapshot']['step']=999
            elif mutation=='rollout':termination['binding']['rollout_id']='r3'
            elif mutation=='value':event['value']=.7
            elif mutation=='receipt_key':event['proof']['artifact_key']='x'
            else:artifact['report']['score']=1.
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):compile_fixture(batch,config)

    def test_stop_capture_sha256_must_be_canonical(self):
        batch=fixture();roll=batch['rollouts'][0]
        termination=copy.deepcopy(next(iter(batch['private_terminations'].values())))
        termination['capture']['capture_sha256']='not-a-sha256'
        core=copy.deepcopy(termination);core.pop('event_id')
        termination['event_id']=shared_digest(core)
        with self.assertRaisesRegex(ValueError,'termination capture sha256'):
            validate_termination_event(termination)

    def test_evidence_gain_artifact_tamper_rejected(self):
        for mutation in ('value','scorer','snapshot','result','missing'):
            batch=fixture();roll=batch['rollouts'][0];rec=roll['records'][1]
            event=next(e for e in rec['task_events'] if e['kind']=='evidence_gain')
            key=event['proof']['artifact_key'];artifact=batch['private_evidence_scores'][key]
            if mutation=='value':event['value']+=.1
            elif mutation=='scorer':event['proof']['scorer']='b'*64
            elif mutation=='snapshot':rec['evidence_snapshot']['step']=99
            elif mutation=='result':artifact['after_result']['coverage'][0]['status']='unknown'
            else:del batch['private_evidence_scores'][key]
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):compile_fixture(batch,CONFIG)

    def test_frozen_cost_drift_rejected(self):
        batch=fixture();batch['rollouts'][0]['records'][1]['task_events'][1]['value']=-.5
        with self.assertRaises(ValueError):compile_fixture(batch,CONFIG)

    def test_malformed_judge_json_becomes_pending(self):
        for result in ([],None,{'scores':['status_accuracy','citation_support','scope_fidelity'],'coverage':[],'reason':'bad'}):
            with self.subTest(result=result),tempfile.TemporaryDirectory() as d:
                calls=[]
                def transport(*args):calls.append(1);return json.dumps(result)
                receipt=PrivateJudge(model='test',cache=d,transport=transport).score(self.packet())
                self.assertEqual(receipt['status'],'pending');self.assertEqual(len(calls),2)

    def test_malformed_api_envelope_becomes_pending(self):
        for envelope in ({'choices':[]},{'choices':[None]},{'choices':[{'finish_reason':'stop','message':[]}]}):
            with self.subTest(envelope=envelope),tempfile.TemporaryDirectory() as d:
                def transport(*args):return response_content(envelope)
                receipt=PrivateJudge(model='test',cache=d,transport=transport).score(self.packet())
                self.assertEqual(receipt['status'],'pending')
        self.assertEqual(response_content({'choices':[{'finish_reason':'stop','message':{'content':'{}'}}]}),'{}')


if __name__=='__main__':unittest.main()
