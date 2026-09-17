"""Joint agent credit contracts. No model, Judge, or historical relabeling."""
import hashlib
import json
import math
from collections import Counter, defaultdict

SCHEMA = 'joint_agent_rl_v1_7'
STAGES = {'checklist', 'decision', 'state', 'stop', 'final'}
CHANNELS = {'checklist', 'tool', 'search_query', 'browse_source_focus', 'state', 'stop', 'final'}
STAGE_CHANNELS = {'checklist': {'checklist'}, 'decision': {'tool', 'search_query', 'browse_source_focus'},
                  'state': {'state'}, 'stop': {'stop'}, 'final': {'final'}}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def finite(x):
    if isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x):
        raise ValueError('expected finite numeric value')
    return float(x)


def observed(value):
    """An unknown label is None, never an invented zero."""
    return None if value is None else finite(value)


def validate_config(config):
    if set(config)!={'channels','task'} or set(config['channels'])!=CHANNELS:
        raise ValueError('full channel/task configuration required')
    task=config['task']
    if set(task)!={'stop_scale','coverage_values','tool_cost','policy_penalty'}:
        raise ValueError('missing task reward parameters')
    if not 0 < finite(task['stop_scale']) <= 1:raise ValueError('invalid Stop scale')
    if any(finite(task[k])<0 for k in ('tool_cost','policy_penalty')):raise ValueError('invalid costs')
    v=task['coverage_values']
    if set(v)!={'unknown','partial','direct'} or not 0==finite(v['unknown'])<=finite(v['partial'])<=finite(v['direct'])==1:
        raise ValueError('invalid fixed coverage mapping')
    for weights in config['channels'].values():
        if set(weights)!={'local','final','loss_weight'} or any(finite(x)<0 for x in weights.values()):
            raise ValueError('invalid channel weights')
    if config['channels']['final']['local']!=0 or config['channels']['final']['final']!=1:
        raise ValueError('Final score must enter Final exactly once')


def validate(batch):
    if batch.get('schema') != SCHEMA or batch.get('data_split') != 'train':
        raise ValueError('fresh train batch required')
    if type(batch.get('context_limit')) is not int or not 0 < batch['context_limit'] <= 8192:
        raise ValueError('context limit must be within the supported 8192-token budget')
    identity = batch['identity']
    for key in ('policy', 'runtime', 'tokenizer', 'scorer', 'reward_config', 'task_modes'):
        if not isinstance(identity.get(key), str) or len(identity[key]) != 64:
            raise ValueError('missing frozen identity: '+key)
        int(identity[key], 16)
    if not batch['rollouts']:
        raise ValueError('empty batch')
    scopes=batch.get('task_scopes')
    if not isinstance(scopes,dict) or set(scopes)!={r['question_id'] for r in batch['rollouts']}:
        raise ValueError('frozen task scope registry required')
    for qid,scope in scopes.items():
        required={'question','requirements','constraints','task_mode','task_scope_digest'}
        if not isinstance(scope,dict) or set(scope)!=required or scope['task_mode'] not in {'evidence_grounded','self_contained'}:
            raise ValueError('task scope registry fields')
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'shared'))
        from gain_contract import task_scope
        if scope['task_scope_digest']!=task_scope(scope['question'],scope['requirements'],
                                                  scope['constraints'],scope['task_mode']):
            raise ValueError('task scope digest mismatch')
    question_rollouts = defaultdict(set)
    global_ids,global_executions = set(),set()
    for roll in batch['rollouts']:
        qid, rid = roll['question_id'], roll['rollout_id']
        if rid in global_ids:
            raise ValueError('duplicate rollout')
        global_ids.add(rid)
        question_rollouts[qid].add(rid)
        if roll['policy'] != identity['policy']:
            raise ValueError('mixed behavior policy')
        if roll.get('task_scope_digest')!=scopes[qid]['task_scope_digest']:
            raise ValueError('rollout task scope drift')
        observed(roll['final_reward'])
        if roll['final_reward'] is not None and not isinstance(roll.get('final_reward_proof'),dict):
            raise ValueError('Final reward requires a private receipt')
        ids, events, business_events = set(), set(), set()
        boundary_seen=False
        stages = Counter()
        for rec in roll['records']:
            if stages['final'] or (stages['stop'] and rec['stage'] != 'final'):
                raise ValueError('records after terminal decision/Final')
            if rec['id'] in ids or rec['stage'] not in STAGES:
                raise ValueError('duplicate record or unknown stage')
            ids.add(rec['id']); stages[rec['stage']] += 1
            if rec['policy'] != identity['policy'] or rec.get('executed_text_as_target'):
                raise ValueError('policy drift or fabricated execution target')
            p, c, lp = rec['input_ids'], rec['output_ids'], rec['behavior_logps']
            if not p or not c or len(p)+len(c) > batch['context_limit'] or len(c) != len(lp):
                raise ValueError('invalid token lineage')
            if any(type(x) is not int or x < 0 for x in p+c):
                raise ValueError('invalid token ID')
            if any(finite(x) > 1e-5 for x in lp):
                raise ValueError('invalid log probability')
            sampling = rec['sampling']
            if sampling.get('distribution') == 'temperature_captured_support_v1':
                from grammar_support import validate_support
                validate_support(rec)
            elif (finite(sampling['temperature']) <= 0
                    or sampling.get('top_p') != 1.0
                    or sampling.get('top_k') != 0
                    or sampling.get('grammar') is not None
                    or sampling.get('implementation') != 'explicit_multinomial_v1'
                    or sampling.get('distribution') != 'temperature_full_support_v1'):
                raise ValueError('unsupported sampling distribution')
            if rec['token_digest'] != digest([p, c]):
                raise ValueError('token drift')
            used = set()
            for name, channel in rec['channels'].items():
                if name not in STAGE_CHANNELS[rec['stage']]:
                    raise ValueError('stage/channel mismatch')
                indices = channel['indices']
                if not indices or len(set(indices)) != len(indices):
                    raise ValueError('empty or duplicated token scope')
                if any(type(i) is not int or not 0 <= i < len(c) for i in indices) or used.intersection(indices):
                    raise ValueError('overlapping or invalid token scope')
                used.update(indices)
                value=observed(channel.get('local_reward'))
                if name in {'tool', 'stop'} and channel.get('local_reward') is not None:
                    raise ValueError('tool/stop use task ledger, not an ignored local_reward')
                if value is not None and name not in {'tool','stop'}:
                    proof=channel.get('local_reward_proof')
                    if not isinstance(proof,dict) or set(proof)!={'kind','artifact_key'} or proof['kind']!='private_stage_score_v2':
                        raise ValueError('local score requires a private stage receipt')
                elif channel.get('local_reward_proof') is not None:
                    raise ValueError('unobserved local score cannot carry a receipt')
            execution_id=rec.get('tool_execution_id')
            action_channels=set(rec['channels']) & {'search_query','browse_source_focus'}
            if rec['stage']=='decision':
                if len(action_channels)!=1 or 'tool' not in rec['channels']:
                    raise ValueError('Search/Browse decision requires one semantic channel and tool accounting')
            if rec['stage']=='decision' and 'tool' in rec['channels']:
                if not isinstance(execution_id,str) or not execution_id or execution_id in global_executions:
                    raise ValueError('unique tool execution identity required')
                global_executions.add(execution_id)
            elif execution_id is not None:
                raise ValueError('tool execution identity on non-tool record')
            for event in rec.get('task_events', []):
                if rec['stage'] not in {'decision', 'stop'}:
                    raise ValueError('state/final/checklist cannot mint task revenue')
                if event['id'] in events or event['kind'] not in {'evidence_gain', 'tool_cost', 'policy_penalty', 'stop_boundary'}:
                    raise ValueError('duplicate or invalid task event')
                events.add(event['id'])
                if event['kind']!='stop_boundary':
                    if event.get('tool_execution_id')!=execution_id:
                        raise ValueError('task event/tool execution mismatch')
                    business=(execution_id,event['kind'])
                    if business in business_events:
                        raise ValueError('duplicate business event')
                    business_events.add(business)
                score = observed(event['value'])
                if event['kind']=='stop_boundary':
                    if rec['stage']!='stop' or boundary_seen:
                        raise ValueError('one voluntary Stop boundary per rollout')
                    boundary_seen=True
                    proof=event.get('proof',{})
                    required={'kind','termination_event_id','artifact_key','scorer','reward_config'}
                    if set(proof)!=required or proof.get('kind')!='trusted_stop_boundary_v2' or not proof.get('termination_event_id'):
                        raise ValueError('Stop requires trusted termination and scoring provenance')
                    if proof.get('scorer')!=identity['scorer'] or proof.get('reward_config')!=identity['reward_config']:
                        raise ValueError('Stop scorer/reward identity mismatch')
                    if score is not None and not -1 <= score <= 1:raise ValueError('Stop boundary out of range')
                if event['kind']=='evidence_gain':
                    proof=event.get('proof',{})
                    if (set(proof)!={'kind','artifact_key','tool_execution_id','scorer','reward_config'} or
                        proof.get('kind') not in {'private_evidence_gain_v4','private_evidence_gain_v5'}):
                        raise ValueError('evidence gain requires private bound proof')
                    if proof.get('tool_execution_id')!=execution_id:
                        raise ValueError('evidence gain execution proof mismatch')
                    if proof.get('scorer')!=identity['scorer'] or proof.get('reward_config')!=identity['reward_config']:
                        raise ValueError('evidence gain scorer/reward identity mismatch')
                if event['kind']=='tool_cost':
                    proof=event.get('proof',{})
                    if proof!={'kind':'trusted_tool_execution_v1','tool_execution_id':execution_id}:
                        raise ValueError('tool cost requires execution proof')
                    if score is None:
                        raise ValueError('executed tool cost cannot be unobservable')
                if event['kind']=='policy_penalty':
                    proof=event.get('proof',{})
                    if not isinstance(proof,dict) or set(proof)!={'kind','violation_event_id'} or proof.get('kind')!='trusted_policy_violation_v1' or not proof.get('violation_event_id'):
                        raise ValueError('policy penalty requires trusted violation proof')
                    if score is None:
                        raise ValueError('policy penalty with proof must be observed')
                if score is not None and ((event['kind']=='evidence_gain' and score < 0)
                                          or (event['kind'] in {'tool_cost','policy_penalty'} and score > 0)):
                    raise ValueError('wrong task reward sign')
        if stages['final'] > 1:
            raise ValueError('one Final per rollout; retries need separate explicit handling')
        if stages['stop'] and not boundary_seen:
            raise ValueError('Stop boundary must be explicitly scored or recorded as missing')
        if roll['final_reward'] is not None and stages['final'] != 1:
            raise ValueError('final reward without sampled Final')
        for rec in roll['records']:
            if rec['stage']=='decision' and 'tool' in rec['channels']:
                costs=[event for event in rec.get('task_events',[]) if event['kind']=='tool_cost']
                if len(costs)!=1:
                    raise ValueError('exactly one tool cost per execution required')
    if any(len(v) != 4 for v in question_rollouts.values()):
        raise ValueError('exactly four rollouts per question required')
    # GRPO baselines must use the complete frozen four-rollout group. A missing
    # required semantic core is not zero and must not silently shrink the group.
    required_semantic={'checklist','search_query','browse_source_focus','state'}
    for qid in question_rollouts:
        group=[roll for roll in batch['rollouts'] if roll['question_id']==qid]
        if any(roll['final_reward'] is None for roll in group):
            raise ValueError('question group has required pending core reward')
        for roll in group:
            for rec in roll['records']:
                if any(name in required_semantic and channel.get('local_reward') is None
                       for name,channel in rec['channels'].items()):
                    raise ValueError('question group has required pending core reward')


def compile_batch(batch, config, collector_lookup=None, score_lookup=None,
                  gain_lookup=None, termination_lookup=None,
                  execution_manifest_lookup=None, violation_lookup=None,
                  gain_disposition_lookup=None, coverage_receipt_lookup=None,
                  evidence_transition_lookup=None):
    validate_config(config)
    validate(batch)
    if digest(config) != batch['identity']['reward_config']:
        raise ValueError('reward configuration drift')
    # Private artifacts are batch-side audit data, never policy input text.
    from judge import stop_boundary_event,evidence_gain_event,stage_score_event
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'shared'))
    from gain_contract import (record_target_digest,validate_execution_manifest,
        canonical_coverage_view,canonical_coverage_result,coverage_basis_digest,
        validate_gain_assessment,validate_gain_disposition,
        validate_evidence_transition,validate_tool_execution,
        validate_violation_event)
    unknown_task_records=set()
    not_applicable_stop_records=set()
    # Coverage is inherited by semantic evidence identity, not by a display
    # step number.  Re-reading the same evidence cannot create a new scoring
    # identity merely because the collector advanced its step counter.
    coverage_identities={}
    for roll in batch['rollouts']:
        evidence_ledger_tail=None
        if not callable(execution_manifest_lookup):
            raise ValueError('trusted rollout execution manifest resolver required')
        manifest=execution_manifest_lookup(roll['rollout_id'])
        declared={rec['tool_execution_id'] for rec in roll['records']
                  if rec.get('tool_execution_id') is not None}
        actual=validate_execution_manifest(manifest,question_id=roll['question_id'],
                                           rollout_id=roll['rollout_id'])
        if declared!=actual:
            raise ValueError('batch/tool execution manifest mismatch')
        for record_index,rec in enumerate(roll['records']):
            for name,channel in rec['channels'].items():
                if channel.get('local_reward') is None:
                    continue
                proof=channel['local_reward_proof']
                if not callable(score_lookup):
                    raise ValueError('trusted stage-score resolver required')
                artifact=score_lookup(proof['artifact_key'])
                if not isinstance(artifact,dict) or digest(artifact)!=proof['artifact_key']:
                    raise ValueError('missing or changed private stage score')
                expected=dict(question_id=roll['question_id'],rollout_id=roll['rollout_id'],
                    record_id=rec['id'],channel=name,task_scope_digest=roll['task_scope_digest'],
                    policy=roll['policy'],token_digest=rec['token_digest'],
                    target_digest=record_target_digest(question_id=roll['question_id'],
                        rollout_id=roll['rollout_id'],record=rec,channel=name,
                        task_scope_digest=roll['task_scope_digest']))
                score=stage_score_event(artifact,config=config,scorer=batch['identity']['scorer'],expected=expected)
                if abs(score-finite(channel['local_reward']))>1e-12:
                    raise ValueError('local reward differs from private stage score')
            execution=None
            if rec.get('tool_execution_id') is not None:
                if not callable(collector_lookup):
                    raise ValueError('trusted collector resolver required for tool execution')
                execution=collector_lookup(rec['tool_execution_id'])
                if not isinstance(execution,dict):raise ValueError('missing trusted tool execution')
                validate_tool_execution(execution)
                expected=dict(question_id=roll['question_id'],rollout_id=roll['rollout_id'],
                    record_id=rec['id'],policy=roll['policy'],token_digest=rec['token_digest'],
                    task_scope_digest=roll['task_scope_digest'])
                if any(execution.get(k)!=v for k,v in expected.items()):
                    raise ValueError('tool execution/record binding mismatch')
            gain_status=None
            if rec['stage']=='decision' and 'browse_source_focus' in rec['channels']:
                after=(roll['records'][record_index+1]
                       if record_index+1<len(roll['records']) and
                       roll['records'][record_index+1]['stage']=='state' else None)
                if not callable(evidence_transition_lookup):
                    raise ValueError('trusted evidence transition resolver required for every Browse')
                evidence_transition=evidence_transition_lookup(rec['tool_execution_id'])
                verified_transition=validate_evidence_transition(
                    evidence_transition,tool_execution=execution,
                    expected=dict(question_id=roll['question_id'],
                        rollout_id=roll['rollout_id'],browse_record_id=rec['id'],
                        tool_execution_id=rec['tool_execution_id'],policy=roll['policy'],
                        token_digest=rec['token_digest'],
                        task_scope_digest=roll['task_scope_digest']))
                if rec.get('evidence_snapshot')!=verified_transition['before_snapshot']:
                    raise ValueError('Browse record differs from trusted before-evidence ledger')
                if after is not None and after.get('evidence_snapshot')!=verified_transition['after_snapshot']:
                    raise ValueError('State record differs from trusted after-evidence ledger')
                current_before=(verified_transition['before_snapshot'],
                                verified_transition['before_evidence'],
                                verified_transition['before_source_headers'])
                if evidence_ledger_tail is not None and current_before!=evidence_ledger_tail:
                    raise ValueError('Browse evidence ledger is not continuous across rollout')
                evidence_ledger_tail=(verified_transition['after_snapshot'],
                                      verified_transition['after_evidence'],
                                      verified_transition['after_source_headers'])
                if not callable(gain_disposition_lookup):
                    raise ValueError('trusted gain disposition resolver required for every Browse')
                disposition=gain_disposition_lookup(rec['tool_execution_id'])
                disposition_expected=dict(question_id=roll['question_id'],
                    rollout_id=roll['rollout_id'],browse_record_id=rec['id'],
                    after_record_id=after['id'] if after is not None else None,
                    tool_execution_id=rec['tool_execution_id'],
                    policy=roll['policy'],token_digest=rec['token_digest'],
                    task_scope_digest=roll['task_scope_digest'])
                gain_status=validate_gain_disposition(disposition,disposition_expected)
                gain_events=[event for event in rec.get('task_events',[])
                             if event['kind']=='evidence_gain']
                completed={'observed_positive','observed_zero','unobservable','needs_review'}
                if gain_status in completed:
                    if not callable(gain_lookup):
                        raise ValueError('trusted evidence-gain resolver required')
                    artifact=gain_lookup(disposition['artifact_key'])
                    if not isinstance(artifact,dict) or digest(artifact)!=disposition['artifact_key']:
                        raise ValueError('missing or changed trusted gain assessment artifact')
                    if artifact.get('task_modes_digest')!=batch['identity']['task_modes']:
                        raise ValueError('evidence gain task-mode registry drift')
                    expected_binding=dict(question_id=roll['question_id'],
                        rollout_id=roll['rollout_id'],browse_record_id=rec['id'],
                        after_record_id=after['id'] if after is not None else None,
                        policy=roll['policy'],
                        before_snapshot=verified_transition['before_snapshot'],
                        after_snapshot=verified_transition['after_snapshot'],
                        tool_execution_id=rec['tool_execution_id'],
                        task_scope_digest=roll['task_scope_digest'])
                    assessment=validate_gain_assessment(artifact,
                        scorer=batch['identity']['scorer'],
                        reward_config_digest=digest(config),
                        coverage_values=config['task']['coverage_values'],
                        expected_binding=expected_binding,
                        coverage_receipt_lookup=coverage_receipt_lookup,
                        evidence_transition=evidence_transition)
                    if assessment['status']!=gain_status:
                        raise ValueError('gain disposition differs from verified assessment')
                    if gain_status in {'unobservable','needs_review'}:
                        unknown_task_records.add((roll['rollout_id'],rec['id']))
                    for label,key,snapshot,result in (
                        ('before',artifact['before_key'],artifact['binding']['before_snapshot'],artifact['before_result']),
                        ('after',artifact['after_key'],artifact['binding']['after_snapshot'],artifact['after_result'])):
                        view=artifact[label+'_view']
                        if view!=canonical_coverage_view(view):
                            raise ValueError('noncanonical coverage view reached compiler')
                        coverage_identity=(roll['question_id'],roll['rollout_id'],
                            roll['task_scope_digest'],batch['identity']['scorer'],
                            coverage_basis_digest(view))
                        canonical=(key,canonical_coverage_result(result))
                        previous=coverage_identities.get(coverage_identity)
                        if previous is not None and previous!=canonical:
                            raise ValueError('conflicting coverage for identical evidence basis')
                        coverage_identities[coverage_identity]=canonical
                    if gain_status=='observed_positive':
                        if len(gain_events)!=1:
                            raise ValueError('observed-positive Browse requires exactly one gain event')
                        expected=evidence_gain_event(artifact,event_id=gain_events[0]['id'],
                            config=config,scorer=batch['identity']['scorer'],
                            expected_binding=expected_binding,
                            coverage_receipt_lookup=coverage_receipt_lookup,
                            evidence_transition=evidence_transition)
                        if gain_events[0]!=expected:
                            raise ValueError('evidence gain event differs from verified artifact')
                    elif gain_events:
                        raise ValueError('non-positive Browse cannot carry gain event')
                elif gain_status=='failed_no_evidence':
                    if execution['status']!='failed' or execution['response']['payload']['evidence']:
                        raise ValueError('failed-no-evidence disposition mismatch')
                    if verified_transition['change']!='failed_no_change':
                        raise ValueError('failed Browse evidence ledger mismatch')
                    if gain_events:
                        raise ValueError('failed Browse cannot carry gain event')
                elif gain_status=='observed_zero_no_change':
                    if execution['status']!='succeeded' or verified_transition['change']!='no_change':
                        raise ValueError('zero-no-change disposition mismatch')
                    if gain_events:
                        raise ValueError('zero-no-change Browse cannot carry gain event')
                else:
                    if gain_events:
                        raise ValueError('unresolved Browse cannot carry gain event')
                    unknown_task_records.add((roll['rollout_id'],rec['id']))
            for event in rec.get('task_events',[]):
                if event['kind']=='stop_boundary':
                    if not callable(termination_lookup):
                        raise ValueError('trusted termination resolver required')
                    termination=termination_lookup(event['proof']['termination_event_id'])
                    artifact=None
                    if event['proof']['artifact_key'] is not None:
                        if not callable(score_lookup):raise ValueError('trusted Stop score resolver required')
                        artifact=score_lookup(event['proof']['artifact_key'])
                        if not isinstance(artifact,dict) or digest(artifact)!=event['proof']['artifact_key']:
                            raise ValueError('missing or changed private Stop score')
                    expected_binding=dict(question_id=roll['question_id'],rollout_id=roll['rollout_id'],
                        record_id=rec['id'],token_digest=rec['token_digest'],policy=batch['identity']['policy'],
                        task_scope_digest=roll['task_scope_digest'],
                        target_digest=record_target_digest(question_id=roll['question_id'],
                            rollout_id=roll['rollout_id'],record=rec,channel='stop',
                            task_scope_digest=roll['task_scope_digest']))
                    expected=stop_boundary_event(artifact,termination,event_id=event['id'],config=config,
                                                 scorer=batch['identity']['scorer'],expected=expected_binding)
                    if event!=expected:raise ValueError('Stop event differs from verified score/config')
                    if termination['type']!='active':
                        not_applicable_stop_records.add((roll['rollout_id'],rec['id']))
                elif event['kind']=='evidence_gain':
                    if gain_status!='observed_positive':
                        raise ValueError('gain event lacks observed-positive disposition')
                elif event['kind']=='tool_cost':
                    if event['value'] != -config['task']['tool_cost']:
                        raise ValueError('task cost differs from frozen configuration')
                elif event['kind']=='policy_penalty':
                    if event['value'] != -config['task']['policy_penalty']:
                        raise ValueError('task penalty differs from frozen configuration')
                    if not callable(violation_lookup):
                        raise ValueError('trusted policy violation resolver required')
                    violation=violation_lookup(event['proof']['violation_event_id'])
                    expected_violation=dict(question_id=roll['question_id'],rollout_id=roll['rollout_id'],
                        record_id=rec['id'],tool_execution_id=rec['tool_execution_id'],
                        policy=roll['policy'],token_digest=rec['token_digest'],
                        task_scope_digest=roll['task_scope_digest'])
                    validate_violation_event(violation,expected_violation)
                    if violation['violation_event_id']!=event['proof']['violation_event_id']:
                        raise ValueError('policy violation event identity mismatch')
        if roll['final_reward'] is not None:
            proof=roll['final_reward_proof']
            if not isinstance(proof,dict) or set(proof)!={'kind','artifact_key'} or proof['kind']!='private_stage_score_v2':
                raise ValueError('Final reward proof fields')
            if not callable(score_lookup):
                raise ValueError('trusted stage-score resolver required')
            artifact=score_lookup(proof['artifact_key'])
            if not isinstance(artifact,dict) or digest(artifact)!=proof['artifact_key']:
                raise ValueError('missing or changed Final score artifact')
            finals=[rec for rec in roll['records'] if rec['stage']=='final']
            expected=dict(question_id=roll['question_id'],rollout_id=roll['rollout_id'],
                record_id=finals[0]['id'],channel='final',task_scope_digest=roll['task_scope_digest'],
                policy=roll['policy'],token_digest=finals[0]['token_digest'],
                target_digest=record_target_digest(question_id=roll['question_id'],
                    rollout_id=roll['rollout_id'],record=finals[0],channel='final',
                    task_scope_digest=roll['task_scope_digest']))
            score=stage_score_event(artifact,config=config,scorer=batch['identity']['scorer'],expected=expected)
            if abs(score-finite(roll['final_reward']))>1e-12:
                raise ValueError('Final reward differs from private stage score')
    finals = defaultdict(dict)
    local = defaultdict(lambda: defaultdict(list))
    rows = []
    for roll in batch['rollouts']:
        qid, rid = roll['question_id'], roll['rollout_id']
        if roll['final_reward'] is not None:
            finals[qid][rid] = finite(roll['final_reward'])
        rtg, valid = 0., True
        returns = {}
        for rec in reversed(roll['records']):
            record_identity=(rid,rec['id'])
            for ev in rec.get('task_events', []):
                if ev['value'] is None:
                    if ev['kind']=='stop_boundary' and record_identity in not_applicable_stop_records:
                        continue
                    valid = False
                else: rtg += finite(ev['value'])
            if record_identity in unknown_task_records:
                valid=False
            returns[rec['id']] = rtg if valid else None
        decision_index = 0
        for rec in roll['records']:
            if rec['stage'] in {'decision', 'stop'}: decision_index += 1
            for name, ch in rec['channels'].items():
                if name=='stop' and (rid,rec['id']) in not_applicable_stop_records:
                    score=None
                else:
                    score = returns[rec['id']] if name in {'tool', 'stop'} else ch.get('local_reward')
                # Tool/Stop share a decision-position task baseline. Other channels
                # compare other trajectories' channel means, not individual steps.
                key = (qid, 'task', decision_index) if name in {'tool', 'stop'} else (qid, name)
                if score is not None and name != 'final': local[key][rid].append(finite(score))
                rows.append(dict(question_id=qid, rollout_id=rid, record=rec, channel=name,
                                 indices=ch['indices'], local_score=score, baseline_key=key))
    counts = Counter((r['rollout_id'], r['channel']) for r in rows)
    for row in rows:
        rid, name, qid = row['rollout_id'], row['channel'], row['question_id']
        other = [sum(v)/len(v) for k,v in local[row['baseline_key']].items() if k != rid]
        la = row['local_score']-sum(other)/len(other) if other and row['local_score'] is not None and name!='final' else None
        f = finals[qid]
        siblings = [v for k,v in f.items() if k != rid]
        fa = f[rid]-sum(siblings)/len(siblings) if rid in f and siblings else None
        w = config['channels'][name]
        row.update(local_advantage=la, final_advantage=fa,
                   advantage=(0 if la is None else w['local']*la)+(0 if fa is None else w['final']*fa),
                   observed=(la is not None and w['local']>0) or (fa is not None and w['final']>0),
                   loss_weight=w['loss_weight']/counts[(rid,name)])
        del row['baseline_key']
    return rows


def compile_batch_report(batch, config, **authorities):
    """Compile once and expose a fail-closed orchestration status.

    Only the specific group-level pending-core condition is converted into a
    report.  Integrity, provenance, schema, and authority failures still raise.
    """
    try:
        rows = compile_batch(batch, config, **authorities)
    except ValueError as exc:
        if str(exc) != 'question group has required pending core reward':
            raise
        return {
            'status': 'needs_attention',
            'reward_export_authorized': False,
            'reason': 'required_pending_core',
            'rows': None,
        }
    return {
        'status': 'reward_compiled',
        'reward_export_authorized': True,
        'reason': None,
        'rows': rows,
    }
