#!/usr/bin/env python3
"""All-question reconstruction on the immediately preceding 1053-question package.
No Agent/backend/model execution. Replays and corrective inputs remain synthetic.
Run --tokenizer /local/EXACT_RUNTIME_TOKENIZER to build token-exact projections;
without it all measured lengths are explicitly ESTIMATES requiring exact preflight.
"""
from __future__ import annotations
import argparse,copy,collections,gzip,hashlib,html,json,os,re,sys
from pathlib import Path
from shared_interface import *
from repair_content import *
OUTPUT=re.compile(r'<tool_output\b[^>]*>(.*?)</tool_output>',re.S)
CALL=re.compile(r'<call_tool\s+name=["\'](?P<tool>[^"\']+)["\'](?P<attrs>[^>]*)>(?P<body>.*?)</call_tool>',re.S)
ATTR=re.compile(r'(\w+)\s*=\s*(["\'])(.*?)\2',re.S)
ORD={'unknown':0,'missing':0,'partial':1,'direct':2}
RUNTIME_RANKER=None

def iterrows(path):
 op=gzip.open if str(path).endswith('.gz')else open
 with op(path,'rt',encoding='utf8')as f:
  for l in f:
   if l.strip():yield json.loads(l)
def filehash(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def terminal(s):
 d=json.JSONDecoder()
 for m in re.finditer(r'\{',s):
  try:
   x,e=d.raw_decode(s[m.start():])
   if not s[m.start()+e:].strip():return x
  except ValueError:pass
 raise ValueError('no_terminal_payload')
def default_prompt_count(s,counter):
 # Exact chat template when supplied. No fixed-token shortcut.
 return counter.n(render_prompt(counter.tokenizer,s)) if counter.exact else counter.n(render_plain_prompt(s))
def render_plain_prompt(s):
 return ''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n'for m in s['messages'])+'<|im_start|>assistant\n'+(s.get('assistant_prefix')or'')
def render_prompt(tokenizer,s):
 if tokenizer is None:return render_plain_prompt(s)
 # Template options are fixed by stage, never inferred from target content.
 text=tokenizer.apply_chat_template(s['messages'],tokenize=False,add_generation_prompt=True,**template_options(s['stage']))
 # Some older templates auto-open <think>. We explicitly own the stage boundary
 # and allow optional thinking; don't place historical tools inside a think block.
 text=re.sub(r'<think>\s*$','',text)
 return text+(s.get('assistant_prefix')or'')
def new_id(s):return s['id'].replace(':all1053-v2:',':aligned-v4:') if ':all1053-v2:'in s['id']else s['id']+':aligned-v4'
def allstages(r):return r.get('new_stages',[])+r.get('targeted_examples',[])+r.get('optional_cards',[])
def desc_items(items,initial):
 by={x['id']:x['description']for x in initial['requirements']};out=copy.deepcopy(items)
 for x in out:
  if x['id']not in by:raise ValueError('unexpected_item_id')
  x['description']=by[x['id']]
 return out

def summarize_history(prefix,view):
 obs=list(OUTPUT.finditer(prefix));history=[]
 for m in CALL.finditer(prefix):
  # An error mentioned inside a tool_output is not another executed call.
  if any(o.start()<=m.start()<o.end()for o in obs):continue
  nexto=next((o for o in obs if o.start()>=m.end()),None)
  if nexto is None:continue
  try:b=json.loads(nexto[1])
  except ValueError:continue
  attrs={k:html.unescape(v)for k,_,v in ATTR.findall(m['attrs'])}
  arr=b.get('data',[])if isinstance(b,dict)else[]
  record={'tool':m['tool'],'argument':html.unescape(m['body']).strip(),'query':attrs.get('query'),
    'executed':not (b.get('executed')is False or b.get('runtime_feedback',{}).get('executed')is False),
    'returned_ids':[x['source_id']for x in arr if isinstance(x,dict)and x.get('source_id')]}
  if b.get('error'):record['error']=b['error']
  if not arr:record['result_count']=b.get('total_returned',0)
  else:record['result_count']=len(arr)
  history.append(record)
 return history

def render_stage(s,kind,payload=None,state=None,history=None,feedback=None):
 if kind in ('decision','decision_stop'):
  q=s['_question'];s['messages']=[{'role':'system','content':DECISION_PROMPT},{'role':'user','content':'Question: '+q}]
  # No private lineage, reward, gold/weak status or answer labels enter the input.
  s['assistant_prefix']='<tool_output id="history">'+dumps({'history':history or []})+'</tool_output>\n'
  state_for_model=copy.deepcopy(state);state_for_model['current_view']=compact_source_headers(state_for_model['current_view'])
  s['assistant_prefix']+='<tool_output id="current_view">'+dumps(state_for_model)+'</tool_output>\n'
  for f in feedback or []:s['assistant_prefix']+='<tool_output id="feedback">'+dumps(f)+'</tool_output>\n'
 elif kind=='checklist_init':
  s['messages']=[{'role':'user','content':INIT_PROMPT+'\n\n'+dumps(payload)}];s['assistant_prefix']=''
 elif kind=='state_update':
  s['messages']=[{'role':'user','content':STATE_PROMPT+'\n\n'+dumps(compact_source_headers(payload))}];s['assistant_prefix']=''
 elif kind=='final':
  fp=copy.deepcopy(payload);st=fp.get('policy_evidence_state',{});fp['policy_evidence_state']={k:st[k]for k in ('requirements','optional_requirements')if k in st}
  s['messages']=[{'role':'system','content':FINAL_PROMPT},{'role':'user','content':dumps(compact_source_headers(fp))}];s['assistant_prefix']=''

def trim_to_cards(opened,question,item_text,counter,budget):
 """Only triggered on overflow. Extractive not generative cards; selection uses
 public question/Checklist, NOT completion, reward, hidden rubric or source target.
 Whole sentences first; original offsets and SHA are retained outside output claim.
 """
 if not opened:return [],[]
 cards=[];audit=[];per=max(48,budget//len(opened)-45)
 for d in opened:
  x=opened_public(d);raw=x.get('text','');cap=min(counter.n(raw),per)
  text,sp=select_preview(raw,question,item_text,counter,cap)
  x['text']=text;x['text_kind']='extractive_opened_evidence_card';x['parent_text_sha256']=digest(raw)
  x['source_spans']=[{'start':z['start'],'end':z['end']}for z in sp]
  if text:cards.append(x);audit.append({'source_id':d['source_id'],'original_text_sha256':digest(raw),'spans':sp})
 return cards,audit

def apply_ref_visibility(updates,opened,prior_opened=None):
 ids={d['source_id']for d in opened};out=copy.deepcopy(updates)
 for u in out:
  keep=[x for x in u['evidence_ids']if x in ids]
  if not keep and u['status']in('partial','direct'):
   u.update(status='unknown',evidence_ids=[],revision_reason='The supplied packet does not contain the previously cited opened text; the earlier support is not established here.')
  elif len(keep)!=len(u['evidence_ids']):
   u['status']='partial';u['evidence_ids']=keep;u['revision_reason']='Only the listed opened chunks remain available; the previous complete support set is no longer present.'
 return out

def restore_thinking(old_final,new_answer,items,row,scenario_mapping=None):
 # User-supplied brief rationales may be retained. Do not synthesize hidden CoT.
 thinks=THINK.findall(old_final or '');thinking=thinks[-1].strip()if thinks else ''
 if not thinking:return new_answer,{'mode':'allowed_no_invented_trace','retained':False,'reason':'no_prior_brief_rationale'}
 strong=bool(re.search(r'\b(?:demonstrates? superiority|proves?|establishes? (?:that|a benefit)|all requirements|all gaps|complete evidence|evidence is sufficient)\b',thinking,re.I))
 disagreement=(row in(295,367) and bool(re.search(r'\b(?:shows? (?:a )?benefit|improv\w*|effective)\b',thinking,re.I)))
 if (strong and any(x['status']!='direct'for x in items)) or disagreement or len(thinking.split())>90:
  return new_answer,{'mode':'allowed_no_invented_trace','retained':False,'reason':'unsupported_or_overlong_prior_rationale','old_rationale_sha256':digest(thinking)}
 # Relabel as retained teacher rationale; it is not an independently validated trace.
 return '<think>'+thinking+'</think>\n'+new_answer,{'mode':'optional_brief_teacher_rationale','retained':True,
  'verbatim_original':True,'not_a_new_reasoning_trace':True,'old_rationale_sha256':digest(thinking)}

def process_stage(original,r,plan,initial,counter,current=None,standalone=False,mapping=None):
 s=copy.deepcopy(original);sid=s['id'];s['id']=new_id(s);kind=s['stage'];q=plan['question'];s['_question']=q
 s['source_stage_id']=sid;s['adaptation_schema']=VERSION;s['template_options']=template_options(kind)
 s['loss_scope']='completion_only';s['eos_supervised']=True
 s['semantic_gold']=False;s['requires_weak_label_opt_in']=True;s['requires_exact_preflight']=not counter.exact
 s['model_limits']={'context':LIMITS.context,'reserved_output':getattr(LIMITS,kind,600)}
 s.setdefault('provenance',{}).update({'alignment_version':VERSION,'generated_demonstration':True,'real_agent_execution':False,
  'spec_source':'user_supplied_v9_contract','exact_v9_source_available':False,'source_stage_sha256':digest(original)})
 s.setdefault('pending_reasons',[])
 if 'augmentation'in s:
  s['augmentation']['parent_stage_id']=new_id({'id':s['augmentation'].get('parent_stage_id',sid)})
  if s['augmentation'].get('scenario_id'):s['augmentation']['scenario_id']=new_id({'id':s['augmentation']['scenario_id']})
 state=payload=None;history=[];feedback=[];updates=[];orig_opened=[]
 if kind=='checklist_init':
  s['completion']=dumps({'anchors':plan['anchors']});payload={'question':q,'anchor_initialization':True}
  render_stage(s,kind,payload=payload)
 elif kind in('decision','decision_stop'):
  blocks=[json.loads(o[1])for o in OUTPUT.finditer(original['assistant_prefix'])]
  stateold=next(b for b in reversed(blocks)if 'current_view'in b)
  # Non-base corrective samples intentionally contain a changed input state.
  items=copy.deepcopy(current or initial['requirements'])
  history=summarize_history(original['assistant_prefix'],stateold['current_view'])
  focused=next((h['argument']for h in reversed(history)if h['tool'].endswith('_search')),'')
  state,ca=project_decision(blocks,q,items,counter,ranker=RUNTIME_RANKER,focused_query=focused);s['public_projection_audit']=ca
  state['current_view']['opened_evidence']=pdf_scenario_evidence([opened_public(d)for d in stateold['current_view'].get('opened_evidence',[])],s.get('augmentation',{}))
  orig_opened=copy.deepcopy(state['current_view']['opened_evidence'])
  # Standalone prefix retains its own previously visible policy state, with new descriptions.
  if standalone:
   memory=stateold.get('runtime_working_memory','')
   for it in items:
    m=re.search(r'CORE '+re.escape(it['id'])+r':.*?\|\s*(unknown|missing|partial|direct)\s*\|\s*evidence_ids=(\[[^\n]*\])',memory)
    if m:
     it['status']=m[1]
     try:it['evidence_ids']=json.loads(m[2])
     except ValueError:pass
   state['runtime_working_memory']=wm(items)
  for b in blocks:
   if 'current_view'not in b and 'runtime_feedback'in b:feedback.append(b)
  # Current candidate pool exposure is computed before parsing the demonstration target.
  render_stage(s,kind,state=state,history=history,feedback=feedback)
  if kind=='decision':
   m=CALL.search(s['completion'])
   if m and m['tool'].startswith('browse_'):
    chosen=html.unescape(m['body']).strip();visible={x['source_id']for x in state['current_view']['current_candidates']}
    s['visibility_audit']={'chosen_source':chosen,'visible_ids':list(visible),'target_visible':chosen in visible,'target_promoted':False}
    if chosen not in visible:s['pending_reasons'].append('browse_target_outside_current_8_window')
 elif kind in('state_update','final'):
  payload=terminal(original['messages'][-1]['content']);orig_opened=pdf_scenario_evidence([opened_public(d)for d in payload.get('opened_evidence',[])],s.get('augmentation',{}))
  payload['opened_evidence']=copy.deepcopy(orig_opened)
  if kind=='state_update':
   # In the ordinary chain use new prior states. In recovery retain the injected
   # wrong status in the input; only the corrected target is supervised.
   if standalone:before=desc_items(payload['items'],initial)
   else:before=desc_items(current or payload['items'],initial)
   payload['items']=before
   updates=json.loads(THINK.sub('',s['completion']))['updates'];updates=apply_ref_visibility(updates,orig_opened)
   priorby={x['id']:x for x in before};by={x['id']:x for x in initial['requirements']}
   support=s.get('annotation_audit',{}).get('item_support',[]);basis={a['id']:a for a in support}
   for u in updates:
    prev=priorby[u['id']]
    if u.get('revision_reason')or ORD[u['status']]<ORD[prev['status']]or (prev['evidence_ids'] and u['evidence_ids']!=prev['evidence_ids']):
     u['revision_reason']=specific_reason(q,by[u['id']],u,orig_opened,basis.get(u['id']))
   s['completion']=dumps({'updates':updates});s.setdefault('annotation_audit',{}).update({'before_items_sha256':digest(before),'evidence_sha256':digest(orig_opened),'updates':updates,'alignment_note':'reasons_specific_to_supplied_scope_no_new_medical_review'})
  else:
   fst=copy.deepcopy(initial);fst['requirements']=copy.deepcopy(current or desc_items(payload.get('policy_evidence_state',{}).get('requirements',initial['requirements']),initial))
   payload['policy_evidence_state']=fst
   answer,aa=organize_final(s['completion'],plan,orig_opened,mapping)
   olds='\n'.join(m['content']for m in r['old_trajectory'].get('conversations',[])if m.get('role')=='assistant')
   oldfinal=olds[olds.rfind('</tool_output>')+len('</tool_output>'):]if '</tool_output>'in olds else olds
   answer,ta=restore_thinking(oldfinal,answer,fst['requirements'],plan['row_number'],mapping)
   s['completion']=answer;s['final_reorganization_audit']=aa;s['thinking_audit']=ta
   s['answer_item_map']=aa['claim_item_map']
  render_stage(s,kind,payload=payload)
 elif kind=='evidence_card':
  # Retain original extractive-only card targets and their source content.
  s['messages']=copy.deepcopy(original['messages']);s['assistant_prefix']=original.get('assistant_prefix','')
 else:raise ValueError('unsupported_stage:'+kind)

 # Compact only AFTER the same public input has been constructed, and only when
 # the inference reserve does not fit. Neither target IDs nor rewards select cards.
 reserve=getattr(LIMITS,kind,600);before_n=default_prompt_count(s,counter)
 length_audit={'counter':'exact_runtime_tokenizer'if counter.exact else'explicit_estimate_not_measured_tokens',
  'prompt_before':before_n,'reserve':reserve,'input_budget':LIMITS.context-reserve,'cards_triggered':False}
 if before_n+reserve>LIMITS.context and kind in('decision','decision_stop','state_update','final'):
  original_evidence=orig_opened
  # Keep reference-independent source excerpts; later gates ensure no target
  # depends on a removed source passage. No truncated target or source hallucination.
  for factor in (.75,.55,.4):
   evbudget=max(300,int((LIMITS.context-reserve-1100)*factor))
   cards,cardaudit=trim_to_cards(original_evidence,q,' '.join(plan['terms']),counter,evbudget)
   if state is not None:state['current_view']['opened_evidence']=cards;render_stage(s,kind,state=state,history=history,feedback=feedback)
   elif payload is not None:payload['opened_evidence']=cards;render_stage(s,kind,payload=payload)
   length_audit.update(cards_triggered=True,extractive_card_audit=cardaudit)
   if default_prompt_count(s,counter)+reserve<=LIMITS.context:break
  # Check source content, not merely ID existence, before letting reduced evidence
  # support unchanged targets. Conservatively quarantine affected labels instead
  # of pretending retained IDs certify the same claims.
  oldby={d['source_id']:d.get('text','')for d in original_evidence}
  newby={d['source_id']:d.get('text','')for d in cards}
  targetrefs=refs(s['completion'])if kind=='final'else[x for u in updates for x in u['evidence_ids']]if kind=='state_update'else[]
  affected=[]
  for sid2 in set(targetrefs):
   if oldby.get(sid2,'')!=newby.get(sid2,''):
    # Some annotations provide exact supporting spans. Require all such supplied
    # spans; no hidden teacher score is used in card selection.
    quotes=[]
    audit=s.get('annotation_audit',{})
    for sup in audit.get('item_support',[]):quotes += [z['quote']for z in sup.get('support_quotes',[])if z.get('source_id')==sid2]
    if not quotes or not all(z in newby.get(sid2,'')for z in quotes):affected.append(sid2)
  if affected:s['pending_reasons'].append('target_support_not_verified_after_conditional_card');length_audit['affected_target_sources']=affected
 length_audit['prompt_after']=default_prompt_count(s,counter)
 length_audit['completion_plus_eos']=counter.n(s['completion']+'<|im_end|>')
 length_audit['input_plus_reserve']=length_audit['prompt_after']+reserve
 length_audit['sft_total']=counter.n(render_prompt(counter.tokenizer,s)+s['completion']+'<|im_end|>')
 if length_audit['input_plus_reserve']>LIMITS.context:s['pending_reasons'].append('input_reserve_overflow')
 if length_audit['sft_total']>LIMITS.context:s['pending_reasons'].append('sft_total_overflow')
 if length_audit['completion_plus_eos']>reserve:s['pending_reasons'].append('target_exceeds_generation_cap')
 s['budget_audit']=length_audit
 # Tool-side budget is audited separately from model generation. Legacy source
 # snapshots are not silently trimmed to a newer backend's arbitrary boundaries.
 if orig_opened:
  s['evidence_packet_origin']='inherited_opened_source_snapshot_not_new_v9_browse_execution'
 s['pending_reasons']=list(dict.fromkeys(s['pending_reasons']));s['training_eligible']=bool(original.get('training_eligible'))and not s['pending_reasons']
 s.pop('_question',None)
 after=None
 if kind=='state_update':
  after=copy.deepcopy(payload['items']);by={u['id']:u for u in json.loads(s['completion'])['updates']}
  for x in after:
   for k in ('status','evidence_ids'):x[k]=copy.deepcopy(by[x['id']][k])
 return s,after

def audit_browse_history(events,counter):
 out=[]
 for e in events:
  if not e['tool'].startswith('browse_'):continue
  arr=[d for d in e.get('observation',{}).get('data',[])if isinstance(d,dict)and d.get('text')]
  cap=LIMITS.paper_browse_tokens if e['tool']=='browse_document'else LIMITS.web_browse_tokens
  txt='\n'.join(d['text']for d in arr)
  if len(arr)>3 or len(txt)>LIMITS.browse_chars or counter.n(txt)>cap:
   out.append({'legacy_event_index':e.get('legacy_event_index',e.get('index')),'tool':e['tool'],'source_id':e.get('argument'),
    'chunks':len(arr),'text_chars':len(txt),'tokens_or_estimate':counter.n(txt),'request_token_cap':cap,
    'status':'legacy_snapshot_exceeds_new_requested_return_budget','not_claimed_as_current_v9_return':True})
 return out

def main():
 global RUNTIME_RANKER
 a=argparse.ArgumentParser();a.add_argument('--input',required=True);a.add_argument('--out',required=True);a.add_argument('--tokenizer');a.add_argument('--ranker',help='Optional module:function from the actual V9 runtime, signature (candidates,question,focused_query)');args=a.parse_args()
 if args.ranker:
  import importlib
  mod,fn=args.ranker.split(':',1);RUNTIME_RANKER=getattr(importlib.import_module(mod),fn)
 dest=Path(args.out);dest.mkdir(parents=True,exist_ok=True);tok=None
 if args.tokenizer:
  from transformers import AutoTokenizer
  tok=AutoTokenizer.from_pretrained(args.tokenizer,use_fast=True,local_files_only=True)
  if not tok.is_fast:raise SystemExit('Fast tokenizer required')
 if filehash(args.input)!='2109ed19c6a5d0195eda18e63c18e472d6f843628d464e710a8992332ba375b1':
  raise SystemExit('This builder requires the immediately preceding full-rebuilt input; do not feed the aligned output back into it.')
 counter=Counter(tok);counts=collections.Counter();pending=collections.Counter();splits=collections.Counter();exports=collections.Counter()
 files={k:(dest/(k+'.jsonl')).open('w',encoding='utf8') for k in ['old_new_all_1053.aligned','train.unified','train.interface','train.final','train.cards','dev.aligned','per_question_manifest','changes','excluded','length_audit','candidate_visibility','claim_assignment','checklist_acceptance','browse_snapshot_budget_audit']}
 def write(k,x):files[k].write(dumps(x)+'\n')
 for n,r0 in enumerate(iterrows(args.input),1):
  r=copy.deepcopy(r0);plan,initial=repair_plan(r0['all_question_rebuild']['plan']);current=copy.deepcopy(initial['requirements']);rebuilt=[];alignid={}
  for old in r0['new_stages']:
   s,after=process_stage(old,r0,plan,initial,counter,current)
   alignid[old['id']]=s['id'];rebuilt.append(s)
   if after is not None:current=after
  r['new_stages']=rebuilt
  # PDF scenarios are complete independent chains; use the same repaired plan,
  # projection and prompt builder, not parent states carrying the wrong IDs.
  scenegroups=collections.defaultdict(list);standalone=[]
  for s in r0.get('targeted_examples',[]):
   ag=s.get('augmentation',{})
   if ag.get('scenario_id')and ag.get('kind')in('dual_pdf_s2','dual_pdf_abstract_fallback'):scenegroups[ag['scenario_id']].append(s)
   else:standalone.append(s)
  targets=[]
  for old in standalone:
   s,_=process_stage(old,r0,plan,initial,counter,standalone=True);targets.append(s);alignid[old['id']]=s['id']
  for scene,oldstages in scenegroups.items():
   curr=copy.deepcopy(initial['requirements']);newscene=[];mapping=oldstages[0].get('augmentation',{}).get('id_mapping')
   for old in oldstages:
    s,after=process_stage(old,r0,plan,initial,counter,curr,mapping=mapping)
    newscene.append(s);alignid[old['id']]=s['id']
    if after is not None:curr=after
   bad=[s['id']for s in newscene if not s['training_eligible']]
   if bad:
    for s in newscene:s['training_eligible']=False;s['pending_reasons']=list(dict.fromkeys(s['pending_reasons']+['atomic_pdf_scenario_excluded']))
    counts['pdf_scenarios_excluded']+=1
   else:counts['pdf_scenarios_kept']+=1
   targets+=newscene
  r['targeted_examples']=targets;r['optional_cards']=[]
  for old in r0.get('optional_cards',[]):
   s,_=process_stage(old,r0,plan,initial,counter,standalone=True);r['optional_cards'].append(s);alignid[old['id']]=s['id']
  r['new_trajectory']['initial_checklist']=initial;r['new_trajectory']['stage_ids']=[s['id']for s in rebuilt]
  for ev in r['new_trajectory'].get('sft_demonstration_events',[]):
   if ev['tool'].startswith('browse_'):
    ev['requested_runtime_budget']=browse_request_kwargs(ev['tool'])
    ev['observation_budget_provenance']='retained_legacy_snapshot_not_claimed_as_new_budget_execution'
  for scene in r.get('rebuilt_pdf_scenarios',[]):
   scene['stage_ids']=[alignid[x]for x in scene['stage_ids']];scene['scenario_id']=new_id({'id':scene['scenario_id']})
  r['all_question_rebuild']['previous_plan_sha256']=digest(r0['all_question_rebuild']['plan']);r['all_question_rebuild']['plan']=plan
  r['engineering_alignment']={'version':VERSION,'all_stages_visited':True,'source_row_sha256':digest(r0),
    'exact_v9_code_available':False,'actual_runtime_execution':False,'tokenizer_exact':counter.exact,
    'ranker':'declared_snapshot_rank_or_lexical_replay_not_verified_v9','new_runtime_shared_module':'shared_interface.py',
    'clinical_source_text_modified':False,'weak_labels_not_reward_truth':True,'checkpoint_lineage_must_be_new':True}
  ba=audit_browse_history(r0['new_trajectory'].get('sft_demonstration_events',[]),counter)
  if ba:write('browse_snapshot_budget_audit',{'question_id':r['question_id'],'row':n,'events':ba});counts['questions_legacy_browse_budget_mismatch']+=1
  num_changed=sum(digest(a)!=digest(b)for a,b in zip(allstages(r0),allstages(r)))
  man={'row':n,'question_id':r['question_id'],'data_split':r['data_split'],'all_stages_rebuilt':True,
   'items':len(initial['requirements']),'anchor_repairs':plan['anchor_repair_notes'],
   'separator_spans':initial['allowed_separator_spans'],'stages':len(allstages(r)),
   'eligible':sum(s['training_eligible']for s in allstages(r)),
   'final_thinking_retained':next(s['thinking_audit']['retained']for s in rebuilt if s['stage']=='final'),
   'old_record_unchanged':r['old_trajectory']==r0['old_trajectory'],'historical_events_unchanged':r['new_trajectory']['events']==r0['new_trajectory']['events'],
   'source_stage_ids':[s['id']for s in allstages(r0)],'new_stage_sha256':digest(allstages(r))}
  write('per_question_manifest',man);write('changes',{'question_id':r['question_id'],'before':digest(allstages(r0)),'after':digest(allstages(r)),'stage_count':len(allstages(r))})
  write('checklist_acceptance',{'row':n,'question_id':r['question_id'],'accepted':True,'anchors':plan['anchors'],'descriptions':[x['description']for x in initial['requirements']],'allowed_separator_spans':initial['allowed_separator_spans'],'semantic_verified':False})
  write('old_new_all_1053.aligned',r);counts['questions']+=1;counts['questions_multi'if len(initial['requirements'])>1 else'questions_single']+=1;splits[r['data_split']]+=1
  if plan['anchor_repair_notes']:counts['questions_anchor_content_repaired']+=1
  for s in allstages(r):
   counts['stages']+=1;counts['stage_'+s['stage']]+=1
   write('length_audit',{'id':s['id'],'question_id':s['question_id'],'stage':s['stage'],'split':s['data_split'],'eligible':s['training_eligible'],**s['budget_audit']})
   if s.get('visibility_audit'):write('candidate_visibility',{'id':s['id'],**s['visibility_audit']})
   if s.get('final_reorganization_audit'):
    write('claim_assignment',{'id':s['id'],**s['final_reorganization_audit']});counts['duplicate_final_claims_removed']+=s['final_reorganization_audit']['duplicate_claims_removed'];counts['final_think_retained']+=int(s['thinking_audit']['retained'])
   if s['budget_audit']['cards_triggered']:counts['conditional_cards_'+s['stage']]+=1
   if s['data_split']=='dev':write('dev.aligned',s);exports['dev.aligned']+=1
   if not s['training_eligible']:
    for reason in s['pending_reasons']:pending[reason]+=1
    write('excluded',{'id':s['id'],'question_id':s['question_id'],'stage':s['stage'],'split':s['data_split'],'reasons':s['pending_reasons']});continue
   if s['data_split']!='train':continue
   if s['stage']=='evidence_card':write('train.cards',s);exports['train.cards']+=1;continue
   write('train.unified',s);exports['train.unified']+=1
   kind='train.final'if s['stage']=='final'else'train.interface';write(kind,s);exports[kind]+=1
  if n%100==0:print('processed',n,dict(exports),flush=True)
 for f in files.values():f.close()
 summary={'version':VERSION,'input_sha256':filehash(args.input),'counts':counts,'splits':splits,'exports':exports,'excluded_reasons':pending,
  'runtime_limits':asdict(LIMITS),'tokenizer_exact':counter.exact,'ranker_exact_v9_verified':False,'external_ranker_adapter':args.ranker,
  'not_run':['model_training','Agent_tools','network_medical_search','server_or_PBS_changes'],
  'requires_local_exact_tokenizer_rebuild':not counter.exact}
 (dest/'build_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
