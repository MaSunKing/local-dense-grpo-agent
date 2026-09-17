#!/usr/bin/env python3
"""Independent deterministic full-file validator. Not medical entailment scoring."""
import argparse,collections,copy,hashlib,json,re
from pathlib import Path
from build_alignment import iterrows,terminal,allstages,OUTPUT,CALL,render_plain_prompt
from shared_interface import assemble_checklist,expand_source_headers,digest,Counter,wm,LIMITS,template_options
from repair_content import refs,THINK,ANSWER,claim_atoms

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--rebuilt',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 errors=[];counts=collections.Counter();stageids=set();splits=collections.defaultdict(set);counter=Counter();examples=[]
 src=iterrows(a.source);rows=iterrows(a.rebuilt)
 for num,(old,r)in enumerate(zip(src,rows),1):
  qid=r['question_id'];counts['questions']+=1;splits[r['data_split']].add(qid)
  def err(sid,reason):errors.append({'row':num,'qid':qid,'stage_id':sid,'reason':reason})
  if old['old_trajectory']!=r['old_trajectory']:err(None,'old_trajectory_mutated')
  else:counts['original_trajectories_unchanged']+=1
  if old['new_trajectory']['events']!=r['new_trajectory']['events']:err(None,'historical_events_mutated')
  else:counts['historical_events_unchanged']+=len(old['new_trajectory']['events'])
  if old['data_split']!=r['data_split']:err(None,'split_changed')
  plan=r['all_question_rebuild']['plan'];ini=assemble_checklist(plan['question'],plan['anchors']);desc={it['id']:it['description']for it in ini['requirements']}
  counts['checklists_accepted']+=1;counts['request_items']+=len(desc);counts['multi'if len(desc)>1 else'single']+=1
  before={s['id']:s for s in allstages(old)};cur=copy.deepcopy(ini['requirements'])
  for s in allstages(r):
   sid=s['id'];kind=s['stage'];counts['stages']+=1
   if sid in stageids:err(sid,'duplicate_stage_id')
   stageids.add(sid)
   if s['question_id']!=qid or s['data_split']!=r['data_split']:err(sid,'stage_identity_mismatch')
   if s.get('loss_scope')!='completion_only':err(sid,'wrong_loss_scope')
   if s.get('semantic_gold')is not False:err(sid,'overclaims_gold')
   if s.get('template_options')!=template_options(kind):err(sid,'stage_template_mismatch')
   if s['training_eligible']and s.get('pending_reasons'):err(sid,'eligible_with_pending')
   if s['completion'].count('<think>')!=s['completion'].count('</think>') or s['completion'].count('<think>')>1:err(sid,'thinking_format')
   original=before[s['source_stage_id']]
   if digest(s)==digest(original):err(sid,'not_rebuilt')
   if kind=='checklist_init':
    ans=json.loads(THINK.sub('',s['completion']))
    if ans!={'anchors':plan['anchors']}:err(sid,'init_target_mismatch')
   evidence=[]
   if kind in('state_update','final','evidence_card'):
    try:
     pp=terminal(s['messages'][-1]['content']);pp=expand_source_headers(pp);evidence=pp.get('opened_evidence',[])
    except ValueError as e:
     if kind!='evidence_card':err(sid,'payload_parse:'+str(e))
     pp={}
   if kind in('decision','decision_stop'):
    bb=[json.loads(m[1])for m in OUTPUT.finditer(s['assistant_prefix'])];st=next(b for b in bb if 'current_view'in b)
    view=expand_source_headers(st['current_view']);c=view['current_candidates'];evidence=view['opened_evidence']
    if len(c)>8:err(sid,'candidate_window_exceeded')
    ids={x['source_id']for x in c}
    if len(ids)!=len(c):err(sid,'duplicate_current_source')
    audit=s['public_projection_audit']
    if audit['target_used_in_ranking']is not False:err(sid,'target_in_ranking')
    for d in c:
     cap=128 if d['source_id'].startswith(('PMID:','S2:','PMC:','DOI:')) else 96
     if counter.n(d['preview'])>cap+2 and not audit['counter_exact']:err(sid,'estimated_preview_cap')
    counts['public_views_checked']+=1
    if kind=='decision':
     cm=list(CALL.finditer(s['completion']))
     if len(cm)!=1:err(sid,'decision_not_one_action')
     elif cm[0]['tool'].startswith('browse_'):
      counts['browse_decisions']+=1
      if s['training_eligible']and s.get('visibility_audit',{}).get('target_visible')is not True:err(sid,'eligible_invisible_browse')
      if 'query='not in cm[0]['attrs']:err(sid,'missing_browse_query')
    if kind=='decision_stop'and THINK.sub('',s['completion']).strip()!='FINAL_READY':err(sid,'stop_target')
    if s in r['new_stages'] and st['runtime_working_memory']!=wm(cur):err(sid,'state_memory_chain_mismatch')
   if kind=='state_update':
    up=json.loads(THINK.sub('',s['completion']))['updates'];initems=pp.get('items',[])
    if [x['id']for x in up]!=list(desc):err(sid,'state_item_order')
    for x in initems:
     if desc.get(x['id'])!=x['description']:err(sid,'state_description_mismatch')
    ev={d['source_id']for d in evidence}
    for u in up:
     if u['status']not in('unknown','missing','partial','direct'):err(sid,'invalid_status')
     if not set(u['evidence_ids'])<=ev:err(sid,'state_unopened_citation')
     if u['status']in('unknown','missing') and u['evidence_ids']:err(sid,'unknown_with_citation')
     if u['status']in('partial','direct')and not u['evidence_ids']:err(sid,'support_without_citation')
    if s in r['new_stages']:
     if initems!=cur:err(sid,'state_prior_mismatch')
     for item,u in zip(cur,up):item.update(status=u['status'],evidence_ids=u['evidence_ids'])
    counts['state_updates_checked']+=1
   if kind=='final':
    if s['completion'].count('<answer>')!=1 or s['completion'].count('</answer>')!=1:err(sid,'answer_format')
    ev={d['source_id']for d in evidence}
    if not set(refs(s['completion']))<=ev:err(sid,'final_unopened_citation')
    counts['final_citation_occurrences']+=sum(len(m.split(','))for m in re.findall(r'<cite\s+id="([^\"]+)"',s['completion']))
    if s in r['new_stages'] and pp.get('policy_evidence_state',{}).get('requirements')!=cur:err(sid,'final_state_mismatch')
    atoms,_=claim_atoms(s['completion']);clean=[re.sub(r'\W','',z['text'].lower())for z in atoms]
    if len(set(clean))!=len(clean):err(sid,'exact_duplicate_final_claim')
    counts['finals_checked']+=1
   # Lossless full text, or extractive span trace. No newly written source prose.
   oldev=[]
   if original['stage']in('state_update','final'):
    oldev=terminal(original['messages'][-1]['content']).get('opened_evidence',[])
   elif original['stage']in('decision','decision_stop'):
    obs=[json.loads(m[1])for m in OUTPUT.finditer(original['assistant_prefix'])]
    oldev=next(b['current_view']for b in reversed(obs)if'current_view'in b).get('opened_evidence',[])
   ob={d['source_id']:d.get('text','')for d in oldev}
   for d in evidence:
    oldtext=ob.get(d['source_id'])
    if oldtext is None and kind!='evidence_card':err(sid,'future_or_invented_source')
    elif oldtext==d.get('text',''):counts['verbatim_evidence_copies']+=1
    elif d.get('text_kind')=='extractive_opened_evidence_card':
     ss=d.get('source_spans',[]);parts=[oldtext[z['start']:z['end']].strip()for z in ss]
     if ' … '.join(parts)!=d['text']:err(sid,'card_text_not_exact_spans')
     if digest(oldtext)!=d.get('parent_text_sha256'):err(sid,'card_parent_hash')
     counts['exact_extractive_cards_checked']+=1
    elif kind!='evidence_card':err(sid,'untraced_evidence_change')
  if num in(604,750,761,767,806,910,922,934,983):
   examples.append({'row':num,'question_id':qid,'question':plan['question'],'anchors':plan['anchors'],
    'descriptions':desc,'final':next(s['completion']for s in r['new_stages']if s['stage']=='final')})
 if next(src,None)is not None or next(rows,None)is not None:errors.append({'reason':'different_question_counts'})
 if len(splits['train']&splits['dev']):errors.append({'reason':'train_dev_overlap'})
 out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
 result={'counts':counts,'splits':{k:len(v)for k,v in splits.items()},'train_dev_overlap':len(splits['train']&splits['dev']),
  'errors':errors,'error_count':len(errors),'scope':'all-stage structural, visibility and source-copy validation; NOT exact Qwen tokens, medical entailment or V9 server parity'}
 (out/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');(out/'examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False,indent=2))
 if errors:raise SystemExit(1)
if __name__=='__main__':main()
