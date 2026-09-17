#!/usr/bin/env python3
"""Exact-tokenizer final gate. No model inference or training; no silent truncation.
The caller must first rebuild projections using the SAME local tokenizer:
python build_alignment.py --input PREVIOUS_FULL.jsonl --tokenizer LOCAL --out exact
Then encode exact/train.unified.jsonl here. This prevents target-blind preview
selection from silently changing after construction of training states.
"""
from __future__ import annotations
import argparse,hashlib,json,collections
from pathlib import Path
from shared_interface import LIMITS,VERSION,digest
from build_alignment import render_prompt,iterrows

def encode_stage(tok,s):
 if not tok.is_fast:raise ValueError('fast_tokenizer_offsets_required')
 if s.get('loss_scope')!='completion_only':raise ValueError('wrong_loss_scope')
 prompt=render_prompt(tok,s);completion=s['completion'];eos=tok.eos_token
 if not isinstance(completion,str)or not completion.strip():raise ValueError('empty_target')
 if not eos or tok.eos_token_id is None:raise ValueError('missing_eos')
 if not completion.endswith(eos):completion+=eos
 res=tok(prompt+completion,add_special_tokens=False,return_offsets_mapping=True);ids=res['input_ids'];boundary=len(prompt)
 if len(ids)>LIMITS.context:raise ValueError('sft_total_overflow')
 reserve=getattr(LIMITS,s['stage'],600);pn=len(tok.encode(prompt,add_special_tokens=False));cn=len(tok.encode(completion,add_special_tokens=False))
 if pn+reserve>LIMITS.context:raise ValueError('input_plus_reserved_output_overflow')
 if cn>reserve:raise ValueError('completion_including_think_and_eos_overflow')
 labels=[]
 for k,(ident,(a,b))in enumerate(zip(ids,res['offset_mapping'])):
  if a<boundary<b:raise ValueError('input_completion_cross_boundary_token')
  active=a>=boundary and b>a
  if a==b and ident==tok.eos_token_id and k==len(ids)-1:active=True
  labels.append(ident if active else -100)
 if not any(v!=-100 for v in labels[:-1])or labels[-1]!=tok.eos_token_id:raise ValueError('invalid_target_mask')
 first=next(k for k,v in enumerate(labels)if v!=-100)
 return {'id':s['id'],'question_id':s['question_id'],'data_split':s['data_split'],'stage':s['stage'],
  'input_ids':ids,'attention_mask':[1]*len(ids),'labels':labels,'sample_weight':float(s.get('sample_weight',1)),
  'first_supervised_token':first,'prompt_tokens':pn,'target_tokens':cn,'reserved_output':reserve,
  'rendered_sha256':hashlib.sha256((prompt+completion).encode()).hexdigest()},prompt

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
 p.add_argument('--split',choices=['train','dev'],default='train');p.add_argument('--allow-weak-labels',action='store_true');a=p.parse_args()
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(a.model,use_fast=True,local_files_only=True)
 rows=[];bad=[];scenebad=set();seen=set();counts=collections.Counter()
 for s in iterrows(a.input):
  if s['id']in seen:raise SystemExit('duplicate_stage_id: do not stack exports')
  seen.add(s['id']);scene=s.get('augmentation',{}).get('scenario_id')
  try:
   if s.get('requires_exact_preflight'):raise ValueError('rebuild_with_exact_tokenizer_first')
   if s['data_split']!=a.split:raise ValueError('wrong_split')
   if not s.get('training_eligible')or s.get('pending_reasons'):raise ValueError('not_eligible')
   if s.get('requires_weak_label_opt_in')and not a.allow_weak_labels:raise ValueError('weak_labels_not_enabled')
   enc,text=encode_stage(tok,s);rows.append((enc,scene));counts['encoded_'+s['stage']]+=1
  except (ValueError,TypeError,KeyError)as e:
   bad.append({'id':s['id'],'reason':str(e),'scenario_id':scene})
   if scene:scenebad.add(scene)
 dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True);accepted=0
 with dest.open('w')as f:
  for row,scene in rows:
   if scene in scenebad and scene is not None:bad.append({'id':row['id'],'reason':'atomic_scenario_excluded','scenario_id':scene});continue
   f.write(json.dumps(row,ensure_ascii=False)+'\n');accepted+=1
 report={'accepted':accepted,'rejected':len(bad),'by_stage_before_scenario_filter':counts,'exact_tokenizer':str(Path(a.model).resolve()),
  'template_sha256':hashlib.sha256(str(tok.chat_template).encode()).hexdigest(),'limits':LIMITS.__dict__,
  'input_and_history_loss_masked':True,'current_completion_and_eos_supervised':True,'no_truncation':True}
 Path(str(dest)+'.report.json').write_text(json.dumps(report,indent=2)+'\n')
 Path(str(dest)+'.excluded.jsonl').write_text(''.join(json.dumps(x)+'\n'for x in bad))
 print(json.dumps(report,indent=2))
 if not accepted:raise SystemExit('No accepted samples; inspect report. Do not bypass the exact rebuild gate.')
if __name__=='__main__':main()
