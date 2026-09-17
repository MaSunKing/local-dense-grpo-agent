"""CPU-only release gate, no network/model weights/optimizer."""
import collections,json,sys
from pathlib import Path
from build_alignment import iterrows,allstages
from prepare_dataset import encode_stage
from runtime_contract import initialize
from shared_interface import template_options
from transformers import AutoTokenizer

def main():
 root=Path(__file__).resolve().parent;tok=AutoTokenizer.from_pretrained(sys.argv[1],local_files_only=True,use_fast=True)
 counts=collections.Counter();missing_sentence='The supplied opened passages do not separately establish this requested result.'
 for r in iterrows(root/'data/old_new_all_1053.aligned.jsonl'):
  plan=r['all_question_rebuild']['plan']
  class Fake:
   def choices(self,**kwargs):return [json.dumps({'anchors':plan['anchors']})]
  state=initialize(Fake(),plan['question'])[0][1]
  assert state==r['new_trajectory']['initial_checklist']
  counts['runtime_checklist_matches']+=1
  for s in allstages(r):
   if s['stage']in ('checklist_init','state_update','evidence_card'):json.loads(s['completion'])
   assert s['template_options']==template_options(s['stage'])
   if s['stage']=='final':assert missing_sentence not in s['completion']
 for s in iterrows(root/'data/train.unified.jsonl'):
  assert s['data_split']=='train' and s['training_eligible'] and not s['pending_reasons']
  assert not s['requires_exact_preflight']
  enc,_=encode_stage(tok,s);counts[s['stage']]+=1
  assert enc['labels'][:enc['first_supervised_token']]==[-100]*enc['first_supervised_token']
  assert enc['labels'][enc['first_supervised_token']:]==enc['input_ids'][enc['first_supervised_token']:]
  if sum(counts[k]for k in ('decision','decision_stop','final','state_update','checklist_init'))%2000==0:print(dict(counts),flush=True)
 out={'status':'passed','checks':counts,'gpu_training_tested':False,'legacy_runner_installed':False,'full_live_runtime_parity':False}
 (root/'release_validation.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
