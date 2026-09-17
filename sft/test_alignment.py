import unittest,copy,json,sys
import torch
from shared_interface import *
from repair_content import repair_plan,organize_final,refs,THINK,claim_atoms
from prepare_dataset import encode_stage
from train_joint_sft import weighted_completion_loss
class CharTok:
 # Stub solely for loss-mask logic; never used to report real Qwen token lengths.
 is_fast=True;eos_token='\x03';eos_token_id=3
 def encode(self,s,add_special_tokens=False):return [ord(c)for c in s]
 def __call__(self,s,add_special_tokens=False,return_offsets_mapping=False):
  return {'input_ids':self.encode(s),'offset_mapping':[(i,i+1)for i in range(len(s))]}
 def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True,enable_thinking=True):
  return ''.join(m['role']+':'+m['content']+'\n'for m in messages)+'assistant:\n'
class Tests(unittest.TestCase):
 def test_list_delimiter_allowed(self):
  q='For patients X, compare A and B for mortality and function.'
  x=assemble_checklist(q,[['For patients X, compare A and B for','mortality','.'],['For patients X, compare A and B for','function','.']])
  self.assertEqual([a['text']for a in x['allowed_separator_spans']],['and'])
 def test_combination_connector_not_dropped(self):
  q='Does A and B reduce death?'
  with self.assertRaises(ValueError):assemble_checklist(q,[['Does A','reduce death?'],['Does','B reduce death?']])
 def test_alternative_not_globally_ignored(self):
  q='Should I use A or B?'
  with self.assertRaises(ValueError):assemble_checklist(q,[['Should I use A','?'],['Should I use','B?']])
 def test_inside_word_rejected(self):
  q='For incontinence compare continence and pain.'
  with self.assertRaises(ValueError):assemble_checklist(q,[['For in','continence','compare continence and pain.']])
 def test_negation_not_dropped(self):
  with self.assertRaises(ValueError):assemble_checklist('Is A not effective?',[['Is A','effective?']])
 def test_population_not_dropped(self):
  with self.assertRaises(ValueError):assemble_checklist('In women, does A reduce death?',[['does A reduce death?']])
 def test_shared_header_roundtrip(self):
  x={'opened_evidence':[{'source_id':'P#1','text':'one','title':'t','url':'u'},{'source_id':'P#2','text':'two','title':'t','url':'u'}]}
  self.assertEqual(expand_source_headers(compact_source_headers(x)),x)
 def test_header_missing_rejected(self):
  with self.assertRaises(ValueError):expand_source_headers({'opened_evidence':[{'source_id':'x','text':'a','header_ref':'Hbad'}],'source_headers':{}})
 def test_ranking_does_not_promote_target(self):
  c=[{'source_id':'PMID:'+str(i),'title':'A','discovery_text':'text','rrf_score':20-i}for i in range(12)]
  out,_=project_candidates(c,'A','A',Counter())
  self.assertEqual(len(out),8)
  # Public BM25/source-ID order is independent of the supervised target.
  self.assertEqual(out,project_candidates(c,'A','A',Counter())[0])
 def test_no_rewritten_preview(self):
  raw='BACKGROUND The question concerns exercise. RESULTS No effect was observed. CONCLUSION Applicability is limited.'
  text,sp=select_preview(raw,'exercise effect','exercise',Counter(),30)
  for x in sp:self.assertEqual(raw[x['start']:x['end']].strip(),x['text'])
 def test_final_claim_not_repeated(self):
  q='Compare urinary function and bowel symptoms.'
  a='<answer>Urinary function worsened. <cite id="P#1">urinary</cite> Bowel symptoms recovered. <cite id="P#2">bowel</cite> Bowel symptoms recovered. <cite id="P#2">bowel</cite></answer>'
  b,x=organize_final(a,{'terms':['urinary function','bowel symptoms'],'row_number':0},[{'source_id':'P#1'},{'source_id':'P#2'}])
  self.assertEqual(b.count('Bowel symptoms recovered.'),1)
 def test_final_unknown_ref_omitted(self):
  b,_=organize_final('<answer>A. <cite id="X">x</cite></answer>',{'terms':['Answer'],'row_number':0},[])
  self.assertNotIn('X',refs(b))
 def test_mask_input_and_eos(self):
  s={'id':'s','question_id':'q','data_split':'train','stage':'decision','messages':[{'role':'user','content':'PROMPT tool result'}],
    'assistant_prefix':'<tool_output>history</tool_output>\n','completion':'FINAL_READY','loss_scope':'completion_only','sample_weight':.2}
  e,p=encode_stage(CharTok(),s);b=e['first_supervised_token']
  self.assertEqual(e['labels'][:b],[-100]*b);self.assertEqual(e['labels'][b:],[ord(c)for c in 'FINAL_READY\x03']);self.assertEqual(e['sample_weight'],.2)
 def test_reserved_output_not_sft_only(self):
  s={'id':'s','question_id':'q','data_split':'train','stage':'final','messages':[{'role':'user','content':'x'*5900}],'assistant_prefix':'','completion':'<answer>A</answer>','loss_scope':'completion_only'}
  with self.assertRaisesRegex(ValueError,'reserved_output'):encode_stage(CharTok(),s)
 def test_weight_does_not_cancel_microbatch_one(self):
  logits=torch.zeros((1,4,10));labels=torch.tensor([[-100,1,2,3]])
  a=weighted_completion_loss(logits,labels,torch.tensor([1.]));b=weighted_completion_loss(logits,labels,torch.tensor([.2]))
  self.assertAlmostEqual(float(b/a),.2,places=6)
 def test_rejected_action_only_input(self):
  s={'id':'s','question_id':'q','data_split':'train','stage':'decision','messages':[{'role':'user','content':'executed=false: <call_tool name="bad">bad</call_tool>'}], 'assistant_prefix':'','completion':'FINAL_READY','loss_scope':'completion_only'}
  e,p=encode_stage(CharTok(),s);b=e['first_supervised_token'];self.assertTrue(all(t==-100 for t in e['labels'][:b]))
 def test_budget_request_caps(self):
  self.assertEqual(browse_request_kwargs('browse_document'),{'top_k':3,'max_chars':4200,'max_output_tokens':700})
  self.assertEqual(browse_request_kwargs('browse_webpage')['max_output_tokens'],1000)
if __name__=='__main__':unittest.main()
