import unittest,tempfile,json
from pathlib import Path
from repair_content import organize_final,split_sentences
from shared_interface import template_options,assemble_checklist
from train_tc2 import complete_checkpoints,prune_checkpoints
from runtime_contract import initialize,render_request
from test_alignment import CharTok
from build_alignment import render_prompt

class Fixes(unittest.TestCase):
 def test_no_lexical_absence_claim(self):
  a='<answer>Recommendations are supplied. <cite id="P#1">source</cite></answer>'
  b,_=organize_final(a,{'terms':['Guidance and eligibility','Supporting evidence'],'row_number':0},[{'source_id':'P#1'}])
  self.assertIn('Recommendations are supplied.',b)
  self.assertNotIn('do not separately establish',b)
 def test_abbreviation(self):
  self.assertEqual(split_sentences('The supplied U.S. MEC material applies. Next sentence.'),['The supplied U.S. MEC material applies.','Next sentence.'])
 def test_stage_modes(self):
  for s in ('checklist_init','state_update','evidence_card'):self.assertFalse(template_options(s)['enable_thinking'])
  self.assertTrue(template_options('final')['enable_thinking'])
 def test_runtime_export_same_render(self):
  tok=CharTok();p={'question':'Q','anchor_initialization':True}
  x=render_request('checklist_init','Q',tok,payload=p)
  self.assertEqual(x['rendered'],render_prompt(tok,dict(stage='checklist_init',**{k:x[k]for k in ('messages','assistant_prefix')})))
 def test_runtime_assembler(self):
  q='For patients X, compare A and B for mortality and function.'
  anchors=[['For patients X, compare A and B for',t,'.']for t in ('mortality','function')]
  class Fake:
   def choices(self,**kw):
    assert kw['json_mode'];return [json.dumps({'anchors':anchors})]
  self.assertEqual(initialize(Fake(),q)[0][1],assemble_checklist(q,anchors))
 def test_retention(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   for i in range(5):
    p=root/f'checkpoint-{i:08d}';p.mkdir()
    for n in ('COMPLETE','adapter_config.json','adapter_model.safetensors','state.pt'):(p/n).write_text('stub')
   (root/'.saving-00000005').mkdir()
   (root/'checkpoint-00000006').mkdir()
   prune_checkpoints(root)
   self.assertEqual([p.name for p in complete_checkpoints(root)],[f'checkpoint-{i:08d}'for i in (2,3,4)])
   self.assertTrue((root/'.saving-00000005').exists())
 def test_chunked_completion_loss(self):
  import torch
  from types import SimpleNamespace
  from train_tc2 import loss
  torch.manual_seed(2)
  class Base(torch.nn.Module):
   def __init__(self):
    super().__init__();self.emb=torch.nn.Embedding(13,5);self.lm_head=torch.nn.Linear(5,13);self.config=SimpleNamespace(vocab_size=13)
   def model(self,input_ids,use_cache=False):return SimpleNamespace(last_hidden_state=self.emb(input_ids))
   def get_base_model(self):return self
  b=Base();ids=torch.tensor([[1,2,3,4,5,6]])
  actual=loss(b,ids,3,.25)
  expected=torch.nn.functional.cross_entropy(b.lm_head(b.emb(ids))[:,2:-1].reshape(-1,13),ids[:,3:].reshape(-1))*.25
  self.assertTrue(torch.allclose(actual,expected));actual.backward()
  self.assertTrue(torch.isfinite(b.lm_head.weight.grad).all())
if __name__=='__main__':unittest.main()
