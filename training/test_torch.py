"""CPU only: loss, sampler probability, and atomic checkpoint smoke tests."""
import tempfile
import unittest
from pathlib import Path
import torch
from unittest.mock import patch
from policy_loss import clipped_loss
from policy_loss import selected_logps
from sampling import Capture,generate
from checkpoints import save,complete,verify,model_identity
from train import has_learning_signal


class ToyAdapter:
    def save_pretrained(self,path,**kwargs):
        (Path(path)/'adapter_config.json').write_text('{}')
        (Path(path)/'adapter_model.safetensors').write_bytes(b'toy-only-not-real-model')


class Contracts(unittest.TestCase):
    def test_actual_multinomial_capture_and_trainer_agree(self):
        from transformers import Qwen3Config,Qwen3ForCausalLM
        from types import SimpleNamespace
        cfg=Qwen3Config(vocab_size=32,hidden_size=32,intermediate_size=64,
            num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8)
        model=Qwen3ForCausalLM(cfg).eval()
        # Hostile defaults must not affect the explicit sampling implementation.
        model.generation_config.top_p=.2;model.generation_config.top_k=2
        model.generation_config.temperature=.6
        model.get_base_model=lambda:model
        tokenizer=SimpleNamespace(eos_token_id=None,decode=lambda ids,**kw:str(ids))
        original=torch.multinomial
        for temperature in (1.,.8):
            actual=[]
            def record(probs,**kw):
                sampled=original(probs,**kw)
                actual.append(float(probs[0,sampled.item()].log()))
                return sampled
            with patch('torch.multinomial',side_effect=record):
                rec=generate(model,tokenizer,[1,2,3],policy='a'*64,max_new_tokens=4,temperature=temperature,seed=42)
            self.assertEqual(rec['behavior_logps'],actual)
            with torch.no_grad():computed=selected_logps(model,rec,list(range(4)))
            self.assertTrue(torch.allclose(computed,torch.tensor(actual),atol=1e-5,rtol=1e-5))
            self.assertEqual(rec['sampling']['implementation'],'explicit_multinomial_v1')

    def test_zero_signal_skips_optimizer_and_scheduler_with_momentum(self):
        parameter=torch.nn.Parameter(torch.tensor([1.]))
        optimizer=torch.optim.AdamW([parameter],lr=.1,weight_decay=0.)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        parameter.sum().backward();optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
        before=parameter.detach().clone();epoch=scheduler.last_epoch
        group=[dict(observed=True,advantage=0.,loss_weight=1.)]
        if has_learning_signal(group):
            parameter.sum().mul(0.).backward();optimizer.step();scheduler.step()
        self.assertTrue(torch.equal(before,parameter));self.assertEqual(epoch,scheduler.last_epoch)

    def test_generation_config_in_model_identity(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d)/'base';adapter=Path(d)/'adapter';base.mkdir();adapter.mkdir()
            (base/'config.json').write_text('{}');(base/'tokenizer.json').write_text('{}')
            (base/'model.safetensors').write_bytes(b'toy')
            (adapter/'adapter_config.json').write_text('{}');(adapter/'adapter_model.safetensors').write_bytes(b'toy')
            before=model_identity(base,adapter)
            (base/'generation_config.json').write_text('{"top_p":0.5}')
            self.assertNotEqual(before[0],model_identity(base,adapter)[0])

    def test_gradient_direction(self):
        for adv in (1.,-1.,0.):
            current=torch.tensor([-2.,-3.],requires_grad=True)
            loss,_=clipped_loss(current,current.detach().clone(),adv)
            loss.backward()
            self.assertTrue(torch.isfinite(current.grad).all())
            self.assertEqual(float(current.grad.sum()),-adv)

    def test_exact_temperature_capture(self):
        cap=Capture(.8)
        scores=torch.tensor([[1.,2.,3.]])
        cap(torch.tensor([[1,2]]),scores)
        cap(torch.tensor([[1,2,0]]),scores+1)
        result=cap.finish(torch.tensor([[1,2,0,2]]))
        expected=torch.log_softmax(scores[0]/.8,dim=0)
        self.assertAlmostEqual(result[0],float(expected[0]),places=6)
        self.assertAlmostEqual(result[1],float(expected[2]),places=6)

    def test_checkpoint_restore_and_retention(self):
        with tempfile.TemporaryDirectory() as d:
            model=ToyAdapter()
            for step in range(4):save(d,step,model,{'step':step,'cursor':step,'optimizer':{'mock':step}})
            paths=complete(d);self.assertEqual(len(paths),3)
            verify(paths[-1])
            state=torch.load(paths[-1]/'state.pt',weights_only=False)
            self.assertEqual(state['step'],3)
            self.assertEqual(state['optimizer'],{'mock':3})
            with self.assertRaises(ValueError):save(d,3,model,state)
            (paths[-1]/'adapter_model.safetensors').write_bytes(b'corrupted')
            with self.assertRaises(ValueError):verify(paths[-1])


if __name__=='__main__':unittest.main()
