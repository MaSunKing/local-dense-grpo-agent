"""Single-GPU, weighted completion-only QLoRA; atomic full resume checkpoints."""
import argparse, hashlib, json, math, os, random, signal, time
from pathlib import Path

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def encode(tok, r):
    assert r['training_eligible'] and not r.get('pending_reasons')
    assert r['loss_scope']=='completion_only' and 0<float(r['sample_weight'])<=1
    from prepare_dataset import encode_stage
    assert not r.get('requires_exact_preflight'), 'exact rebuild required'
    encoded,_=encode_stage(tok,r)
    return encoded['input_ids'],encoded['first_supervised_token'],float(r['sample_weight'])

def complete_checkpoints(out):
    import re
    names=('COMPLETE','adapter_config.json','adapter_model.safetensors','state.pt')
    return sorted(p for p in out.iterdir() if re.fullmatch(r'checkpoint-\d{8}',p.name)
      and p.is_dir() and not p.is_symlink() and all((p/n).is_file() and not (p/n).is_symlink() and (p/n).stat().st_size for n in names))

def prune_checkpoints(out):
    import shutil
    root=out.resolve()
    for p in complete_checkpoints(root)[:-3]:
        assert p.resolve().parent==root and not p.is_symlink()
        shutil.rmtree(p)
        print('PRUNED_CHECKPOINT='+str(p),flush=True)

def loss(model, ids, start, weight):
    # Do not materialize 8192 x vocabulary logits. Checkpoint small head chunks.
    import torch
    from torch.utils.checkpoint import checkpoint
    base=model.get_base_model()
    hidden=base.model(input_ids=ids,use_cache=False).last_hidden_state
    h=hidden[:,start-1:-1,:]; targets=ids[:,start:]
    total=hidden.sum()*0
    for i in range(0,targets.shape[1],64):
        def ce(x,y):
            return torch.nn.functional.cross_entropy(base.lm_head(x).float().reshape(-1,base.config.vocab_size),y.reshape(-1),reduction='sum')
        total=total+checkpoint(ce,h[:,i:i+64],targets[:,i:i+64],use_reentrant=False)
    return total/targets.numel()*weight

def main():
    started=time.monotonic(); stopping=[False]
    for name in ('SIGTERM','SIGUSR1','SIGINT'):
        sig=getattr(signal,name,None)
        if sig is not None:signal.signal(sig,lambda *_:stopping.__setitem__(0,True))
    a=argparse.ArgumentParser()
    a.add_argument('--model',required=True); a.add_argument('--output',required=True)
    a.add_argument('--data',default=str(Path(__file__).parent/'data/train.unified.jsonl'))
    a.add_argument('--max-steps',type=int,default=0); a.add_argument('--epochs',type=int,default=1)
    a.add_argument('--accum',type=int,default=8); a.add_argument('--lr',type=float,default=1e-4)
    a.add_argument('--seconds',type=int,default=6300); a.add_argument('--save-every',type=int,default=10)
    a.add_argument('--preflight',action='store_true'); args=a.parse_args()
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training,PeftModel
    package=Path(__file__).parent
    manifest=json.loads((package/'manifest.json').read_text())
    for name,digest in manifest.items():
        assert sha(package/name)==digest, f'package changed: {name}'
    for name,digest in json.loads((package/'tokenizer_hashes.json').read_text()).items():
        assert sha(Path(args.model)/name)==digest, f'tokenizer changed: {name}'
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    rows=[json.loads(x) for x in Path(args.data).open(encoding='utf8') if x.strip()]
    assert all(r['data_split']=='train' for r in rows)
    examples=[encode(tok,r) for r in rows]
    assert examples and all(0<n<len(ids)<=8192 for ids,n,w in examples)
    print('INPUT=passed samples='+str(len(examples)),flush=True)
    if args.preflight:return
    assert torch.cuda.is_available()
    random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    identity=dict(data=sha(args.data),code=sha(__file__),package=manifest,model=str(Path(args.model).resolve()),epochs=args.epochs,accum=args.accum,lr=args.lr,max_steps=args.max_steps)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    checkpoints=complete_checkpoints(out)
    latest=checkpoints[-1] if checkpoints else None
    state=torch.load(latest/'state.pt',map_location='cpu',weights_only=False) if latest else None
    if state:assert state['identity']==identity,'resume identity mismatch'
    if (out/'TRAINING_COMPLETE').exists():
        assert state is not None,'completion marker without valid checkpoint'
        print('ALREADY_COMPLETE'); return
    base=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':0},quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
    model=PeftModel.from_pretrained(base,str(latest),is_trainable=True) if latest else get_peft_model(base,LoraConfig(r=32,lora_alpha=64,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
    model.train(); optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr)
    total=math.ceil(len(examples)/args.accum)*args.epochs
    if args.max_steps:total=min(total,args.max_steps)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min(1.,(s+1)/max(1,int(total*.05)))*.5*(1+math.cos(math.pi*min(s,total)/total)))
    step=0; epoch=0; position=0
    if state:
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        step,epoch,position=state['step'],state['epoch'],state['position']
        random.setstate(state['random']); torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])
        print('RESUMED step='+str(step),flush=True)
    def save():
        final=out/f'checkpoint-{step:08d}'
        if final.exists():
            assert final in complete_checkpoints(out), 'existing incomplete checkpoint; do not overwrite'
            return
        temp=out/f'.saving-{step:08d}-{os.getpid()}'; temp.mkdir()
        model.save_pretrained(temp)
        torch.save(dict(identity=identity,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),step=step,epoch=epoch,position=position,random=random.getstate(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),temp/'state.pt')
        (temp/'COMPLETE').write_text('ok'); os.replace(temp,final)
        print('CHECKPOINT='+str(final),flush=True)
        prune_checkpoints(out)
    while epoch<args.epochs and step<total:
        order=list(range(len(examples))); random.Random(42+epoch).shuffle(order)
        while position<len(order) and step<total:
            batch=order[position:position+args.accum]; optimizer.zero_grad(set_to_none=True); value=0.
            for index in batch:
                ids,n,w=examples[index]
                with torch.autocast('cuda',dtype=torch.bfloat16): l=loss(model,torch.tensor([ids],device='cuda'),n,w)/len(batch)
                assert torch.isfinite(l); l.backward(); value+=l.item()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            assert torch.isfinite(norm)
            optimizer.step(); scheduler.step(); step+=1; position+=len(batch)
            print(json.dumps(dict(step=step,loss=value,grad_norm=float(norm),position=position,epoch=epoch)),flush=True)
            if step%args.save_every==0:save()
            if stopping[0] or time.monotonic()-started>=args.seconds:
                save(); print('PAUSED_RESUBMIT_SAME_COMMAND',flush=True); return
        epoch+=1; position=0
    save(); (out/'TRAINING_COMPLETE').write_text(str(step)); print('TRAINING_COMPLETE',flush=True)

if __name__=='__main__':main()
