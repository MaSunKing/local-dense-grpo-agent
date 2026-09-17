"""One frozen-behavior batch, shared LoRA, no SFT loss and no Judge dependency.

Requires fresh token-exact scored records. Does not load or control a resident.
Run only after the inference service has released the project's single-GPU lock.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import random
import signal
import time
from pathlib import Path

from core import compile_batch, digest
from checkpoints import complete, verify, save, sha, model_identity
from authority import TrustedAuthorityBundle


def has_learning_signal(group):
    return any(r['observed'] and r['loss_weight']>0 and r['advantage']!=0 for r in group)


def main():
    p = argparse.ArgumentParser()
    for name in ('batch','model','adapter','output'): p.add_argument('--'+name,required=True)
    p.add_argument('--authority',required=True,
                   help='read-only trusted reward authority bundle bound to the exact batch file')
    p.add_argument('--config',default=str(Path(__file__).with_name('config.json')))
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--lr',type=float,default=1e-6)
    p.add_argument('--epsilon',type=float,default=.2)
    p.add_argument('--target-kl',type=float,default=.02)
    p.add_argument('--parity-tolerance',type=float,default=.05)
    p.add_argument('--seconds',type=int,default=6300)
    p.add_argument('--save-every',type=int,default=1)
    args=p.parse_args()
    if not 0 < args.lr <= 1e-3 or not 0 < args.epsilon < 1 or args.save_every<1 or args.seconds<0:
        raise ValueError('invalid training options')
    batch_bytes=Path(args.batch).read_bytes()
    batch_sha256=hashlib.sha256(batch_bytes).hexdigest()
    batch=json.loads(batch_bytes.decode('utf8'))
    config=json.loads(Path(args.config).read_text(encoding='utf8'))
    authority=TrustedAuthorityBundle(args.authority,batch_bytes)
    rows=compile_batch(batch,config,
        collector_lookup=authority.tool_execution,
        score_lookup=authority.stage_score,
        gain_lookup=authority.evidence_gain,
        termination_lookup=authority.termination,
        execution_manifest_lookup=authority.execution_manifest,
        violation_lookup=authority.policy_violation,
        gain_disposition_lookup=authority.gain_disposition,
        coverage_receipt_lookup=authority.coverage_receipt,
        evidence_transition_lookup=authority.evidence_transition)
    counts=defaultdict(int)
    for row in rows:
        if row['observed']: counts[row['channel']]+=1
    print('JOINT_BATCH_PREFLIGHT='+json.dumps(dict(counts)),flush=True)
    if not any(r['observed'] and r['advantage']!=0 and r['loss_weight']>0 for r in rows):
        raise ValueError('no nonzero observed learning signal; do not fake differences')
    if args.preflight:
        print('SCOPE=batch_structure_and_advantages_only; model/GPU/parity not checked')
        return
    started=time.monotonic(); stopping=[False]
    for name in ('SIGTERM','SIGUSR1','SIGINT'):
        sig=getattr(signal,name,None)
        if sig is not None:signal.signal(sig,lambda *_:stopping.__setitem__(0,True))
    def pause_requested():
        return stopping[0] or time.monotonic()-started>=args.seconds
    out=Path(args.output).resolve()
    basepath, adapterpath=Path(args.model).resolve(),Path(args.adapter).resolve()
    if any(out==p or out in p.parents or p in out.parents for p in (basepath,adapterpath)):
        raise ValueError('training output must be separate from model and adapter')
    if out.exists() and not (out/'JOINT_RUN.json').exists():
        raise ValueError('output exists without joint run identity; refuse reuse')
    import fcntl
    gpu_lock=open('/tmp/medgap-resident-sft-18771.lock','a')
    fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # No second model may be loaded while the project's resident holds this lock.
    import torch
    from transformers import AutoModelForCausalLM,BitsAndBytesConfig
    from peft import PeftModel,prepare_model_for_kbit_training
    from policy_loss import selected_logps,clipped_loss
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    policy,tokenizer=model_identity(basepath,adapterpath)
    if policy!=batch['identity']['policy'] or tokenizer!=batch['identity']['tokenizer']:
        raise ValueError('batch was not sampled by this frozen model/tokenizer')
    identity=dict(batch=batch_sha256,config=digest(config),policy=policy,
                  code={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},
                  lr=args.lr,epsilon=args.epsilon,target_kl=args.target_kl,
                  parity_tolerance=args.parity_tolerance,precision='NF4_bfloat16')
    out.mkdir(parents=True,exist_ok=True)
    from judge import atomic_json  # File utility only; no scorer is instantiated or called.
    if (out/'JOINT_RUN.json').exists():
        if json.loads((out/'JOINT_RUN.json').read_text())!=identity: raise ValueError('resume identity mismatch')
    else: atomic_json(out/'JOINT_RUN.json',identity)
    old=complete(out); latest=old[-1] if old else None
    if latest: verify(latest)
    state=torch.load(latest/'state.pt',map_location='cpu',weights_only=False) if latest else None
    if state and state['identity']!=identity: raise ValueError('checkpoint identity mismatch')
    if (out/'TRAINING_COMPLETE').exists():
        if not state: raise ValueError('completion marker without checkpoint')
        print('ALREADY_COMPLETE'); return
    model=AutoModelForCausalLM.from_pretrained(str(basepath),local_files_only=True,
        torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
          bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
                                         gradient_checkpointing_kwargs={'use_reentrant':False})
    model=PeftModel.from_pretrained(model,str(latest or adapterpath),is_trainable=True)
    parameters=[p for n,p in model.named_parameters() if p.requires_grad]
    if not parameters or any('lora_' not in n for n,p in model.named_parameters() if p.requires_grad):
        raise ValueError('only shared LoRA parameters may train')
    model.gradient_checkpointing_disable()
    # Nested functional replay owns activation recomputation.
    model.train()
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout): module.eval()
    optimizer=torch.optim.AdamW(parameters,lr=args.lr,weight_decay=0.)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    cursor=step=0
    if state:
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        cursor,step=state['cursor'],state['step']
        random.setstate(state['random']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])
    else:
        # Do not overwrite captured behavior probabilities with training-start guesses.
        with torch.no_grad():
            seen=set(); max_delta=0.
            for row in rows:
                if pause_requested():
                    print('PAUSED_BEFORE_UPDATE: parity must rerun; original adapter unchanged',flush=True)
                    return
                rec=row['record'];key=(row['rollout_id'],rec['id'])
                if key in seen: continue
                seen.add(key)
                current=selected_logps(model,rec,list(range(len(rec['output_ids']))))
                oldlp=torch.tensor(rec['behavior_logps'],device=current.device)
                max_delta=max(max_delta,float((current-oldlp).abs().max()))
            if max_delta>args.parity_tolerance: raise ValueError('sampler/trainer parity failed: '+str(max_delta))
            print('BEHAVIOR_PARITY_MAX_DELTA='+str(max_delta),flush=True)
    grouped=defaultdict(list)
    for row in rows:
        if row['observed'] and row['loss_weight']>0:grouped[row['question_id']].append(row)
    questions=sorted(grouped)
    # Precompute skips deterministically from the frozen batch. They never move
    # optimizer/scheduler state or create an uncheckpointed cursor-only update.
    skipped=[q for q in questions if not has_learning_signal(grouped[q])]
    atomic_json(out/'SKIPPED_NO_SIGNAL.json',dict(questions=skipped))
    questions=[q for q in questions if has_learning_signal(grouped[q])]
    last_saved=step if state else None
    def checkpoint():
        nonlocal last_saved
        if last_saved==step:return
        save(out,step,model,dict(identity=identity,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
             cursor=cursor,step=step,random=random.getstate(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
        last_saved=step
    while cursor<len(questions):
        if pause_requested():
            checkpoint(); print('PAUSED_RESUBMIT_SAME_COMMAND');return
        group=grouped[questions[cursor]];optimizer.zero_grad(set_to_none=True)
        total_weight=sum(r['loss_weight'] for r in group); metrics=[]
        for row in group:
            if pause_requested():
                optimizer.zero_grad(set_to_none=True)
                checkpoint(); print('PAUSED_RESUBMIT_SAME_COMMAND');return
            current=selected_logps(model,row['record'],row['indices'])
            oldlp=torch.tensor([row['record']['behavior_logps'][i] for i in row['indices']],device=current.device)
            loss,metric=clipped_loss(current,oldlp,row['advantage'],args.epsilon)
            (loss*row['loss_weight']/total_weight).backward()
            metrics.append(dict(channel=row['channel'],loss=float(loss.detach()),**metric))
        if max(m['kl_k3'] for m in metrics)>args.target_kl:
            optimizer.zero_grad(set_to_none=True);checkpoint()
            atomic_json(out/'EARLY_STOP.json',dict(step=step,cursor=cursor,reason='target_kl',metrics=metrics))
            print('EARLY_STOP_TARGET_KL');return
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        optimizer.step();scheduler.step();step+=1;cursor+=1
        atomic_json(out/f'update-{step:08d}.json',dict(step=step,grad_norm=float(norm),metrics=metrics))
        if step%args.save_every==0:checkpoint()
        if pause_requested():
            checkpoint(); print('PAUSED_RESUBMIT_SAME_COMMAND');return
    checkpoint();atomic_json(out/'TRAINING_COMPLETE',dict(step=step,cursor=cursor))
    print('JOINT_BATCH_TRAINING_COMPLETE')


if __name__=='__main__':main()
