"""Bounded-memory token/log-prob capture for a fresh joint-RL sampler.

Full-support temperature policy only. No top-p or grammar approximation.
"""
import torch
from transformers import LogitsProcessor, set_seed
from core import digest


class Capture(LogitsProcessor):
    def __init__(self, temperature):
        self.temperature=temperature
        self.pending=None
        self.values=[]
        self.position=None

    def __call__(self, input_ids, scores):
        if input_ids.shape[0]!=1 or scores.shape[0]!=1:
            raise ValueError('capture supports a single sequence')
        if self.pending is not None:
            if input_ids.shape[1]!=self.position+1:raise ValueError('generation prefix drift')
            self.values.append(float(self.pending[input_ids[0,-1]].item()))
        # Hold one vocabulary vector, not max_new_tokens x vocabulary scores.
        self.pending=torch.log_softmax(scores[0].float()/self.temperature,dim=-1).detach()
        self.position=input_ids.shape[1]
        return scores

    def finish(self, sequence):
        if self.pending is None or sequence.shape[1]!=self.position+1:
            raise ValueError('missing final sampled token')
        self.values.append(float(self.pending[sequence[0,-1]].item()))
        self.pending=None
        return self.values


def generate(model, tokenizer, input_ids, *, policy, max_new_tokens, temperature,
             seed, stopping_criteria=None, json_schema=None):
    if not input_ids or len(input_ids)+max_new_tokens>8192 or max_new_tokens<=0 or temperature<=0:
        raise ValueError('invalid generation budget')
    device=next(model.parameters()).device
    inputs=torch.tensor([input_ids],device=device)
    # An explicit autoregressive loop avoids version-dependent generate() config
    # fallback and hidden logits warpers. Exactly these probabilities are sampled.
    base=model.get_base_model() if hasattr(model,'get_base_model') else model
    if not hasattr(base,'model') or not hasattr(base,'lm_head'):
        raise ValueError('sampler requires the Qwen-style backbone/head contract')
    modes=[(module,module.training) for module in model.modules()]
    model.eval();set_seed(seed)
    output=[];logps=[];past=None;sequence=inputs
    eos=tokenizer.eos_token_id
    eos_ids=set(eos if isinstance(eos,(list,tuple)) else ([] if eos is None else [eos]))
    grammar = None
    allowed_fn = None
    support_pool = {}
    support_refs = []
    vocab_size = None
    if json_schema is not None:
        from importlib.metadata import version
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
        if not isinstance(json_schema, dict):
            raise ValueError("schema must be a dict")
        grammar = dict(engine="lm-format-enforcer",
                       version=version("lm-format-enforcer"),
                       schema=json_schema, schema_digest=digest(json_schema))
        allowed_fn = build_transformers_prefix_allowed_tokens_fn(
            tokenizer, JsonSchemaParser(json_schema))
    try:
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                feed=sequence if past is None else sequence[:,-1:]
                result=base.model(input_ids=feed,attention_mask=torch.ones_like(sequence),
                                  past_key_values=past,use_cache=True,return_dict=True)
                past=result.past_key_values
                logits=base.lm_head(result.last_hidden_state[:,-1,:]).float()/temperature
                if allowed_fn is not None:
                    from grammar_support import masked_logits
                    logits, allowed = masked_logits(
                        logits, list(allowed_fn(0, sequence[0])))
                    vocab_size = int(logits.shape[-1])
                    support_key = digest(allowed)
                    support_pool[support_key] = allowed
                    support_refs.append(support_key)
                probs=torch.softmax(logits,dim=-1)
                if not torch.isfinite(probs).all():raise ValueError('nonfinite sampling distribution')
                token=torch.multinomial(probs,num_samples=1)
                token_id=int(token.item())
                # Store log of the actual tensor passed to multinomial, not a
                # pre-warp approximation. Trainer parity is a separate check.
                logps.append(float(probs[0,token_id].log()))
                output.append(token_id);sequence=torch.cat((sequence,token),dim=1)
                if token_id in eos_ids:break
                if stopping_criteria is not None and bool(stopping_criteria(sequence,logits)):break
        if len(logps)!=len(output):raise ValueError('capture/token mismatch')
        record = dict(policy=policy,input_ids=list(input_ids),output_ids=output,
                    behavior_logps=logps,token_digest=digest([input_ids,output]),
                    raw_completion=tokenizer.decode(output,skip_special_tokens=True),
                    audit_completion=tokenizer.decode(output,skip_special_tokens=False),
                    sampling=dict(distribution='temperature_full_support_v1',temperature=temperature,
                                  top_p=1.,top_k=0,grammar=None,seed=seed,
                                  implementation='explicit_multinomial_v1',
                                  eos_token_ids=sorted(eos_ids)),
                    executed_text_as_target=False)
        if grammar is not None:
            from grammar_support import DISTRIBUTION, IMPLEMENTATION, validate_support
            record["sampling"].update(
                distribution=DISTRIBUTION, implementation=IMPLEMENTATION,
                grammar=grammar)
            support = dict(vocab_size=vocab_size, pool=support_pool, refs=support_refs)
            record["support"] = dict(support, digest=digest(support))
            validate_support(record)
        return record
    finally:
        for module,training in modes:module.training=training
