"""Opt-in V4 interface entry. Does not patch a running process.

Both dataset construction and a V4 inference caller use these same functions.
The existing v51 production runner is NOT automatically migrated by this module.
"""
import json
from shared_interface import assemble_checklist,INIT_PROMPT,template_options
from build_alignment import render_stage,render_prompt

def initialize(policy,question,seed=None):
    payload={'question':question,'anchor_initialization':True}
    prompt=INIT_PROMPT+'\n\n'+json.dumps(payload,ensure_ascii=False,separators=(',',':'))
    for attempt in range(2):
        raw=policy.choices(messages=[{'role':'user','content':prompt}],n=1,temperature=.1,
            max_tokens=1200,json_mode=True,seed=(seed or 0)+attempt)[0]
        try:
            obj=json.loads(raw)
            if set(obj)!={'anchors'}:raise ValueError('only anchors allowed')
            state=assemble_checklist(question,obj['anchors'])
            return [(raw,state)]
        except (ValueError,TypeError,KeyError):
            prompt=INIT_PROMPT+'\n\n'+json.dumps(payload,ensure_ascii=False,separators=(',',':'))+'\nReturn valid exact anchors covering the original scope.'
    return [(raw,assemble_checklist(question,[[question]]))]

def render_request(stage,question,tokenizer,**kwargs):
    # Inputs only: render_stage never needs or receives a target completion.
    obj={'stage':stage,'_question':question}
    render_stage(obj,stage,**kwargs)
    return {'messages':obj['messages'],'assistant_prefix':obj['assistant_prefix'],
      'template_options':template_options(stage),'rendered':render_prompt(tokenizer,obj)}
