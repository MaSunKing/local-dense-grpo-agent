"""Leading closing-think compatibility; no tools/results fabricated."""
import re

def normalize(raw):
    text=str(raw).strip()
    if not text.lower().startswith('</think>'):return None
    # Only strip a contiguous leading run; never repair action bodies or prose.
    suffix=re.sub(r'^(?:</think>\s*)+', '', text, flags=re.I).strip()
    if not re.fullmatch(r'<call_tool\s+name=["\'](?:pubmed_search|medical_web_search|browse_document|browse_webpage)["\'][^<>]*>[^<>]+</call_tool>',suffix,re.S):
        return None
    return suffix

def install(common):
    original=common.parse_agent_action
    def parse(raw,candidates,v71):
        result=original(raw,candidates,v71)
        if result is not None:return result
        clean=normalize(raw)
        if clean is None:return None
        result=original(clean,candidates,v71)
        if result is None:return None
        result['completion']=raw
        # An inner parser may already have corrected a registered Browse alias.
        # Keep the action actually executed, not the intermediate normalized text.
        result.setdefault('execution_completion',clean)
        outer={'kind':'orphan_think_action_v46','raw_completion':raw,
            'normalized_completion':clean,'training_eligible':False,'parameters_changed':False}
        if 'execution_compatibility' in result:
            result['outer_execution_compatibility']=outer
        else:
            result['execution_compatibility']=outer
        return result
    common.parse_agent_action=parse
