"""V18.8: separate option presence from complete comparative support."""
from active_policy_v6 import PROMPTS as PREVIOUS


OLD = "one_or_more_requested_options_supported"
NEW = "any_requested_option_or_member_present"

PRESENCE_RULE = '''OPTION PRESENCE RULE: Evaluate any_requested_option_or_member_present independently from whether the complete requested comparison is available. Set it true when the visible material concerns any requested option, comparator, class, intervention, or an identifiable member of one of those classes. For a request comparing A with B, material about A alone, B alone, or a within-A or within-B comparison makes this atom true while complete_requirement_support remains false. Do not require all requested arms for this presence atom.\n'''

PROMPTS = {name: text.replace(OLD, NEW) for name, text in PREVIOUS.items()}
for name in ('browse', 'state_truth', 'state_citation'):
    PROMPTS[name] = PRESENCE_RULE + PROMPTS[name]
