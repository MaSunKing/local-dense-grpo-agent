"""V18.5: decompose comparative support before judging completeness."""
from active_policy_v3 import PROMPTS as PREVIOUS


PROMPTS = dict(PREVIOUS)

COMPONENT_SUPPORT_RULE = '''COMPARATIVE COMPONENT-SUPPORT RULE: For a requirement that asks to compare option A with option B for one or more outcomes, material_help does NOT require a direct or indirect A-versus-B effect estimate. Decide material_help for the selected material itself, not by ranking it against other visible candidates; a better alternative affects only optional selection-efficiency auxiliary judgment. If the visible candidate or opened evidence supplies a concrete requested outcome, safety result, or other requested fact for A, for B, or for a clearly identified member of either requested option/class in the requested population or a materially applicable population, material_help must be true. This includes one-side evidence, a within-A or within-B comparison, and an add-on A+B versus A or B design. The absence of the other requested option is a reason for complete_scope=false, not material_help=false. Such material is incomplete when the other option or the requested comparison is missing, so complete_scope must be false. Same-topic mentions without a requested result, a materially wrong population, or evidence about neither requested option remain material_help false. Never infer the missing arm or promote component evidence into a comparative conclusion.
'''

STATE_CITATION_COMPONENT_RULE = '''PARTIAL STATE-CITATION RULE: When the Agent claims partial, a citation correctly supports that status if it supplies a concrete requested result for either requested option or an applicable member of that option/class in the requested or materially applicable population, even when it cannot establish the requested comparison. The absence of the other option or a comparative effect must not by itself produce an incorrect verdict for a partial status. Do not require the citation to support the missing arm or a comparative effect for a partial status. A direct status still requires the complete requested scope, and a citation with no requested result remains incorrect.
'''

for task in ('browse', 'state_truth', 'coverage'):
    PROMPTS[task] = COMPONENT_SUPPORT_RULE + PREVIOUS[task]
PROMPTS['state_citation'] = STATE_CITATION_COMPONENT_RULE + PREVIOUS['state_citation']
