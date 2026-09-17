"""V18.3: code-mapped partial for Browse and State truth."""
from active_policy_v2 import (
    PROMPTS as PREVIOUS,
    JUDGMENT_OUTPUT_CONTRACT,
    STATUS_OUTPUT_CONTRACT,
)

PROMPTS=dict(PREVIOUS)

BINARY_RULE='''BINARY SCOPE DECISION RULE: Do not choose a final correct/partial/incorrect or direct/partial/unknown label. First decide material_help: true only when the supplied material supports at least one concrete part of the frozen requirement, not merely the same topic or drug name. Then decide complete_scope: true only when all material dimensions explicitly required by that requirement are satisfied, including applicable population, intervention, comparator, outcome, time and design. Use false when observable and absent. Use null for both fields only when genuinely unobservable. Code maps false/false to incorrect or unknown, true/false to partial, and true/true to correct or direct. One-side, within-option, add-on, mixed-population, surrogate-only, or subset-outcome evidence is never complete unless the frozen requirement itself explicitly asks only for that scope.\n'''
BROWSE_OUTPUT='''EXACT BROWSE OUTPUT CONTRACT: Return {"judgments":{"source_relevance":{"material_help":true|false|null,"complete_scope":true|false|null,"reason":"brief"},"scope_fidelity":{"material_help":true|false|null,"complete_scope":true|false|null,"reason":"brief"}},"target_requirement_ids":[],"auxiliary":AUX}. Every judgments object has exactly material_help, complete_scope, and reason. Do not output verdict, grade, status, direct, partial, unknown, correct, or incorrect inside judgments.\n'''
STATE_OUTPUT='''EXACT STATE-TRUTH OUTPUT CONTRACT: Return {"coverage":[{"id":"policy item ID","material_help":true|false|null,"complete_scope":true|false|null,"evidence_ids":[],"reason":"brief"}]}, exactly one row per policy item. Every row has exactly id, material_help, complete_scope, evidence_ids, and reason. Do not output status, verdict, grade, direct, partial, or unknown. material_help true requires actual evidence_ids. material_help false or null requires an empty evidence_ids list.\n'''

browse=PREVIOUS['browse'].replace(JUDGMENT_OUTPUT_CONTRACT,'',1)
browse=browse.replace(
    'Return {"judgments":{"source_relevance":GRADE,"scope_fidelity":GRADE},"target_requirement_ids":[],"auxiliary":AUX}. GRADE has verdict correct/partial/incorrect/unobservable and reason. ',
    '',1)
PROMPTS['browse']=BROWSE_OUTPUT+BINARY_RULE+browse

state=PREVIOUS['state_truth'].replace(STATUS_OUTPUT_CONTRACT,'',1)
state=state.replace(
    'Return {"coverage":[{"id":"policy item ID","status":"direct|partial|unknown|unobservable","evidence_ids":[],"reason":"brief"}]}. ',
    '',1)
PROMPTS['state_truth']=STATE_OUTPUT+BINARY_RULE+state
