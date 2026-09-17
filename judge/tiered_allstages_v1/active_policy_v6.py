"""V18.7: remove ambiguity between one requested option and all options."""
from active_policy_v4 import (
    PROMPTS as PREVIOUS,
    COMPONENT_SUPPORT_RULE,
    STATE_CITATION_COMPONENT_RULE,
)
from active_policy_v3 import BINARY_RULE, BROWSE_OUTPUT, STATE_OUTPUT
from active_policy_v2 import MISSING_VS_NEGATIVE, STATE_CITATION_COMPARATIVE_SCOPE
from prompts import BASE


PROMPTS = dict(PREVIOUS)

ATOMIC_RULE = '''ATOMIC SUPPORT RULE: Do not decide material_help, complete_scope, direct, partial, unknown, correct, or incorrect. Report only the requested factual atoms. one_or_more_requested_options_supported is true when the visible material concerns at least one requested option, class, intervention, comparator, or clearly identified member. It does not require every requested option, and absence of the other comparison arm must not make this atom false. requested_outcome_support is true when the material visibly concerns at least one outcome, safety event, or other fact requested by the evaluated requirement. population_applicable is true when the population matches or is materially applicable; false requires a material mismatch, not merely incomplete reporting. complete_requirement_support is true only when the material supplies the complete requested comparison and every material scope condition. A missing comparison arm makes complete_requirement_support false while one_or_more_requested_options_supported remains true. Code derives no-help, partial, or complete from these atoms. For an unobservable decision, use null for all four atoms. Do not rank a candidate against alternatives in these core atoms; selection efficiency is auxiliary only. Never infer a missing arm, outcome, population, or comparative effect.\n'''

BROWSE_ATOMIC_OUTPUT = '''EXACT BROWSE OUTPUT CONTRACT: Return {"judgments":{"source_relevance":{"one_or_more_requested_options_supported":true|false|null,"requested_outcome_support":true|false|null,"population_applicable":true|false|null,"complete_requirement_support":true|false|null,"reason":"brief"},"scope_fidelity":{"one_or_more_requested_options_supported":true|false|null,"requested_outcome_support":true|false|null,"population_applicable":true|false|null,"complete_requirement_support":true|false|null,"reason":"brief"}},"target_requirement_ids":[],"auxiliary":AUX}. Every judgments object has exactly the four named support atoms and reason. Do not output requested_option_support, material_help, complete_scope, verdict, grade, status, direct, partial, unknown, correct, or incorrect inside judgments. At Browse time, use only the visible title, snippet, source type, selected candidate, and focus. requested_outcome_support may reflect a visibly named outcome category or a concrete prospective outcome signal; it does not claim that the unread body contains a result. If only one requested option is visible, set one_or_more_requested_options_supported=true and complete_requirement_support=false.\n'''

STATE_ATOMIC_OUTPUT = '''EXACT STATE-TRUTH OUTPUT CONTRACT: Return {"coverage":[{"id":"policy item ID","one_or_more_requested_options_supported":true|false|null,"requested_outcome_support":true|false|null,"population_applicable":true|false|null,"complete_requirement_support":true|false|null,"evidence_ids":[],"reason":"brief"}]}, exactly one row per policy item. Every row has exactly id, the four named support atoms, evidence_ids, and reason. Do not output requested_option_support, status, material_help, complete_scope, verdict, grade, direct, partial, or unknown. When the first three atoms are true, cite actual evidence_ids. Otherwise evidence_ids must be empty. Evidence for one requested option and a requested outcome in an applicable population sets one_or_more_requested_options_supported=true even when the other comparison arm is absent.\n'''

STATE_CITATION_ATOMIC_OUTPUT = '''EXACT STATE-CITATION OUTPUT CONTRACT: Return {"items":[{"id":"policy item ID","one_or_more_requested_options_supported":true|false|null,"requested_outcome_support":true|false|null,"population_applicable":true|false|null,"complete_requirement_support":true|false|null,"evidence_ids":[],"reason":"brief"}]}, exactly one row per supplied item. Every row has exactly id, the four named support atoms, evidence_ids, and reason. Do not output requested_option_support, verdict, grade, status, correct, partial, or incorrect. Judge only the ACTUAL evidence attached to that item. When the first three atoms are true, cite actual attached evidence_ids. Otherwise evidence_ids must be empty. Code evaluates these atoms against claimed_status: a partial claim needs concrete support for at least one requested option and outcome in an applicable population; a direct claim additionally needs complete_requirement_support=true. Missing the other comparison arm does not make one_or_more_requested_options_supported false.\n'''

STATE_CITATION_TASK_RULE = '''For each items entry, assess only whether its ACTUAL attached evidence supplies the four factual support atoms for that item's description. Each item has its own attached evidence list. References may only come from that list, not other items or future sources. Do not repair citations or item scope. The code, not the Judge, compares these atoms with claimed_status.\n'''


def _strip(text, *parts):
    for part in parts:
        if part in text:
            text = text.replace(part, '', 1)
    return text


PROMPTS['browse'] = BROWSE_ATOMIC_OUTPUT + ATOMIC_RULE + _strip(
    PREVIOUS['browse'], COMPONENT_SUPPORT_RULE, BROWSE_OUTPUT, BINARY_RULE
)
PROMPTS['state_truth'] = STATE_ATOMIC_OUTPUT + ATOMIC_RULE + _strip(
    PREVIOUS['state_truth'], COMPONENT_SUPPORT_RULE, STATE_OUTPUT, BINARY_RULE
)
PROMPTS['coverage'] = PREVIOUS['coverage']
PROMPTS['state_citation'] = (
    STATE_CITATION_ATOMIC_OUTPUT
    + ATOMIC_RULE
    + STATE_CITATION_COMPONENT_RULE
    + STATE_CITATION_COMPARATIVE_SCOPE
    + MISSING_VS_NEGATIVE
    + BASE
    + STATE_CITATION_TASK_RULE
)
