"""V18.6: expose factual support atoms; code derives material/complete labels."""
from active_policy_v4 import (
    PROMPTS as PREVIOUS,
    COMPONENT_SUPPORT_RULE,
    STATE_CITATION_COMPONENT_RULE,
)
from active_policy_v3 import BINARY_RULE, BROWSE_OUTPUT, STATE_OUTPUT


PROMPTS = dict(PREVIOUS)

ATOMIC_RULE = '''ATOMIC SUPPORT RULE: Do not decide material_help, complete_scope, direct, partial, unknown, correct, or incorrect. Report only the requested factual atoms. requested_option_support is true when the visible material concerns at least one requested option, class, or clearly identified member; the other requested option is not required for this atom. requested_outcome_support is true when it visibly concerns at least one outcome, safety event, or other fact requested by the evaluated requirement. population_applicable is true when the population matches or is materially applicable; false requires a material mismatch, not merely incomplete reporting. complete_requirement_support is true only when the material supplies the complete requested comparison and every material scope condition. Missing the other comparison arm makes only complete_requirement_support false. Code derives no-help, partial, or complete from these atoms. For an unobservable decision, use null for all four atoms. Do not rank a candidate against alternatives in these core atoms; selection efficiency is auxiliary only. Never infer a missing arm, outcome, population, or comparative effect.\n'''

BROWSE_ATOMIC_OUTPUT = '''EXACT BROWSE OUTPUT CONTRACT: Return {"judgments":{"source_relevance":{"requested_option_support":true|false|null,"requested_outcome_support":true|false|null,"population_applicable":true|false|null,"complete_requirement_support":true|false|null,"reason":"brief"},"scope_fidelity":{"requested_option_support":true|false|null,"requested_outcome_support":true|false|null,"population_applicable":true|false|null,"complete_requirement_support":true|false|null,"reason":"brief"}},"target_requirement_ids":[],"auxiliary":AUX}. Every judgments object has exactly the four named support atoms and reason. Do not output material_help, complete_scope, verdict, grade, status, direct, partial, unknown, correct, or incorrect inside judgments. At Browse time, use only the visible title, snippet, source type, selected candidate, and focus. requested_outcome_support may reflect a visibly named outcome category or a concrete prospective outcome signal; it does not claim that the unread body contains a result.\n'''

STATE_ATOMIC_OUTPUT = '''EXACT STATE-TRUTH OUTPUT CONTRACT: Return {"coverage":[{"id":"policy item ID","requested_option_support":true|false|null,"requested_outcome_support":true|false|null,"population_applicable":true|false|null,"complete_requirement_support":true|false|null,"evidence_ids":[],"reason":"brief"}]}, exactly one row per policy item. Every row has exactly id, the four named support atoms, evidence_ids, and reason. Do not output status, material_help, complete_scope, verdict, grade, direct, partial, or unknown. When the first three atoms are true, cite actual evidence_ids. Otherwise evidence_ids must be empty.\n'''

STATE_CITATION_ATOMIC_OUTPUT = '''EXACT STATE-CITATION OUTPUT CONTRACT: Return {"items":[{"id":"policy item ID","requested_option_support":true|false|null,"requested_outcome_support":true|false|null,"population_applicable":true|false|null,"complete_requirement_support":true|false|null,"evidence_ids":[],"reason":"brief"}]}, exactly one row per supplied item. Every row has exactly id, the four named support atoms, evidence_ids, and reason. Do not output verdict, grade, status, correct, partial, or incorrect. Judge only the ACTUAL evidence attached to that item. When the first three atoms are true, cite actual attached evidence_ids. Otherwise evidence_ids must be empty. Code evaluates these atoms against claimed_status: partial needs concrete support for one requested option/outcome in an applicable population; direct additionally needs complete_requirement_support=true.\n'''


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
PROMPTS['state_citation'] = STATE_CITATION_ATOMIC_OUTPUT + ATOMIC_RULE + _strip(
    PREVIOUS['state_citation'], STATE_CITATION_COMPONENT_RULE
)
