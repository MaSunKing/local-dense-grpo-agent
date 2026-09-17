"""V18.11: atomic cumulative-coverage decisions for evidence gain."""
from active_policy_v8 import PROMPTS as PREVIOUS, STATE_COMBINED_PROMPT
from active_policy_v7 import PRESENCE_RULE


PROMPTS = dict(PREVIOUS)

COVERAGE_ATOMIC_OUTPUT = '''EXACT EVIDENCE-COVERAGE JSON OUTPUT CONTRACT: Return {"coverage":[{"id":"frozen requirement ID","any_requested_option_or_member_present":true|false|null,"requested_outcome_support":true|false|null,"population_applicable":true|false|null,"complete_requirement_support":true|false|null,"evidence_ids":[],"reason":"brief"}]}, exactly one row per frozen requirement. Every row has exactly id, the four named support atoms, evidence_ids, and reason. Do not output status, material_help, complete_scope, verdict, grade, direct, partial, unknown, correct, or incorrect. Judge cumulative support in the supplied evidence only. When the first three atoms are true, cite the actual evidence_ids that establish that material support. Otherwise evidence_ids must be empty. For a genuinely unobservable decision use null for all four atoms and no evidence_ids.\n'''

COVERAGE_ATOMIC_RULE = '''CUMULATIVE COVERAGE RULE: Evaluate the four atoms independently. any_requested_option_or_member_present is true when the supplied evidence concerns at least one requested option, comparator, class, intervention, or identifiable member. requested_outcome_support is true when it reports at least one outcome, safety event, or other fact requested by the requirement. population_applicable is true when the studied population matches or is materially applicable; incomplete reporting alone is not a mismatch. complete_requirement_support is true only when the cumulative evidence supplies the complete requested comparison and every material scope condition. Evidence about only one requested option can be materially useful: keep the first three atoms true when supported and set complete_requirement_support=false. Code, not the Judge, maps the atoms to unknown, partial, direct, or unobservable. Do not infer an absent arm, outcome, population, or comparative effect. Source metadata alone does not establish factual coverage.\n'''

PROMPTS['coverage'] = COVERAGE_ATOMIC_OUTPUT + PRESENCE_RULE + COVERAGE_ATOMIC_RULE
