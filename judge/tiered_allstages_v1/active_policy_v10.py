"""V18.12: observable cumulative coverage and retained-evidence accounting."""
from active_policy_v9 import PROMPTS as PREVIOUS, STATE_COMBINED_PROMPT


PROMPTS = dict(PREVIOUS)

COVERAGE_OBSERVABILITY_RULE = '''COVERAGE OBSERVABILITY RULE: The supplied coverage view is a complete materialized snapshot. Every evidence row contains readable text, and an empty evidence list is itself fully observable. Therefore never return null for any coverage atom. When no supplied evidence supports an atom, return false; an empty or irrelevant snapshot maps to unknown in code, not unobservable.\n'''

RETAINED_EVIDENCE_RULE = '''RETAINED EVIDENCE RULE: Judge the entire cumulative evidence list, not only the newest source. Evidence retained from an earlier snapshot remains available and must continue to contribute to every atom it supports. A newly added design paper, indirect source, null result, or unrelated source does not erase material support already supplied by another retained evidence row. Do not drop earlier supporting evidence_ids merely because the complete requested comparison is still absent.\n'''

PROMPTS['coverage'] = (
    COVERAGE_OBSERVABILITY_RULE
    + RETAINED_EVIDENCE_RULE
    + PREVIOUS['coverage']
)

