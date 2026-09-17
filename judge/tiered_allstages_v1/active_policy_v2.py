"""Generic missing-data versus measured-null distinction; no fixture-specific terms."""
from active_policy import PROMPTS as PREVIOUS
PROMPTS=dict(PREVIOUS)
MISSING_VS_NEGATIVE='''STATE/EVIDENCE STATUS CONTRACT: Missing data are NOT a negative or null measured finding. First determine what the item asks for. If it asks for actual values/results of A AND B, an explicit statement that B was not measured does not supply B's result: available A with missing B supports only partial. If no requested result is available, unknown is appropriate. Conversely, if the item explicitly asks WHETHER data exist, WHETHER an outcome was measured/reported, or whether evidence is sufficient, an explicit absence statement can directly answer that availability question. A measured zero/null/negative effect is a genuine result and may directly answer an outcome question; no measurement is not that result. Apply this distinction to the actual item semantics, not keyword triggers. Do not fill missing outcome values from acknowledgments of their absence.\n'''
for task in ('state_truth','state_citation','coverage'):
    PROMPTS[task]=MISSING_VS_NEGATIVE+PREVIOUS[task]

BROWSE_COMPARATIVE_SCOPE='''COMPARATIVE OR MULTI-OPTION BROWSE CONTRACT: First identify only the material dimensions explicitly required by the frozen requirement, such as the target entity or population, option A, option B, outcome, time, setting, subgroup, method, source type, or jurisdiction. Do not require dimensions that the task does not contain. A candidate likely to answer the complete requested comparison may be correct. A candidate covering only one option, a within-option comparison, an add-on comparison, an indirect comparator, one requested outcome, subgroup, or time point is normally partial when the visible title or snippet shows that it can materially advance the requirement. Do not mark it incorrect solely because it cannot establish the final comparison. Such evidence must never be promoted to a complete comparative conclusion. Incorrect requires no material value for the requested dimensions, a materially wrong target/outcome, or an action that falsely characterizes the source. For scope_fidelity, an honestly selected indirect source is not automatically incorrect; grade the material scope retained versus missing.\n'''

COMPARATIVE_COVERAGE_SCOPE='''COMPARATIVE OR MULTI-OPTION COVERAGE CONTRACT: Identify only the material dimensions explicitly requested by the item or frozen requirement. direct requires evidence sufficient to answer the complete requested scope; it does not require a flawless study. partial means the evidence materially advances only part of the scope, including one option, a within-option comparison, an add-on comparison, an indirect comparator, one requested outcome, subgroup, condition, or time point. unknown means the evidence provides no material information that advances the requested scope. Missing one required dimension does not by itself imply unknown when the remaining evidence is materially useful. Do not promote partial evidence to a complete comparative conclusion. A valid comparative estimate, including a suitable indirect estimate, may be direct unless the task explicitly requires head-to-head evidence.\n'''

STATE_CITATION_COMPARATIVE_SCOPE='''STATE CITATION COMPARATIVE CONTRACT: Judge support for the Agent's claimed_status, not support for a stronger unstated conclusion. When claimed_status is partial, cited one-option, within-option, add-on, indirect-comparator, partial-outcome, subgroup, condition, or time-point evidence can correctly support that partial status if it materially advances the item. It does not support a direct status or a complete comparison. Mark incorrect when the citation supplies no material information for the item or the claimed status overstates its scope.\n'''

PROMPTS['browse']=BROWSE_COMPARATIVE_SCOPE+PROMPTS['browse']
for task in ('state_truth','coverage'):
    PROMPTS[task]=COMPARATIVE_COVERAGE_SCOPE+PROMPTS[task]
PROMPTS['state_citation']=STATE_CITATION_COMPARATIVE_SCOPE+PROMPTS['state_citation']

# Core and auxiliary labels deliberately use different field names.  Spell out
# every core row shape so a JSON-mode model cannot infer the auxiliary `grade`
# field for a core decision.  The validator remains fail-closed and does not
# accept aliases.
CHECKLIST_OUTPUT_CONTRACT='''EXACT CHECKLIST OUTPUT CONTRACT: Every coverage row has exactly id, verdict, item_ids, and reason. The scope object has exactly verdict and reason. Use the field name verdict for both. Never use the field name grade in coverage rows or scope. The field grade is reserved only for the optional top-level auxiliary object.\n'''
JUDGMENT_OUTPUT_CONTRACT='''EXACT CORE OUTPUT CONTRACT: Every object inside judgments has exactly two fields named verdict and reason, for example {"verdict":"partial","reason":"brief explanation"}. Never output a field named grade inside judgments. The field grade is reserved only for the optional top-level auxiliary object.\n'''
STATE_CITATION_OUTPUT_CONTRACT='''EXACT STATE-CITATION OUTPUT CONTRACT: Every items row has exactly id, verdict, evidence_ids, and reason. Use the field name verdict, never grade. The field grade is reserved only for an optional top-level auxiliary object when one is requested.\n'''
STATUS_OUTPUT_CONTRACT='''EXACT COVERAGE OUTPUT CONTRACT: Every coverage row has exactly id, status, evidence_ids, and reason. Use the field name status, never verdict or grade, for the coverage label.\n'''

PROMPTS['checklist']=CHECKLIST_OUTPUT_CONTRACT+PROMPTS['checklist']
for task in ('search','browse','stop'):
    PROMPTS[task]=JUDGMENT_OUTPUT_CONTRACT+PROMPTS[task]
PROMPTS['state_citation']=STATE_CITATION_OUTPUT_CONTRACT+PROMPTS['state_citation']
for task in ('state_truth','coverage'):
    PROMPTS[task]=STATUS_OUTPUT_CONTRACT+PROMPTS[task]
