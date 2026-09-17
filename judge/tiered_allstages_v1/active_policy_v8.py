"""V18.9: coalesce State truth/citation transport without merging their scores."""
from active_policy_v7 import PROMPTS


STATE_COMBINED_PROMPT = '''STATE TRANSPORT COALESCING CONTRACT:
Return exactly one JSON object with exactly two fields:
{"state_truth":STATE_TRUTH_RESULT,"state_citation":STATE_CITATION_RESULT}.

The user input contains state_truth_view and state_citation_view. Treat these as
two isolated evaluation compartments carried in one HTTP request:

1. Produce state_truth only from state_truth_view. Do not use, quote, infer, or
   react to state_citation_view when deciding state_truth.
2. Produce state_citation only from each item's ACTUAL attached evidence in
   state_citation_view. Do not use evidence attached to another item.
3. The Agent's claimed direct/partial/unknown labels are deliberately absent.
   Code compares the validated atoms with those claims after this response.
4. Do not average, reconcile, or force agreement between the two outputs.
5. Follow both exact nested output contracts below. Do not add wrapper metadata,
   scores, explanations, markdown, aliases, or extra fields.

STATE_TRUTH_RESULT contract and rules:
''' + PROMPTS['state_truth'] + '''

STATE_CITATION_RESULT contract and rules:
''' + PROMPTS['state_citation']
