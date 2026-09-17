# Evaluation and independent auditing

Separate three claims: structural validation, live execution, and semantic correctness. Unit tests or HTTP success establish neither supported citations nor a better policy.

Use held-out questions with frozen task requirements; do not expose private evaluation requirements to the policy. Compare pinned SFT and RL adapters using the same backbone, retrieval configuration and declared sampling settings. Retain first outputs, protocol failures, tool executions and exact captures.

Report stage scores and Final completeness/fidelity/citation independently, alongside tool usage, valid-generation rate, unresolved judgments and evidence provenance. Never silently discard failed rollouts or treat an unavailable reward as zero. A tiny paired sample and one optimizer update are integration checks, not statistical evidence of improvement.

An independent reviewer can inspect original question, fixed requirements, answer units, actual attached evidence, full Judge inputs and raw judgments. It should distinguish supported paraphrases, partially supported claims, unsupported inference and explicit contradiction. Missing evidence is not contradiction.

Keep audit originals and corrections immutable and versioned. Feed accepted corrections through trusted scoring authority and the same compiler gates. Human/ChatGPT review does not bypass token identity, provenance, group completeness or behavior-policy checks.
