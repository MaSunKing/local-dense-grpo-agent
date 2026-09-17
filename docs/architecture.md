# Architecture and trust boundaries

One shared policy produces Checklist, tool decisions, State and Final completions. SFT teaches the interface; RL assigns separate channels to selected generated tokens. Search/Browse are policy actions; tool responses are observations, not generated tokens to optimize.

1. Freeze the original question, requirements, constraints, policy identity, tokenizer identity, scorer and reward configuration.
2. Capture the exact prompt, generated token IDs, sampling configuration and behavior log probabilities.
3. Execute tools and retain evidence text, source IDs, coordinates, hashes and execution receipts.
4. Plan stage-specific Judge views. Semantic judgments belong to the Judge; code validates schemas, bindings and allowed IDs.
5. Inherit the previous validated coverage receipt. Judge bounded new-evidence support and combine it deterministically.
6. Bind validated score/tool events to record and token spans, then compile relative advantages.
7. Optimize the shared LoRA using the behavior-policy distribution and an atomic output checkpoint.

Hashes establish integrity within a trusted authority boundary, not independent authentication. A rollout must not be allowed to manufacture its own trusted score or execution registry. The synthetic example explicitly uses test-only authorities; production integration must supply trusted lookups separately.

## Citation protocol

The public source retains full chunk IDs and the learned, claim-following citation format:

```xml
<answer>
Evidence-grounded statement.
<cite id="WEB:example#s0-c0">Source description</cite>
</answer>
```

State references evidence IDs; Final generates its own actual attachments. Code can validate tag structure, ID membership and attachment position. It cannot prove that a paraphrase is supported. Citation evaluates actual attached evidence; fidelity evaluates allowed evidence for claims present; completeness evaluates whether requested content was addressed. These are separate dimensions, not a single impression score.

No automatic citation insertion, short-ID format migration, model-weight merge or second answer-writing model is required by this release.

After configuring a real provider model in `judge/effective_config.json`, run `python -B tools/refreeze.py` and re-run the offline checks. That creates a new scorer identity; it does not certify semantic correctness or authorize old receipts under the new identity.

## Portable versus deployment-specific code

This repository exports the portable algorithm/contracts and tool backend. Private-fixture-specific cluster launchers and captured-data orchestration are not included. The root CLI demonstrates the compiler and executes tests; it is not a production rollout service. Deployments must connect their own capture/execution service while preserving the exported contracts.
