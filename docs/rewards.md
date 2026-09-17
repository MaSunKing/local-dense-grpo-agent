# Dense rewards and credit assignment

The semantic channels are Checklist, Search query, Browse source focus, State, Final and eligible Stop. Tool credit is a separate task channel. The example frozen configuration is `training/config.json`.

| Signal | Meaning |
|---|---|
| Checklist | Coverage and fidelity to the original task scope |
| Search | Query relevance to the current information gap |
| Browse source focus | Ex-ante quality of the selected source, before reading it |
| State | Evidence-status assessment and its actual evidence attachments |
| Final | Completeness, evidence fidelity and actual citation support |
| Evidence gain | Code-computed before/after coverage difference from trusted receipts |
| Tool cost/policy event | Verified execution cost or authority-backed violation |
| Stop | Trusted, evaluable voluntary termination; forced termination is N/A |

Browse selection quality is never retroactively replaced by evidence gain. Source identity, schema validity and provenance are gates, not positive reward sources. Pending/unobservable values are not silently converted to zero. An incomplete required core reward prevents its fixed four-rollout question group from entering advantage computation.

## Incremental coverage

`shared/gain_contract.py` fixes the prior receipt as immutable state. The next step uses the previous after-receipt as before-state and examines retained/new evidence under frozen requirements. Atomic judgments describe option/member presence, requested outcomes, applicable population, combined completeness, contradiction and new supporting IDs.

Code maps coverage to `unknown=0`, `partial=0.5`, `direct=1`. Irrelevant new evidence preserves earlier partial support with zero gain. Material support can advance unknown→partial; verified combined completeness can advance partial→direct. Contradiction is recorded separately, not rewarded as new support. Evidence invalidation requires a separate trusted event rather than a free-form downgrade.

## Advantages and objective

For each question, compare each rollout with the other rollouts (leave-one-out). Local channels use the configured channel baseline; task credit accounts for eligible future gain/cost events. Conceptually:

```text
A(channel) = local_weight × local_relative_credit
           + final_weight × final_relative_credit
ratio = exp(current_logprob - behavior_logprob)
loss = -mean(min(ratio × A, clip(ratio, 1-epsilon, 1+epsilon) × A))
```

This implementation is GRPO-style group-relative optimization, not a claim that every GRPO paper's normalization is reproduced. See `training/core.py` for the precise baselines and row weights. In the example configuration only Tool receives `0.5 × Final` propagation; not every earlier stage receives Final credit.

Even a nonnegative stage score can produce negative relative advantage. Verified tool costs and policy penalties also provide negative credit. A near-zero mean loss at the initial behavior-policy ratio does not imply zero gradient.

Behavior log probabilities and training replay must use the same declared sampling distribution/processors. Reward events bind to exact generated-token spans; tool observation text and nearby guessed tokens are not valid substitutes.
