
"""Captured-support probabilities. No API calls or model loading on import."""
import math
from core import digest

DISTRIBUTION = "temperature_captured_support_v1"
IMPLEMENTATION = "explicit_masked_multinomial_v1"

def validate_support(rec):
    s = rec["sampling"]
    t = s["temperature"]
    if type(t) not in (int, float) or not math.isfinite(t) or t <= 0:
        raise ValueError("invalid temperature")
    if (s.get("distribution") != DISTRIBUTION
        or s.get("implementation") != IMPLEMENTATION
        or s.get("top_p") != 1.0 or s.get("top_k") != 0):
        raise ValueError("unsupported constrained distribution")
    grammar = s.get("grammar")
    if not isinstance(grammar, dict) or set(grammar) != {
        "engine", "version", "schema", "schema_digest"
    }:
        raise ValueError("missing grammar identity")
    if grammar["engine"] != "lm-format-enforcer":
        raise ValueError("unsupported grammar engine")
    if not isinstance(grammar["version"], str) or not grammar["version"]:
        raise ValueError("missing grammar version")
    if not isinstance(grammar["schema"], dict):
        raise ValueError("invalid schema")
    if grammar["schema_digest"] != digest(grammar["schema"]):
        raise ValueError("schema drift")

    support = rec["support"]
    if set(support) != {"vocab_size", "pool", "refs", "digest"}:
        raise ValueError("invalid support envelope")
    body = {k: v for k, v in support.items() if k != "digest"}
    if support["digest"] != digest(body):
        raise ValueError("support drift")
    vocab = support["vocab_size"]
    if type(vocab) is not int or vocab <= 0:
        raise ValueError("invalid vocabulary")
    pool, refs = support["pool"], support["refs"]
    if not isinstance(pool, dict) or not isinstance(refs, list):
        raise ValueError("invalid support table")
    if len(refs) != len(rec["output_ids"]):
        raise ValueError("support/token count mismatch")
    for key, ids in pool.items():
        if not isinstance(ids, list) or not ids:
            raise ValueError("empty support")
        if any(type(i) is not int or not 0 <= i < vocab for i in ids):
            raise ValueError("invalid allowed token")
        if ids != sorted(set(ids)) or key != digest(ids):
            raise ValueError("support identity mismatch")
    if any(not isinstance(key, str) or key not in pool for key in refs):
        raise ValueError("unknown support reference")
    if set(refs) != set(pool):
        raise ValueError("unused support entry")
    for token, key in zip(rec["output_ids"], refs):
        if token not in pool[key]:
            raise ValueError("sampled forbidden token")

def masked_logits(logits, allowed):
    import torch
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError("single sequence required")
    ids = sorted(set(allowed))
    if not ids or any(type(i) is not int or not 0 <= i < logits.shape[-1]
                      for i in ids):
        raise ValueError("invalid grammar support")
    index = torch.tensor(ids, device=logits.device, dtype=torch.long)
    result = torch.full_like(logits, -torch.inf)
    return result.index_copy(-1, index, logits.index_select(-1, index)), ids

def selected_supported_logps(logits, targets, positions, rec):
    """logits shape [1,N,V], after temperature; support independent of weights."""
    import torch
    if logits.shape[0] != 1 or logits.shape[1] != len(positions):
        raise ValueError("probability shape mismatch")
    if logits.shape[-1] != rec["support"]["vocab_size"]:
        raise ValueError("vocabulary drift")
    values = []
    support = rec["support"]
    for j, position in enumerate(positions):
        ids = support["pool"][support["refs"][position]]
        target = int(targets[0, j])
        if target not in ids:
            raise ValueError("target outside support")
        index = torch.tensor(ids, device=logits.device, dtype=torch.long)
        row = logits[0, j]
        values.append(row[target] - torch.logsumexp(row.index_select(0, index), 0))
    return torch.stack(values).unsqueeze(0)
