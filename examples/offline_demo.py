"""Synthetic four-rollout example using the real compiler; no semantic inference."""
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def run():
    sys.path.insert(0, str(ROOT / "training"))
    sys.path.insert(0, str(ROOT / "shared"))
    from test_joint import fixture, compile_fixture
    from gain_contract import canonical_coverage_delta_view, merge_coverage_delta

    rows = compile_fixture(fixture())
    assert any(row["advantage"] < 0 for row in rows)
    assert any(row["advantage"] > 0 for row in rows)
    view = canonical_coverage_delta_view({
        "question": "Compare option A with B for outcome O.",
        "requirements": [{"id": "R1", "description": "Compare O", "weight": 1}],
        "constraints": [],
        "prior_coverage": [{"id": "R1", "status": "partial",
                            "evidence_ids": ["E1"], "reason": "Synthetic prior support"}],
        "retained_evidence": [{"source_id": "E1", "text": "Option A reports O."}],
        "new_evidence": [{"source_id": "E2", "text": "Unrelated page navigation."}],
        "source_headers": {},
    })
    judgment = {"coverage": [{
        "id": "R1", "any_requested_option_or_member_present": False,
        "requested_outcome_support": False, "population_applicable": False,
        "combined_requirement_complete": False, "contradiction": False,
        "basis_new_evidence_ids": [], "reason": "Synthetic: no new support",
    }]}
    merged, contradictions = merge_coverage_delta(view, judgment)
    assert merged["coverage"][0]["status"] == "partial"
    assert merged["coverage"][0]["evidence_ids"] == ["E1"]
    assert not contradictions
    return {
        "status": "passed", "fixture": "synthetic_not_model_inference",
        "questions": 1, "rollouts": 4,
        "compiled_channel_rows": dict(sorted(Counter(r["channel"] for r in rows).items())),
        "final_advantages": [{"rollout": r["rollout_id"], "advantage": r["advantage"]}
                             for r in rows if r["channel"] == "final"],
        "negative_and_positive_credit": True,
        "incremental_coverage": {"before": "partial", "after": "partial", "gain": 0.0},
        "api_calls": 0, "optimizer_updates": 0,
    }
