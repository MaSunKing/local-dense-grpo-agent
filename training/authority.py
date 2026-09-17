"""Read-only authority bundle used by the V18 training preflight/compiler."""
import copy
import hashlib
import json
from pathlib import Path

from core import digest


VERSION = "trusted_reward_authority_bundle_v1"
REGISTRIES = {
    "tool_executions", "stage_scores", "evidence_gains", "terminations",
    "execution_manifests", "policy_violations", "gain_dispositions",
    "coverage_receipts", "evidence_transitions",
}


class TrustedAuthorityBundle:
    def __init__(self, path, batch_bytes):
        self.path = Path(path).resolve()
        if not isinstance(batch_bytes, bytes):
            raise ValueError("authority validation requires the exact parsed batch bytes")
        value = json.loads(self.path.read_text(encoding="utf-8"))
        required = {"version", "authority_id", "batch_sha256", "registries"}
        if not isinstance(value, dict) or set(value) != required or value["version"] != VERSION:
            raise ValueError("authority bundle fields")
        actual = hashlib.sha256(batch_bytes).hexdigest()
        if value["batch_sha256"] != actual:
            raise ValueError("authority bundle is not bound to this exact batch file")
        registries = value["registries"]
        if not isinstance(registries, dict) or set(registries) != REGISTRIES or any(
                not isinstance(rows, dict) for rows in registries.values()):
            raise ValueError("authority registry fields")
        core = copy.deepcopy(value)
        core.pop("authority_id")
        if value["authority_id"] != digest(core):
            raise ValueError("authority bundle identity mismatch")
        self.value = value

    def _get(self, registry, key):
        value = self.value["registries"][registry].get(key)
        return copy.deepcopy(value)

    def tool_execution(self, key):
        return self._get("tool_executions", key)

    def stage_score(self, key):
        return self._get("stage_scores", key)

    def evidence_gain(self, key):
        return self._get("evidence_gains", key)

    def termination(self, key):
        return self._get("terminations", key)

    def execution_manifest(self, rollout_id):
        return self._get("execution_manifests", rollout_id)

    def policy_violation(self, key):
        return self._get("policy_violations", key)

    def gain_disposition(self, tool_execution_id):
        return self._get("gain_dispositions", tool_execution_id)

    def coverage_receipt(self, coverage_key):
        return self._get("coverage_receipts", coverage_key)

    def evidence_transition(self, tool_execution_id):
        return self._get("evidence_transitions", tool_execution_id)


def build_for_test(batch_path, registries):
    """Deterministic constructor for offline tests; production assembler stays external."""
    if set(registries) != REGISTRIES:
        raise ValueError("authority registry fields")
    core = {"version": VERSION,
            "batch_sha256": hashlib.sha256(Path(batch_path).read_bytes()).hexdigest(),
            "registries": copy.deepcopy(registries)}
    return {"authority_id": digest(core), **core}
