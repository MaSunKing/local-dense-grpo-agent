"""Offline verification. No network, secrets, model imports or optimizer."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def verify():
    manifest = json.loads((ROOT / 'PACKAGE_MANIFEST.json').read_text())
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
    if any(p.is_symlink() for p in ROOT.rglob('*')):
        raise ValueError('symlink forbidden')
    if actual != set(manifest) | {'PACKAGE_MANIFEST.json'}:
        raise ValueError('package inventory mismatch; use python -B')
    for name, expected in manifest.items():
        p = ROOT / name
        if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError('integrity mismatch: ' + name)
        if p.suffix == '.py':
            compile(p.read_bytes(), name, 'exec')
    sys.path.insert(0, str(ROOT / 'tiered_allstages_v1'))
    from pipeline import profile
    config = json.loads((ROOT / 'effective_config.json').read_text())
    current = profile(config)
    expected = json.loads((ROOT / 'frozen_profile.json').read_text())
    if current != expected:
        raise ValueError('effective scorer/source mismatch')
    return current

if __name__ == '__main__':
    profile = verify()
    suites = [('fixed', 'test_multicite_binding.py'),('tiered_allstages_v1', 'test_stop_context_v112.py'),('tiered_allstages_v1', 'test_evidence_gain.py'),('tiered_allstages_v1', 'test_engine.py'),
              ('tiered_allstages_v1', 'test_revision.py'),
              ('tiered_allstages_v1', 'test_binary_scope_v183.py'),
              ('tiered_allstages_v1', 'test_contract_retry_v184.py'),
              ('tiered_allstages_v1', 'test_component_support_v185.py'),
              ('tiered_allstages_v1', 'test_atomic_support_v186.py'),
              ('tiered_allstages_v1', 'test_state_coalescing_v189.py'),
              ('tiered_allstages_v1', 'test_coverage_atomic_v1811.py'),
              ('tiered_allstages_v1', 'test_coverage_observability_v1812.py'),
              ('tiered_allstages_v1', 'test_retained_coverage_v1813.py'),
              ('tiered_allstages_v1', 'test_portable.py'),
              ('tiered_allstages_v1', 'test_v11_regressions.py'),
              ('tiered_judge_v1', 'test_tiered.py'),
              ('tiered_judge_v2', 'test_isolated.py'),
              ('tiered_judge_v2', 'test_citation_observability_v1810.py')]
    for directory, test in suites:
        subprocess.run([sys.executable, '-B', '-m', 'unittest', '-v', test],
                       cwd=ROOT / directory, check=True)
    print(json.dumps(dict(status='passed', files=len(json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text())),
        scorer=profile['scorer'], api_calls=0, optimizer_updates=0,
        collector_adapter_validated=False, training_ready=False), indent=2))
