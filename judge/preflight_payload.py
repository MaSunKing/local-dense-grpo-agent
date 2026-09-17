"""Read-only canonical input check, deliberately not a collector converter."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from verify_package import verify, ROOT

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', required=True, type=Path)
    parser.add_argument('--task-modes', type=Path, help='Caller-owned frozen registry, never Agent output')
    parser.add_argument('--task-modes-sha256', help='Independently recorded registry digest')
    args = parser.parse_args()
    frozen = verify()
    modes = None
    if args.task_modes:
        raw = args.task_modes.read_bytes()
        if not args.task_modes_sha256 or hashlib.sha256(raw).hexdigest() != args.task_modes_sha256.lower():
            raise ValueError('frozen registry hash mismatch/missing')
        modes = json.loads(raw)
    from pipeline import plan
    value = json.loads(args.payload.read_text(encoding='utf-8'))
    values = value if isinstance(value, list) else [value]
    if not values:
        raise ValueError('empty payload list')
    failures = 0
    for index, payload in enumerate(values):
        try:
            if any(r.get('kind') == 'stop' for r in payload.get('records', [])):
                raise ValueError('Stop requires an externally verified collector_lookup; no fabricated provenance accepted')
            result = plan(payload, frozen['config'], task_modes=modes)
            print(json.dumps(dict(index=index, status='interface_passed', tasks=list(result['tasks']),
                training_ready=False), ensure_ascii=False))
        except (ValueError, KeyError, TypeError, AssertionError) as exc:
            failures += 1
            # Do not echo arbitrary input, request body or exception text.
            print(json.dumps(dict(index=index, status='needs_adapter_or_input_review',
                error_type=type(exc).__name__, training_ready=False)))
    print(json.dumps(dict(api_calls=0, optimizer_updates=0, failures=failures,
        scope='canonical input interface only; not live or semantic acceptance')))
    sys.exit(1 if failures else 0)
