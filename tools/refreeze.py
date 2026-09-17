"""Re-freeze edited public Judge sources; not a semantic acceptance or audit."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    sys.path.insert(0, str(ROOT / "judge/tiered_allstages_v1"))
    from pipeline import profile
    config = json.loads((ROOT / "judge/effective_config.json").read_text())
    frozen = profile(config)
    (ROOT / "judge/frozen_profile.json").write_text(
        json.dumps(frozen, indent=2) + "\n", encoding="utf-8", newline="\n")
    manifest = {p.relative_to(ROOT / "judge").as_posix():
                hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((ROOT / "judge").rglob("*"))
                if p.is_file() and p.name != "PACKAGE_MANIFEST.json"}
    (ROOT / "judge/PACKAGE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "refrozen", "scorer": frozen["scorer"], "api_calls": 0}))


if __name__ == "__main__":
    main()
