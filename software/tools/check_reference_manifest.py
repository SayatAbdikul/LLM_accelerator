"""Verify the frozen golden execution dependencies and gen-2 fixture bytes.

This checker never refreshes pins. A deliberate reference revision must review
the changed source, regenerate fixtures, and update the checked-in manifest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "software/tests/fixtures/gen2_reference_manifest.json"


def reference_files(root: Path = REPO_ROOT) -> list[Path]:
    paths = [root / "software/taccel/__init__.py"]
    for package in ("golden_model", "isa", "assembler", "utils"):
        paths.extend((root / "software/taccel" / package).glob("*.py"))
    paths.append(root / "software/tools/gen_gen2_fixtures.py")
    paths.extend((root / "rtl/verilator/fixtures/gen2").rglob("*.raw"))
    paths.extend((root / "rtl/verilator/fixtures/gen2").rglob("meta.json"))
    return sorted(paths)


def current_hashes(root: Path = REPO_ROOT) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in reference_files(root)}


def verify_reference_manifest(root: Path = REPO_ROOT, manifest_path: Path = MANIFEST) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported golden reference manifest version")
    expected = manifest["sha256"]
    actual = current_hashes(root)
    differences = sorted(path for path in expected.keys() | actual.keys()
                         if expected.get(path) != actual.get(path))
    if differences:
        raise ValueError("FROZEN GOLDEN/FIXTURE DRIFT: " + ", ".join(differences))
    return manifest


if __name__ == "__main__":
    verified = verify_reference_manifest()
    print(f"Verified {len(verified['sha256'])} frozen source/fixture files")
