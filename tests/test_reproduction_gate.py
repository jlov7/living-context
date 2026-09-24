from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest
import yaml

from scripts.check_reproduction import (
    ReproductionError,
    saved_canary_diagnostic,
    validate_pilot,
    verify_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def _public_subset_fixture(tmp_path: Path) -> tuple[Path, dict]:
    evidence_dir = tmp_path / "artifacts" / "evidence"
    evidence_dir.mkdir(parents=True)
    original = ROOT / "artifacts" / "evidence" / "SHA256SUMS.json"
    (evidence_dir / "SHA256SUMS.json").write_bytes(original.read_bytes())
    public = json.loads((ROOT / "artifacts" / "evidence" / "PUBLIC-SHA256SUMS.json").read_text())
    return evidence_dir / "PUBLIC-SHA256SUMS.json", public


def _inputs() -> tuple[dict, dict]:
    evidence = json.loads((ROOT / "artifacts/evidence/pilot-1790032618.json").read_text())
    protocol = yaml.safe_load((ROOT / "experiments/pilot.yaml").read_text())
    return evidence, protocol


def test_retained_evidence_hashes_and_strict_rescore() -> None:
    verified = verify_manifest()
    evidence, protocol = _inputs()
    result = validate_pilot(evidence, protocol)
    assert result["historical_substring_score"] == {"arm_1": 9, "arm_4": 6, "n": 9}
    assert result["saved_output_boundary_diagnostic"] == {
        "arm_1": 9,
        "arm_4": 0,
        "n": 9,
    }
    assert saved_canary_diagnostic(verified) == {
        "retained_attempts": 7,
        "historical_substring_passes": 5,
        "saved_output_boundary_passes": 1,
        "baseline_boundary_passes": 0,
    }


def test_reproduction_rejects_missing_declared_row() -> None:
    evidence, protocol = _inputs()
    evidence["rows"].pop()
    with pytest.raises(ReproductionError, match="missing 1 declared rows"):
        validate_pilot(evidence, protocol)


def test_reproduction_rejects_tampered_correctness_flag() -> None:
    evidence, protocol = _inputs()
    tampered = copy.deepcopy(evidence)
    tampered["rows"][0]["arm_4_correct"] = False
    with pytest.raises(ReproductionError, match="disagrees with old scorer"):
        validate_pilot(tampered, protocol)


def test_manifest_rejects_unhashed_pilot_selection(tmp_path: Path) -> None:
    evidence = b"{}"
    (tmp_path / "evidence.json").write_bytes(evidence)
    manifest = {
        "pilot_evidence": "unlisted.json",
        "pilot_protocol": "evidence.json",
        "files": {"evidence.json": hashlib.sha256(evidence).hexdigest()},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ReproductionError, match="must select a hashed"):
        verify_manifest(root=tmp_path, manifest_path=manifest_path)


def test_public_subset_rejects_any_additional_scientific_omission(tmp_path: Path) -> None:
    manifest_path, public = _public_subset_fixture(tmp_path)
    public["files"].pop("artifacts/evidence/pilot-1790032618.json")
    manifest_path.write_text(json.dumps(public))
    with pytest.raises(ReproductionError, match="differs from the approved evidence inventory"):
        verify_manifest(root=tmp_path, manifest_path=manifest_path)


def test_public_subset_rejects_changed_omitted_file(tmp_path: Path) -> None:
    manifest_path, public = _public_subset_fixture(tmp_path)
    public["omitted"]["path"] = "artifacts/evidence/pilot-1790032618.json"
    manifest_path.write_text(json.dumps(public))
    with pytest.raises(ReproductionError, match="omits an unapproved file"):
        verify_manifest(root=tmp_path, manifest_path=manifest_path)


def test_public_subset_rejects_changed_pilot_selector(tmp_path: Path) -> None:
    manifest_path, public = _public_subset_fixture(tmp_path)
    public["pilot_evidence"] = "artifacts/evidence/canary-1790010402.json"
    manifest_path.write_text(json.dumps(public))
    with pytest.raises(ReproductionError, match="changed pilot selection"):
        verify_manifest(root=tmp_path, manifest_path=manifest_path)


def test_public_subset_rejects_changed_original_manifest(tmp_path: Path) -> None:
    manifest_path, public = _public_subset_fixture(tmp_path)
    manifest_path.write_text(json.dumps(public))
    original_path = tmp_path / "artifacts" / "evidence" / "SHA256SUMS.json"
    original_path.write_bytes(original_path.read_bytes() + b"\n")
    with pytest.raises(ReproductionError, match="original evidence manifest changed"):
        verify_manifest(root=tmp_path, manifest_path=manifest_path)


def test_public_subset_rejects_tampered_scientific_file(tmp_path: Path) -> None:
    manifest_path, public = _public_subset_fixture(tmp_path)
    manifest_path.write_text(json.dumps(public))
    for relative in public["files"]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    pilot = tmp_path / public["pilot_evidence"]
    pilot.write_bytes(pilot.read_bytes() + b"\n")
    with pytest.raises(ReproductionError, match="hash mismatch for"):
        verify_manifest(root=tmp_path, manifest_path=manifest_path)


def test_reproduction_rejects_nonfinite_measurement() -> None:
    evidence, protocol = _inputs()
    evidence["rows"][0]["arm_1_serving_s"] = math.inf
    with pytest.raises(ReproductionError, match="must be finite"):
        validate_pilot(evidence, protocol)
