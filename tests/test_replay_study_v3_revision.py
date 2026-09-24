"""Saved-output replay must fail beyond simple evidence hash checks."""

import json
import shutil
from pathlib import Path

import pytest

from living_context.catalog import VersionedCorpus
from scripts.replay_study_v3_revision import digest, replay

ROOT = Path(__file__).resolve().parents[1]


def copy_public_capsule(destination: Path) -> Path:
    study = ROOT / "research/study-v3-revision"
    protocol = json.loads((study / "protocol.json").read_text())
    for relative in protocol["source_sha256"]:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    frozen_catalog = Path("research/study-v3-revision/frozen-source/src/living_context/catalog.py")
    target = destination / frozen_catalog
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / frozen_catalog, target)
    target_study = destination / "research/study-v3-revision"
    shutil.copyfile(study / "protocol.json", target_study / "protocol.json")
    shutil.copyfile(study / "RESULTS.json", target_study / "RESULTS.json")
    shutil.copyfile(
        study / "PUBLIC-EVIDENCE-MANIFEST.json",
        target_study / "PUBLIC-EVIDENCE-MANIFEST.json",
    )
    shutil.copytree(study / "results", target_study / "results")
    return destination


def rehash_evidence(capsule: Path, name: str) -> None:
    study = capsule / "research/study-v3-revision"
    manifest_path = study / "PUBLIC-EVIDENCE-MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][name] = digest(study / "results" / name)
    manifest_path.write_text(json.dumps(manifest))


def test_source_only_revision_replay(tmp_path: Path) -> None:
    assert replay(copy_public_capsule(tmp_path))["status"] == "REPLAY_PASS"


def test_frozen_catalog_tampering_is_rejected(tmp_path: Path) -> None:
    capsule = copy_public_capsule(tmp_path)
    path = capsule / "research/study-v3-revision/frozen-source/src/living_context/catalog.py"
    path.write_bytes(path.read_bytes() + b"\n# tampered\n")
    with pytest.raises(ValueError, match="frozen source mismatch: src/living_context/catalog.py"):
        replay(capsule)


def test_missing_catalog_pin_cannot_execute_snapshot(tmp_path: Path) -> None:
    capsule = copy_public_capsule(tmp_path)
    protocol_path = capsule / "research/study-v3-revision/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["source_sha256"].pop("src/living_context/catalog.py")
    protocol_path.write_text(json.dumps(protocol))
    path = capsule / "research/study-v3-revision/frozen-source/src/living_context/catalog.py"
    path.write_bytes(path.read_bytes() + b"\nraise RuntimeError('snapshot executed')\n")
    with pytest.raises(ValueError, match="frozen catalog pin missing"):
        replay(capsule)


def test_replay_uses_frozen_catalog_not_current_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_stage_changes(*args: object, **kwargs: object) -> None:
        raise AssertionError("current runtime was used")

    monkeypatch.setattr(VersionedCorpus, "stage_changes", broken_stage_changes)
    assert replay(copy_public_capsule(tmp_path))["status"] == "REPLAY_PASS"


def test_duplicate_worker_rejected_even_with_updated_manifest(tmp_path: Path) -> None:
    capsule = copy_public_capsule(tmp_path)
    path = capsule / "research/study-v3-revision/results/worker-receipts.json"
    receipts = json.loads(path.read_text())
    receipts[1]["doc_id"] = receipts[0]["doc_id"]
    receipts[1]["revision"] = receipts[0]["revision"]
    path.write_text(json.dumps(receipts))
    rehash_evidence(capsule, path.name)
    with pytest.raises(ValueError, match="worker identities"):
        replay(capsule)


def test_missing_config_binding_rejected_even_with_updated_manifest(tmp_path: Path) -> None:
    capsule = copy_public_capsule(tmp_path)
    path = capsule / "research/study-v3-revision/results/admission-final.json"
    admission = json.loads(path.read_text())
    key = next(key for key in admission["artifact_sha256"] if key.endswith("#config"))
    admission["artifact_sha256"].pop(key)
    path.write_text(json.dumps(admission))
    rehash_evidence(capsule, path.name)
    with pytest.raises(ValueError, match="tensor/config inventory"):
        replay(capsule)
