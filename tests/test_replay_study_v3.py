import json
from pathlib import Path

import pytest

from scripts.replay_study_v3 import replay, validate_worker_receipts

ROOT = Path(__file__).resolve().parents[1]


def test_public_v3_evidence_replay() -> None:
    result = replay(ROOT)
    assert result["counts"] == {"adapter": 8, "bare": 0, "text": 8}
    assert result["gate"] == "PILOT_GATE_PASS"


def test_public_v3_evidence_tamper_fails(tmp_path: Path) -> None:
    target = tmp_path / "research/study-v3/results"
    target.mkdir(parents=True)
    source = ROOT / "research/study-v3"
    for path in (source / "results").iterdir():
        (target / path.name).write_bytes(path.read_bytes())
    for path in (
        source / "training.json",
        source / "evaluation.json",
        source / "protocol.json",
        source / "PUBLIC-EVIDENCE-MANIFEST.json",
    ):
        (target.parent / path.name).write_bytes(path.read_bytes())
    # The replay checks source hashes before evidence; link the original
    # source tree via a copy of the small checked files.
    protocol = json.loads((source / "protocol.json").read_text())
    for relative in protocol["source_sha256"]:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    (target / "pilot-raw.json").write_text("{}")
    with pytest.raises(ValueError, match="public evidence hash mismatch"):
        replay(tmp_path)


def test_duplicate_worker_receipt_fails_resource_gate() -> None:
    source = ROOT / "research/study-v3"
    training = json.loads((source / "training.json").read_text())
    receipts = json.loads((source / "results/worker-receipts.json").read_text())
    receipts[-1] = dict(receipts[-2])
    with pytest.raises(ValueError, match="identities"):
        validate_worker_receipts(receipts, training)
