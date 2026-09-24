"""The distributed v2 text evidence must reproduce its entire recorded result."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scripts.verify_study_v2 import ReleaseReplayError, verify_frozen_sources, verify_release_replay

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "research/study-v2/results"
MANIFEST = ROOT / "research/study-v2/PUBLIC-EVIDENCE-MANIFEST.json"
EXPECTED = ROOT / "research/study-v2/RESULTS.json"
FROZEN = ROOT / "research/study-v2/frozen-source"


def test_replay_matches_the_full_recorded_result() -> None:
    result = verify_release_replay(ROOT, EVIDENCE, MANIFEST, EXPECTED)
    assert result["development"]["cached_text_exact"] == 2
    assert result["development"]["prefix_exact"] == 0
    assert result["development"]["prefix_denominator"] == 12
    assert result["development"]["larger_study_admitted"] is False


def test_frozen_study_source_matches_the_protocol() -> None:
    assert verify_frozen_sources(ROOT, FROZEN) == 9


def test_changed_frozen_study_source_is_rejected(tmp_path: Path) -> None:
    copied = tmp_path / "frozen-source"
    shutil.copytree(FROZEN, copied)
    (copied / "pyproject.toml").write_text("changed\n")
    with pytest.raises(ReleaseReplayError, match="frozen study source hash mismatch"):
        verify_frozen_sources(ROOT, copied)


@pytest.mark.parametrize("mutation", ["missing", "extra", "tampered"])
def test_evidence_inventory_fails_closed(tmp_path: Path, mutation: str) -> None:
    copied = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, copied)
    target = copied / "_answer-worker-stdout.log"
    if mutation == "missing":
        target.unlink()
    elif mutation == "extra":
        (copied / "unexpected.json").write_text("{}\n")
    else:
        target.write_text("changed\n")
    with pytest.raises(ReleaseReplayError):
        verify_release_replay(ROOT, copied, MANIFEST, EXPECTED)


def test_duplicate_scored_row_fails_even_when_resealed(tmp_path: Path) -> None:
    from living_context.replacement_study import write_sealed_json

    copied = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, copied)
    path = copied / "development-scores.json"
    data = json.loads(path.read_text())
    data["rows"][1] = data["rows"][0]
    path.unlink()
    path.with_suffix(path.suffix + ".seal.json").unlink()
    write_sealed_json(path, data)
    with pytest.raises(ReleaseReplayError):
        verify_release_replay(ROOT, copied, MANIFEST, EXPECTED)
