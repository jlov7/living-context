from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from experiments.replay_study_v2 import replay
from living_context.replacement_study import StudyValidationError, write_sealed_json

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "research" / "study-v2" / "results"


def test_evidence_only_replay_closes_the_larger_study_without_tensor_files() -> None:
    assert not list(EVIDENCE.rglob("*.safetensors"))
    result = replay(ROOT, EVIDENCE)
    assert result["calibration"]["passed"]
    assert result["development"]["cached_text_exact"] == 2
    assert result["development"]["prefix_exact"] == 0
    assert result["development"]["prefix_denominator"] == 12
    assert not result["development"]["larger_study_admitted"]


def test_replay_rejects_resealed_raw_response_that_disagrees_with_score(tmp_path: Path) -> None:
    copied = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, copied)
    raw_path = copied / "development-raw.json"
    seal_path = copied / "development-raw.json.seal.json"
    raw = json.loads(raw_path.read_text())
    raw["rows"][0]["output"] = '{"answer":"altered"}'
    raw_path.unlink()
    seal_path.unlink()
    write_sealed_json(raw_path, raw)
    with pytest.raises(StudyValidationError, match="score/raw binding mismatch"):
        replay(ROOT, copied)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_replay_rejects_resealed_scored_row_inventory(tmp_path: Path, mutation: str) -> None:
    copied = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, copied)
    path = copied / "development-scores.json"
    seal = path.with_suffix(path.suffix + ".seal.json")
    score = json.loads(path.read_text())
    if mutation == "missing":
        score["rows"].pop()
    elif mutation == "extra":
        score["rows"].append(score["rows"][0])
    else:
        score["rows"][1] = score["rows"][0]
    path.unlink()
    seal.unlink()
    write_sealed_json(path, score)
    with pytest.raises(StudyValidationError, match="scored output inventory|scored row differs"):
        replay(ROOT, copied)
