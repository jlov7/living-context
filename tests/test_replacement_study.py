from __future__ import annotations

import math
from pathlib import Path

import pytest

from living_context.replacement_study import (
    CacheKey,
    RevisionCacheRegistry,
    StudyValidationError,
    assert_prefix_only_training,
    cross_entropy_from_logits,
    lexical_retrieve,
    load_labels,
    load_questions,
    load_sources,
    parse_labels,
    parse_questions,
    parse_sources,
    read_sealed_json,
    score_exact_json,
    target_prediction_slice,
    validate_data_separation,
    validate_frozen_inputs,
    write_sealed_json,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research" / "study-v2"


def test_study_data_are_disjoint_and_complete() -> None:
    sources = load_sources(DATA / "sources.json")
    questions = load_questions(DATA / "questions.json")
    labels = load_labels(DATA / "labels.json")
    validate_data_separation(sources, questions, labels)
    assert {row["split"] for row in sources.values()} == {"calibration", "development", "final"}
    assert all("answer" not in row for row in questions.values())


@pytest.mark.parametrize(
    ("output", "error"),
    [
        ("not json", "invalid_json"),
        ('{"answer":"DEV-7314-KT"} trailing', "invalid_json"),
        ('{"answer":"DEV-7314-KT","note":"ok"}', "invalid_schema"),
        ('{"answer":"wrong","answer":"DEV-7314-KT"}', "invalid_schema"),
        ('{"answer":["DEV-7314-KT"]}', "invalid_schema"),
        ('{"answer":"DEV-7314-KTDEV-7314-KT"}', "wrong_answer"),
    ],
)
def test_exact_parser_rejects_malformed_or_nonexact_output(output: str, error: str) -> None:
    result = score_exact_json(output, "DEV-7314-KT")
    assert not result.correct
    assert result.error == error


def test_exact_parser_accepts_only_normalized_exact_value() -> None:
    assert score_exact_json('  {"answer":"  Dev-7314-Kt  "}\n', "DEV-7314-KT").correct


def test_revision_cache_clones_and_invalidates_old_revision() -> None:
    registry: RevisionCacheRegistry[list[int]] = RevisionCacheRegistry()
    first = CacheKey("m", "t", "p", "doc", "v1", "c1")
    second = CacheKey("m", "t", "p", "doc", "v2", "c2")
    registry.put(first, [1])
    clone = registry.get(first)
    clone.append(2)
    assert registry.get(first) == [1]
    registry.put(second, [3])
    assert len(registry) == 1
    with pytest.raises(KeyError):
        registry.get(first)
    assert registry.get(second) == [3]


def test_reference_cross_entropy_uses_target_logit_shift() -> None:
    logits = [0.0, 1.0, 2.0]
    expected = math.log(math.exp(0.0) + math.exp(1.0) + math.exp(2.0)) - 2.0
    assert cross_entropy_from_logits(logits, target=2) == pytest.approx(expected)
    assert cross_entropy_from_logits(logits, target=2) < cross_entropy_from_logits(logits, target=0)


def test_cross_entropy_rejects_invalid_targets_and_nonfinite_logits() -> None:
    with pytest.raises(StudyValidationError, match="invalid logits"):
        cross_entropy_from_logits([0.0, 1.0], target=2)
    with pytest.raises(StudyValidationError, match="finite"):
        cross_entropy_from_logits([0.0, float("nan")], target=1)


def test_target_prediction_slice_aligns_causal_logits_after_prefix() -> None:
    # With three prefix positions and five input tokens, targets 2..4 are
    # predicted by combined logits at positions 4..6.
    positions = list(range(8))[target_prediction_slice(3, 5, 2)]
    assert positions == [4, 5, 6]


def test_prefix_only_training_contract() -> None:
    assert_prefix_only_training([], ["weights"])
    with pytest.raises(StudyValidationError, match="base model"):
        assert_prefix_only_training(["layers.0.weight"], ["weights"])
    with pytest.raises(StudyValidationError, match="unexpected"):
        assert_prefix_only_training([], ["weights", "other"])


def test_data_validator_rejects_training_question_reuse() -> None:
    sources = load_sources(DATA / "sources.json")
    questions = load_questions(DATA / "questions.json")
    labels = load_labels(DATA / "labels.json")
    questions["dev-code-a"]["question"] = sources["development-vela-v1"]["training_qa"][0][
        "question"
    ]
    with pytest.raises(StudyValidationError, match="held-out wording"):
        validate_data_separation(sources, questions, labels)


def test_lexical_retrieval_selects_documents_without_audit_source_ids() -> None:
    sources = load_sources(DATA / "sources.json")
    active = [sources["final-orchid-v2"], sources["final-juniper-v1"]]
    selected = lexical_retrieve(
        "List the current Orchid access code and Juniper seal code.", active, limit=2
    )
    assert [row["source_id"] for row in selected] == ["final-juniper-v1", "final-orchid-v2"]


def test_verified_input_snapshot_is_immutable_after_path_changes(tmp_path: Path) -> None:
    data = b'{"value":1}'
    path = tmp_path / "input.json"
    path.write_bytes(data)
    import hashlib

    snapshots = validate_frozen_inputs(
        tmp_path, {"inputs": {"input.json": hashlib.sha256(data).hexdigest()}}
    )
    path.write_bytes(b'{"value":2}')
    assert snapshots["input.json"] == data


def test_sealed_output_must_match_before_parse(tmp_path: Path) -> None:
    path = tmp_path / "raw.json"
    seal = write_sealed_json(path, {"rows": []})
    assert read_sealed_json(path, seal) == {"rows": []}
    path.write_text('{"rows":["changed"]}\n')
    with pytest.raises(StudyValidationError, match="seal mismatch"):
        read_sealed_json(path, seal)
    with pytest.raises(StudyValidationError, match="already exists"):
        write_sealed_json(path, {"rows": []})


def test_source_and_question_ids_must_be_unique() -> None:
    import json

    source_payload = json.loads((DATA / "sources.json").read_text())
    source_payload["sources"].append(source_payload["sources"][0])
    with pytest.raises(StudyValidationError, match="unique"):
        parse_sources(source_payload)

    question_payload = json.loads((DATA / "questions.json").read_text())
    question_payload["questions"].append(question_payload["questions"][0])
    with pytest.raises(StudyValidationError, match="unique"):
        parse_questions(question_payload)


def test_training_source_schema_and_label_schema_reject_partial_input() -> None:
    import json

    source_payload = json.loads((DATA / "sources.json").read_text())
    del source_payload["sources"][0]["training_qa"]
    with pytest.raises(StudyValidationError, match="exactly"):
        parse_sources(source_payload)
    with pytest.raises(StudyValidationError, match="labels must map"):
        parse_labels({"schema_version": 1, "labels": {}})


def test_verified_snapshot_rejects_changed_bytes_and_path_escape(tmp_path: Path) -> None:
    target = tmp_path / "input.json"
    target.write_text("{}")
    with pytest.raises(StudyValidationError, match="mismatch"):
        validate_frozen_inputs(tmp_path, {"inputs": {"input.json": "0" * 64}})
    with pytest.raises(StudyValidationError, match="escapes repository"):
        validate_frozen_inputs(tmp_path, {"inputs": {"../outside.json": "0" * 64}})
