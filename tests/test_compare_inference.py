from pathlib import Path

from scripts.compare_inference import compare, revision_rows


def test_comparator_preserves_missing_extra_and_differing_responses() -> None:
    result = compare(
        {"a": {"response": "one", "token_ids": [1]}, "b": {"response": "two", "token_ids": [2]}},
        {
            "a": {"response": "changed", "token_ids": [1]},
            "c": {"response": "three", "token_ids": [3]},
        },
    )
    assert [cell["state"] for cell in result["cells"]] == ["DIFFERENT", "MISSING", "EXTRA"]
    assert result["cells"][0]["reference"]["response"] == "one"
    assert result["cells"][0]["candidate"]["response"] == "changed"


def test_revision_comparator_includes_all_primary_and_stale_cells() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = revision_rows(root / "research/study-v3-revision/results")
    assert len(rows) == 52
    assert sum(key.startswith("initial|") for key in rows) == 24
    assert sum(key.startswith("final|") for key in rows) == 24
    assert sum(key.startswith("stale|") for key in rows) == 4
