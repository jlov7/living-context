from __future__ import annotations

import pytest

from living_context.evaluation import answer_matches


@pytest.mark.parametrize(
    ("text", "answer"),
    [
        ("45 L", "45"),
        ("The required alloy is AL-7075.", "al-7075"),
        ("Status: REVOKED", "revoked"),
    ],
)
def test_answer_matches_complete_span(text: str, answer: str) -> None:
    assert answer_matches(text, answer)


@pytest.mark.parametrize("text", ["454545", "of200200", "AL-70750"])
def test_answer_match_rejects_substring_and_degenerate_repetition(text: str) -> None:
    answer = "AL-7075" if text.startswith("AL") else ("200" if "200" in text else "45")
    assert not answer_matches(text, answer)


def test_empty_answer_is_invalid() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        answer_matches("anything", "")
