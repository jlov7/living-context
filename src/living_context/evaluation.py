"""Small, model-independent evaluation helpers."""

from __future__ import annotations

import re

__all__ = ["answer_matches"]


def answer_matches(text: str, answer: str) -> bool:
    """Match an answer as a complete alphanumeric span, case-insensitively.

    The historical experiments used substring matching, which counted outputs
    such as ``454545`` as an answer of ``45``. Boundary checks reject that
    degenerate repetition while accepting ordinary punctuation and units.
    """
    if not answer:
        raise ValueError("answer must be non-empty")
    pattern = rf"(?<![A-Za-z0-9]){re.escape(answer)}(?![A-Za-z0-9])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None
