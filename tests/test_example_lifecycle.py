"""The installed-package example must describe real catalog and planner behavior."""

from __future__ import annotations

from living_context.example_lifecycle import run_example


def test_example_exercises_complete_admission_and_historical_readback() -> None:
    lines = run_example()
    assert "admitted generation 1: atlas@v1, beacon@v1, cedar@v1" in lines
    assert "arm 4 rebuild plan: atlas@v2" in lines
    assert "arm 5 rebuild plan: atlas@v2, beacon@v1" in lines
    assert "arm 6 text fallback plan: atlas" in lines
    assert "rejected checks: generation 1 still serves atlas@v1" in lines
    assert "admitted generation 2: atlas@v2, beacon@v1, cedar@v1" in lines
    assert "deletion retirement plan: cedar; admitted generation 3" in lines
    assert "historical generation 1: atlas@v1, beacon@v1, cedar@v1" in lines
    assert "catalog as of 2026-02-01: atlas@v2" in lines
