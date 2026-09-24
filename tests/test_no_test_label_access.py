"""Validate the declared scheduler-input boundary in the archived protocol.

There is no scheduler implementation in the retained pilot. These tests check
the YAML contract only; they do not claim runtime non-interference or held-out
evaluation. The historical runner trained on each evaluated answer.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "experiments" / "pilot.yaml"
RUNNER = ROOT / "experiments" / "pilot_run.py"


def _scheduler_inputs() -> tuple[list[str], list[dict]]:
    """The scheduler's allowed inputs and the question list, from the frozen pilot."""
    data = yaml.safe_load(PILOT.read_text())
    return list(data["design"]["scheduler_inputs"]), list(data["questions"])


def test_scheduler_inputs_contain_no_question_ids() -> None:
    """Declared scheduler inputs do not name a question or answer field."""
    scheduler_inputs, questions = _scheduler_inputs()
    qids: set[str] = set()
    for q in questions:
        qid = q.get("id")
        if isinstance(qid, str):
            qids.add(qid)
    for source in scheduler_inputs:
        assert "question" not in source.lower(), (
            f"scheduler input {source!r} leaks the presence of questions"
        )
        assert "answer" not in source.lower()
        assert source not in qids


def test_pilot_yaml_is_frozen() -> None:
    """Status must say FROZEN so deviations are reported, not silently fixed."""
    data = yaml.safe_load(PILOT.read_text())
    assert data["protocol"]["status"] == "FROZEN"
    assert str(data["protocol"]["frozen_at"]) == "2026-09-21"


def test_declared_repeats_and_forbidden_label_access_recorded() -> None:
    data = yaml.safe_load(PILOT.read_text())
    assert isinstance(data["design"]["repeats"], int) and data["design"]["repeats"] >= 3
    assert data["design"]["test_label_access"] == "forbidden"


def test_historical_runner_discloses_trained_question_design_and_is_blocked() -> None:
    source = RUNNER.read_text()
    assert "trained-question demonstration" in source
    assert "pilot v1 is an immutable historical trained-question run" in source
