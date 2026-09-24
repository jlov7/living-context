from threading import Event, Thread

import pytest

from living_context.catalog import Catalog, DocumentRevision, VersionedCorpus
from living_context.refresh import RefreshPlan
from living_context.text_reader import TextReader
from living_context.text_reader_example import walkthrough


def test_caller_reader_generation_revision_and_deletion() -> None:
    catalog = Catalog()
    old = DocumentRevision("a", "r1", "old", "initial", "2026-01-01")
    new = DocumentRevision("a", "r2", "new", "correction", "2026-01-02")
    for revision in (old, new):
        catalog.add_revision(revision)
    corpus = VersionedCorpus(catalog)
    reader = TextReader(corpus)
    corpus.admit(corpus.stage({"a": old}), checks_passed=True)
    pending = corpus.stage_changes({"a": new})
    plan = RefreshPlan(6, frozenset({("a", "r2")}), fallback_to_text=frozenset({"a"}))
    first_read = reader.read("a")
    assert first_read is not None and first_read.revision == "r1"
    with pytest.raises(ValueError, match="not admitted"):
        reader.fallback("a", plan, generation=1)
    corpus.admit(pending, checks_passed=True)
    current_fallback = reader.fallback("a", plan, generation=2)
    historic_read = reader.read("a", generation=1)
    current_read = reader.read("a")
    assert current_fallback is not None and current_fallback.revision == "r2"
    assert historic_read is not None and historic_read.current is False
    assert current_read is not None and current_read.content == "new"
    wrong_plan = RefreshPlan(6, frozenset({("a", "r1")}), fallback_to_text=frozenset({"a"}))
    with pytest.raises(ValueError, match="not admitted"):
        reader.fallback("a", wrong_plan, generation=2)
    try:
        reader.fallback("a", plan, generation=1)
    except ValueError as exc:
        assert "stale" in str(exc)
    else:
        raise AssertionError("stale generation was presented as current")
    corpus.admit(corpus.stage_changes({}, deleted=frozenset({"a"})), checks_passed=True)
    assert reader.read("a") is None
    old_read = reader.read("a", generation=2)
    assert old_read is not None and old_read.current is False


def test_cli_caller_walkthrough_names_revision_and_retirement() -> None:
    assert walkthrough() == [
        "before admission: generation=1 revision=r1 text=Limit: 10",
        "after admission: generation=2 revision=r2 text=Limit: 20",
        "after deletion: current=None",
    ]


def test_fallback_rejects_generation_replaced_during_read() -> None:
    entered, resume = Event(), Event()

    class PausingCorpus(VersionedCorpus):
        pause_next = False

        @property
        def current_generation(self) -> int:
            captured = super().current_generation
            if self.pause_next:
                self.pause_next = False
                entered.set()
                if not resume.wait(5):
                    raise TimeoutError("test admission did not resume")
            return captured

    catalog = Catalog()
    revisions = [
        DocumentRevision("a", f"r{number}", f"value-{number}", "correction", f"2026-01-0{number}")
        for number in (1, 2, 3)
    ]
    for revision in revisions:
        catalog.add_revision(revision)
    corpus = PausingCorpus(catalog)
    for revision in revisions[:2]:
        corpus.admit(corpus.stage_changes({"a": revision}), checks_passed=True)
    plan = RefreshPlan(6, frozenset({("a", "r2")}), fallback_to_text=frozenset({"a"}))
    errors: list[Exception] = []

    def read_fallback() -> None:
        try:
            TextReader(corpus).fallback("a", plan, generation=2)
        except ValueError as exc:
            errors.append(exc)

    corpus.pause_next = True
    thread = Thread(target=read_fallback)
    thread.start()
    assert entered.wait(5)
    try:
        corpus.admit(corpus.stage_changes({"a": revisions[2]}), checks_passed=True)
    finally:
        resume.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], ValueError)
    assert "stale" in str(errors[0])
