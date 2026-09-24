"""A change set cannot both replace and delete the same document."""

import pytest

from living_context.catalog import (
    Catalog,
    DocumentRevision,
    SnapshotValidationError,
    VersionedCorpus,
)


def make_corpus() -> tuple[Catalog, VersionedCorpus, DocumentRevision, DocumentRevision]:
    catalog = Catalog()
    first = DocumentRevision("atlas", "v1", "old", "initial", "2026-09-01")
    second = DocumentRevision("atlas", "v2", "new", "correction", "2026-09-02")
    catalog.add_revision(first)
    catalog.add_revision(second)
    corpus = VersionedCorpus(catalog)
    corpus.admit(corpus.stage({"atlas": first}), checks_passed=True)
    return catalog, corpus, first, second


@pytest.mark.parametrize("already_live", [True, False])
def test_overlapping_update_and_deletion_is_rejected(already_live: bool) -> None:
    catalog, corpus, _, second = make_corpus()
    if not already_live:
        second = DocumentRevision("new-doc", "v1", "new", "initial", "2026-09-02")
        catalog.add_revision(second)
    before = corpus.serve()
    with pytest.raises(SnapshotValidationError, match="overlap"):
        corpus.stage_changes({second.doc_id: second}, deleted=frozenset({second.doc_id}))
    assert corpus.serve() is before
    assert corpus.current_generation == before.generation


def test_disjoint_update_and_delete_can_still_be_admitted() -> None:
    catalog, corpus, _, second = make_corpus()
    other = DocumentRevision("beacon", "v1", "other", "initial", "2026-09-01")
    catalog.add_revision(other)
    corpus.admit(corpus.stage_changes({"beacon": other}), checks_passed=True)
    candidate = corpus.stage_changes({"atlas": second}, deleted=frozenset({"beacon"}))
    assert corpus.is_visible("atlas", "v1")
    result = corpus.admit(candidate, checks_passed=True)
    assert result is not None
    assert result.revisions == {"atlas": second}
    assert result.deleted == frozenset({"beacon"})


def test_explicit_reintroduction_without_deletion_is_still_allowed() -> None:
    _, corpus, _, second = make_corpus()
    corpus.admit(corpus.stage_changes({}, deleted=frozenset({"atlas"})), checks_passed=True)
    result = corpus.admit(corpus.stage_changes({"atlas": second}), checks_passed=True)
    assert result is not None
    assert result.revisions == {"atlas": second}
    assert "atlas" not in result.deleted
