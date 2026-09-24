"""Dependency invalidation — deleted IDs, overlapping updates, wrong as-of.

Milestone A exit criterion: wrong as-of versions, overlapping updates, and
deleted IDs are all caught by test, not by discipline.
"""

from __future__ import annotations

from living_context.catalog import Catalog, DocumentRevision, VersionedCorpus
from living_context.dependencies import DependencyGraph, TrainingEdge


def _rev(doc_id: str, revision: str, as_of: str, content: str = "c") -> DocumentRevision:
    return DocumentRevision(
        doc_id=doc_id, revision=revision, content=content, edit_kind="initial", as_of=as_of
    )


def test_deleted_id_is_absent_from_current_but_present_in_history() -> None:
    catalog = Catalog()
    catalog.add_revision(_rev("route/r7", "v1", "2026-01-01"))
    catalog.add_revision(_rev("route/r7", "v2", "2026-02-01", content="retired"))
    corpus = VersionedCorpus(catalog)

    v2 = {d.doc_id: d for d in (catalog.revisions_of("route/r7")[-1],)}
    corpus.admit(corpus.stage(v2), checks_passed=True)

    # Deletion: the doc is dropped from the current corpus but stays in history.
    deletion = corpus.stage({}, deleted=frozenset({"route/r7"}))
    corpus.admit(deletion, checks_passed=True)

    live = corpus.serve()
    assert "route/r7" not in live.revisions
    assert "route/r7" in live.deleted
    delete_target = catalog.revision_at("route/r7", "2026-02-15")
    assert delete_target is not None and delete_target.revision == "v2"


def test_overlapping_updates_loser_is_rejected() -> None:
    """Two overlapping refreshes on the same doc: one wins, the other is stale."""
    catalog = Catalog()
    catalog.add_revision(_rev("a", "v1", "2026-01-01"))
    corpus = VersionedCorpus(catalog)
    corpus.admit(corpus.stage({"a": catalog.revisions_of("a")[-1]}), checks_passed=True)

    catalog.add_revision(_rev("a", "v2", "2026-02-01", content="A"))
    catalog.add_revision(_rev("a", "v3", "2026-02-02", content="B"))

    w2 = corpus.stage_changes({"a": catalog.revisions_of("a")[-2]}, based_on_generation=1)
    w3 = corpus.stage_changes({"a": catalog.revisions_of("a")[-1]}, based_on_generation=1)
    corpus.admit(w2, checks_passed=True)  # v2 wins; generation becomes 2

    try:
        corpus.admit(w3, checks_passed=True)  # v3 staged against gen 1 = stale
        raise AssertionError("overlapping update was not rejected")
    except RuntimeError:
        pass
    assert corpus.current_generation == 2
    assert corpus.is_visible("a", "v2")


def test_wrong_as_of_revision_is_flagged() -> None:
    """Asking for the CURRENT revision when one meant the OLD one writes nothing —
    but the servable-at-generation distinction must be explicit."""
    catalog = Catalog()
    catalog.add_revision(_rev("a", "v1", "2026-01-01", content="old"))
    catalog.add_revision(_rev("a", "v2", "2026-02-01", content="new"))
    corpus = VersionedCorpus(catalog)
    corpus.admit(corpus.stage({"a": catalog.revisions_of("a")[-1]}), checks_passed=True)

    assert corpus.is_visible("a", "v2")  # current
    assert not corpus.is_visible("a", "v1")  # wrong as-of for current


def test_dependency_graph_invalidation_is_transitive() -> None:
    graph = DependencyGraph()
    # batch b1 co-trains (a@v1, x@v1, z@v1); batch b2 co-trains (x@v1, y@v1).
    graph.add(TrainingEdge("a", "v1", "x", "v1", "b1"))
    graph.add(TrainingEdge("a", "v1", "z", "v1", "b1"))
    graph.add(TrainingEdge("x", "v1", "y", "v1", "b2"))

    affected = graph.invalidated_by({("a", "v1")})
    assert ("x", "v1") in affected
    assert ("z", "v1") in affected
    assert ("y", "v1") in affected  # transitive through x
    assert len(affected) == 4  # the changed triplet + three co-trained
