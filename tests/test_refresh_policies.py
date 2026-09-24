"""Refresh-policy tests — the arms differ only in which artifacts rebuild."""

from __future__ import annotations

import pytest

from living_context.catalog import (
    Catalog,
    DocumentRevision,
    StagedSnapshot,
    VersionedCorpus,
)
from living_context.dependencies import DependencyGraph, TrainingEdge
from living_context.refresh import plan_refresh


def _rev(doc_id: str, revision: str, as_of: str, content: str = "c") -> DocumentRevision:
    return DocumentRevision(
        doc_id=doc_id, revision=revision, content=content, edit_kind="initial", as_of=as_of
    )


def _setup() -> tuple[VersionedCorpus, dict[str, DocumentRevision], StagedSnapshot]:
    catalog = Catalog()
    catalog.add_revision(_rev("a", "v1", "2026-01-01"))
    catalog.add_revision(_rev("b", "v1", "2026-01-01"))
    catalog.add_revision(_rev("c", "v1", "2026-01-01"))
    corpus = VersionedCorpus(catalog)
    old = {
        d.doc_id: d
        for d in (
            catalog.revisions_of("a")[-1],
            catalog.revisions_of("b")[-1],
            catalog.revisions_of("c")[-1],
        )
    }
    corpus.admit(corpus.stage(old), checks_passed=True)

    catalog.add_revision(_rev("a", "v2", "2026-02-01", content="changed"))
    new = corpus.stage(
        {
            "a": catalog.revisions_of("a")[-1],
            "b": catalog.revisions_of("b")[-1],
            "c": catalog.revisions_of("c")[-1],
        }
    )
    present = {
        "a": catalog.revisions_of("a")[-2],
        "b": catalog.revisions_of("b")[-1],
        "c": catalog.revisions_of("c")[-1],
    }
    return corpus, present, new


def test_arm4_rebuilds_only_changed() -> None:
    corpus, present, new = _setup()
    plan = plan_refresh(4, corpus.serve(), new, present)
    assert plan.rebuild == frozenset({("a", "v2")})
    assert plan.fallback_to_text == frozenset()


def test_arm5_widens_through_co_training() -> None:
    corpus, present, new = _setup()
    graph = DependencyGraph()
    # The graph records the OLD artifact revision that was actually trained.
    graph.add(TrainingEdge("a", "v1", "c", "v1", "batch1"))
    plan = plan_refresh(5, corpus.serve(), new, present, graph)
    assert ("a", "v2") in plan.rebuild
    assert ("c", "v1") in plan.rebuild
    assert ("b", "v1") not in plan.rebuild


def test_arm5_maps_old_dependency_edges_to_new_rebuild_revisions() -> None:
    corpus, present, new = _setup()
    graph = DependencyGraph()
    graph.add(TrainingEdge("a", "v1", "c", "v1", "batch1"))
    plan = plan_refresh(5, corpus.serve(), new, present, graph)
    assert plan.rebuild == frozenset({("a", "v2"), ("c", "v1")})


def test_deleted_artifact_is_retired_and_not_used_as_fallback() -> None:
    corpus, present, _ = _setup()
    deleted = corpus.stage_changes({}, deleted=frozenset({"a"}))
    plan = plan_refresh(6, corpus.serve(), deleted, present)
    assert plan.rebuild == frozenset()
    assert plan.retire == frozenset({"a"})
    assert plan.fallback_to_text == frozenset()


def test_arm6_adds_text_fallback_for_rebuilding_docs() -> None:
    corpus, present, new = _setup()
    plan = plan_refresh(6, corpus.serve(), new, present)
    assert plan.rebuild == frozenset({("a", "v2")})
    assert plan.fallback_to_text == frozenset({"a"})


def test_unknown_arm_rejected() -> None:
    corpus, present, new = _setup()
    try:
        plan_refresh(99, corpus.serve(), new, present)
        raise AssertionError("unknown arm accepted")
    except ValueError:
        pass


def test_present_must_match_old_trained_revisions() -> None:
    corpus, present, new = _setup()
    present["a"] = new.revisions["a"]
    with pytest.raises(ValueError, match="must exactly describe"):
        plan_refresh(4, corpus.serve(), new, present)
