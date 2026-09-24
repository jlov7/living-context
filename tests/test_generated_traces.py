"""Deterministic revision traces checked against an independent event-log rebuild.

Grammar: ADD(doc), EDIT(doc), DELETE(doc), RESTORE(doc), STAGE, REJECT or ADMIT.
Each seed produces 48 transitions. The oracle replays accepted events from an
empty map for every read; it does not call the catalog's merge or validation.
"""

from collections.abc import Iterable
from datetime import date, timedelta
from random import Random

import pytest

from living_context.catalog import Catalog, DocumentRevision, VersionedCorpus
from living_context.dependencies import DependencyGraph, TrainingEdge
from living_context.refresh import plan_refresh


def rebuild(events: Iterable[tuple[str, DocumentRevision | str]]) -> dict[str, DocumentRevision]:
    live: dict[str, DocumentRevision] = {}
    for operation, value in events:
        if operation == "put":
            assert isinstance(value, DocumentRevision)
            live[value.doc_id] = value
        else:
            assert isinstance(value, str)
            live.pop(value, None)
    return live


class AlwaysAdmitMutant(VersionedCorpus):
    def admit(self, staged, *, checks_passed, receipt=None):
        return super().admit(staged, checks_passed=True, receipt=receipt)


def run_trace(seed: int, corpus_type=VersionedCorpus, steps: int = 48) -> None:
    rng = Random(seed)
    catalog = Catalog()
    corpus = corpus_type(catalog)
    events: list[tuple[str, DocumentRevision | str]] = []
    counters = {doc: 0 for doc in "abcd"}
    for step in range(steps):
        doc = rng.choice(tuple(counters))
        live = rebuild(events)
        reject = step % 7 == 0
        if doc in live and rng.random() < 0.28:
            operation = ("delete", doc)
            staged = corpus.stage_changes({}, deleted=frozenset({doc}))
        else:
            counters[doc] += 1
            number = counters[doc]
            revision = DocumentRevision(
                doc,
                f"r{number}",
                f"{seed}:{doc}:{number}",
                "initial" if doc not in live else "correction",
                (date(2026, 1, 1) + timedelta(days=number)).isoformat(),
            )
            catalog.add_revision(revision)
            operation = ("put", revision)
            staged = corpus.stage_changes({doc: revision})
        before = corpus.serve()
        observed = corpus.admit(staged, checks_passed=not reject)
        if reject:
            assert observed is None, f"seed={seed} step={step}: rejected candidate admitted"
            assert corpus.serve() is before
        else:
            events.append(operation)
            assert observed is not None
            assert dict(observed.revisions) == rebuild(events)
            assert set(observed.deleted).isdisjoint(observed.revisions)
        assert dict(corpus.serve().revisions) == rebuild(events)


def test_generated_revision_traces_and_minimal_rejection_witness() -> None:
    for seed in (3, 17, 29, 101):
        run_trace(seed)
    # A one-step rejected initial revision is a manually minimized mutant witness.
    with pytest.raises(AssertionError, match="seed=3 step=0"):
        run_trace(3, AlwaysAdmitMutant, steps=1)


def test_dependency_plan_against_separately_written_graph_walk_and_fault_control() -> None:
    catalog = Catalog()
    corpus = VersionedCorpus(catalog)
    old_revisions: dict[str, DocumentRevision] = {}
    for doc in "abcde":
        revision = DocumentRevision(doc, "r1", doc, "initial", "2026-01-01")
        catalog.add_revision(revision)
        old_revisions[doc] = revision
    old = corpus.admit(corpus.stage(old_revisions), checks_passed=True)
    assert old is not None
    changed = DocumentRevision("a", "r2", "new a", "correction", "2026-01-02")
    catalog.add_revision(changed)
    staged = corpus.stage_changes({"a": changed})
    graph = DependencyGraph()
    pairs = (("a", "b"), ("b", "c"), ("d", "e"))
    for left, right in pairs:
        graph.add(TrainingEdge(left, "r1", right, "r1", "batch"))
    actual = plan_refresh(5, old, staged, old_revisions, graph)
    # Oracle uses plain undirected adjacency and repeated whole-edge scans.
    impacted = {"a"}
    while True:
        expanded = (
            impacted
            | {right for left, right in pairs if left in impacted}
            | {left for left, right in pairs if right in impacted}
        )
        if expanded == impacted:
            break
        impacted = expanded
    expected = frozenset((doc, staged.revisions[doc].revision) for doc in impacted)
    assert actual.rebuild == expected == frozenset({("a", "r2"), ("b", "r1"), ("c", "r1")})

    def changed_only_mutant():
        return plan_refresh(4, old, staged, old_revisions, graph)

    with pytest.raises(AssertionError):
        assert changed_only_mutant().rebuild == expected
