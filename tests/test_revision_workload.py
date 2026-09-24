from datetime import date, timedelta

from experiments.revision_workload import N, closure, edges, run
from living_context.catalog import Catalog, DocumentRevision, VersionedCorpus
from living_context.dependencies import DependencyGraph, TrainingEdge
from living_context.refresh import plan_refresh


def test_analytical_closure_matches_reference_planner() -> None:
    for shape in ("disconnected", "chain", "star", "four-cliques", "full-clique"):
        catalog = Catalog()
        corpus = VersionedCorpus(catalog)
        initial = {}
        for index in range(N):
            doc = str(index)
            revision = DocumentRevision(doc, "r1", doc, "initial", "2026-01-01")
            catalog.add_revision(revision)
            initial[doc] = revision
        old = corpus.admit(corpus.stage(initial), checks_passed=True)
        assert old is not None
        updates = {}
        seeds = {0, 17, 65, 121}
        for index in seeds:
            doc = str(index)
            revision = DocumentRevision(
                doc,
                "r2",
                f"new-{doc}",
                "correction",
                (date(2026, 1, 1) + timedelta(days=1)).isoformat(),
            )
            catalog.add_revision(revision)
            updates[doc] = revision
        staged = corpus.stage_changes(updates)
        graph = DependencyGraph()
        for left, right in edges(shape):
            graph.add(TrainingEdge(str(left), "r1", str(right), "r1", "synthetic"))
        planned = plan_refresh(5, old, staged, initial, graph)
        assert {int(doc) for doc, _ in planned.rebuild} == closure(seeds, edges(shape))


def test_frozen_workload_matrix_size_and_counterexamples() -> None:
    rows = run()["rows"]
    assert isinstance(rows, list) and len(rows) == 45
    one_edit = [row for row in rows if len(row["changed_doc_ids"]) == 1]
    assert all(
        row["closure_rebuild_units"] == 128
        for row in one_edit
        if row["shape"] in {"chain", "star", "full-clique"}
    )
