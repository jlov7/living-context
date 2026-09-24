"""Deterministic, model-free revision lifecycle using the public core API."""

from __future__ import annotations

from living_context.catalog import Catalog, DocumentRevision, StagedSnapshot, VersionedCorpus
from living_context.dependencies import DependencyGraph, TrainingEdge
from living_context.refresh import RefreshPlan, plan_refresh


def _revision(
    doc_id: str, revision: str, as_of: str, content: str, edit_kind: str
) -> DocumentRevision:
    return DocumentRevision(doc_id, revision, content, edit_kind, as_of)


def _snapshot(snapshot: StagedSnapshot) -> str:
    return ", ".join(
        f"{doc_id}@{record.revision}" for doc_id, record in sorted(snapshot.revisions.items())
    )


def _rebuild(plan: RefreshPlan) -> str:
    return ", ".join(f"{doc_id}@{revision}" for doc_id, revision in sorted(plan.rebuild))


def run_example() -> tuple[str, ...]:
    """Return an observed lifecycle transcript; no model or artifact bytes are loaded.

    The caller owns catalog mutation, declares dependency edges, and supplies
    validation outcomes to ``admit``. Plans are advisory and are not executed.
    """
    catalog = Catalog()
    initial = {
        record.doc_id: record
        for record in (
            _revision("atlas", "v1", "2026-01-01", "route A", "initial"),
            _revision("beacon", "v1", "2026-01-01", "route B", "initial"),
            _revision("cedar", "v1", "2026-01-01", "route C", "initial"),
        )
    }
    for record in initial.values():
        catalog.add_revision(record)
    corpus = VersionedCorpus(catalog)
    first = corpus.admit(corpus.stage(initial), checks_passed=True)
    assert first is not None
    lines = [f"admitted generation {first.generation}: {_snapshot(first)}"]

    # These edges are authored observations, not inferred by this library.
    graph = DependencyGraph()
    graph.add(TrainingEdge("atlas", "v1", "beacon", "v1", "example-batch"))
    edited = _revision("atlas", "v2", "2026-02-01", "route A2", "number-change")
    catalog.add_revision(edited)
    pending = corpus.stage_changes({"atlas": edited})
    present = dict(first.revisions)
    changed_only = plan_refresh(4, first, pending, present)
    dependency_aware = plan_refresh(5, first, pending, present, graph)
    fallback = plan_refresh(6, first, pending, present)
    lines.extend(
        (
            f"arm 4 rebuild plan: {_rebuild(changed_only)}",
            f"arm 5 rebuild plan: {_rebuild(dependency_aware)}",
            f"arm 6 text fallback plan: {', '.join(sorted(fallback.fallback_to_text))}",
        )
    )

    assert corpus.admit(pending, checks_passed=False) is None
    still_live = corpus.serve()
    assert still_live.generation == first.generation
    lines.append(
        f"rejected checks: generation {still_live.generation} still serves "
        f"atlas@{still_live.revisions['atlas'].revision}"
    )
    second = corpus.admit(pending, checks_passed=True)
    assert second is not None
    lines.append(f"admitted generation {second.generation}: {_snapshot(second)}")

    deleting = corpus.stage_changes({}, deleted=frozenset({"cedar"}))
    retirement = plan_refresh(4, second, deleting, dict(second.revisions))
    third = corpus.admit(deleting, checks_passed=True)
    assert third is not None and "cedar" not in third.revisions
    lines.append(
        f"deletion retirement plan: {', '.join(sorted(retirement.retire))}; "
        f"admitted generation {third.generation}"
    )
    lines.append(f"historical generation {first.generation}: {_snapshot(corpus.serve(1))}")
    as_of = catalog.revision_at("atlas", "2026-02-01")
    assert as_of is not None
    lines.append(f"catalog as of 2026-02-01: atlas@{as_of.revision}")
    return tuple(lines)


def main() -> None:
    print("\n".join(run_example()))


if __name__ == "__main__":
    main()
