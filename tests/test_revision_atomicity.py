"""Atomic admission — a half-updated corpus is NEVER observable.

Milestone A exit criterion: a snapshot becomes active only after checks finish.
If admission fails, the corpus must still serve the OLD complete snapshot.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from living_context.catalog import (
    Catalog,
    DocumentRevision,
    SnapshotValidationError,
    StagedSnapshot,
    StaleBaseError,
    VersionedCorpus,
)


def _rev(
    doc_id: str,
    revision: str,
    as_of: str,
    content: str = "c",
    edit_kind: str = "initial",
) -> DocumentRevision:
    return DocumentRevision(
        doc_id=doc_id, revision=revision, content=content, edit_kind=edit_kind, as_of=as_of
    )


def _loaded_corpus() -> tuple[Catalog, VersionedCorpus]:
    catalog = Catalog()
    catalog.add_revision(_rev("a", "v1", "2026-01-01"))
    catalog.add_revision(_rev("b", "v1", "2026-01-01"))
    corpus = VersionedCorpus(catalog)
    base = corpus.stage({"a": catalog.revisions_of("a")[-1], "b": catalog.revisions_of("b")[-1]})
    corpus.admit(base, checks_passed=True)
    return catalog, corpus


def test_rejected_check_leaves_the_old_corpus_serving() -> None:
    catalog, corpus = _loaded_corpus()
    catalog.add_revision(_rev("a", "v2", "2026-02-01", content="UPDATED"))
    catalog.add_revision(_rev("b", "v2", "2026-02-01", content="UPDATED"))
    staged = corpus.stage({"a": catalog.revisions_of("a")[-1], "b": catalog.revisions_of("b")[-1]})
    rejected = corpus.admit(staged, checks_passed=False)

    assert rejected is None
    live = corpus.serve()
    assert live.generation == 1
    assert live.revisions["a"].content == "c"
    assert live.revisions["b"].content == "c"
    assert not corpus.is_visible("a", "v2")


@pytest.mark.parametrize("untrusted_value", ["false", 1, None])
def test_admission_requires_an_actual_boolean_check_result(untrusted_value: object) -> None:
    catalog, corpus = _loaded_corpus()
    catalog.add_revision(_rev("a", "v2", "2026-02-01"))
    staged = corpus.stage_changes({"a": catalog.revisions_of("a")[-1]})
    with pytest.raises(TypeError, match="checks_passed must be a bool"):
        corpus.admit(staged, checks_passed=untrusted_value)  # type: ignore[arg-type]
    assert corpus.serve().generation == 1


def test_half_staged_revision_is_never_observable_as_current() -> None:
    """A partial mapping cannot be mistaken for a complete snapshot."""
    catalog, corpus = _loaded_corpus()
    catalog.add_revision(_rev("a", "v2", "2026-02-01", content="NEW-A"))
    with pytest.raises(SnapshotValidationError, match="omits live documents"):
        corpus.stage({"a": catalog.revisions_of("a")[-1]})
    live = corpus.serve()
    assert live.revisions["a"].revision == "v1"
    assert live.revisions["b"].revision == "v1"


def test_stale_base_is_rejected_when_generation_moved() -> None:
    """A snapshot staged against a no-longer-current generation must not admit."""
    catalog, corpus = _loaded_corpus()  # generation 1
    catalog.add_revision(_rev("a", "v2", "2026-02-01"))

    w1 = corpus.stage_changes({"a": catalog.revisions_of("a")[-1]}, based_on_generation=1)
    corpus.admit(w1, checks_passed=True)  # generation 2

    stale = corpus.stage_changes({"a": catalog.revisions_of("a")[-1]}, based_on_generation=1)
    try:
        corpus.admit(stale, checks_passed=True)
        raise AssertionError("stale base was not rejected")
    except StaleBaseError:
        pass
    assert corpus.current_generation == 2
    assert corpus.is_visible("a", "v2")


def test_as_of_requests_honour_historical_snapshots_separately() -> None:
    catalog = Catalog()
    catalog.add_revision(_rev("a", "v1", "2026-01-01", content="one"))
    catalog.add_revision(_rev("a", "v2", "2026-02-01", content="two"))
    catalog.add_revision(_rev("a", "v3", "2026-03-01", content="three"))

    v1 = catalog.revision_at("a", "2026-01-15")
    v2 = catalog.revision_at("a", "2026-02-15")
    v3 = catalog.revision_at("a", "2026-04-01")
    assert v1 is not None and v1.revision == "v1"
    assert v2 is not None and v2.revision == "v2"
    assert v3 is not None and v3.revision == "v3"
    assert catalog.revision_at("missing", "2026-04-01") is None


def test_staged_and_admitted_snapshots_cannot_be_mutated() -> None:
    catalog, corpus = _loaded_corpus()
    staged = corpus.stage_changes({})
    with pytest.raises(TypeError):
        staged.revisions["a"] = catalog.revisions_of("a")[-1]  # type: ignore[index]
    admitted = corpus.admit(staged, checks_passed=True)
    assert admitted is not None
    with pytest.raises(TypeError):
        admitted.revisions["a"] = catalog.revisions_of("a")[-1]  # type: ignore[index]


def test_admit_revalidates_a_manually_constructed_snapshot() -> None:
    catalog, corpus = _loaded_corpus()
    forged = StagedSnapshot(
        generation=2,
        based_on_generation=1,
        revisions={"a": catalog.revisions_of("a")[-1]},
        deleted=frozenset(),
    )
    with pytest.raises(SnapshotValidationError, match="omits live documents"):
        corpus.admit(forged, checks_passed=True)


def test_explicit_generation_serves_retained_admitted_snapshot() -> None:
    catalog, corpus = _loaded_corpus()
    catalog.add_revision(_rev("a", "v2", "2026-02-01", content="new"))
    corpus.admit(corpus.stage_changes({"a": catalog.revisions_of("a")[-1]}), checks_passed=True)
    assert corpus.serve(1).revisions["a"].revision == "v1"
    assert corpus.serve().revisions["a"].revision == "v2"


def test_revision_ids_are_unique_but_not_lexicographically_ordered() -> None:
    catalog = Catalog()
    catalog.add_revision(_rev("a", "v9", "2026-01-01"))
    catalog.add_revision(_rev("a", "v10", "2026-01-02"))
    with pytest.raises(ValueError, match="already exists"):
        catalog.add_revision(_rev("a", "v10", "2026-01-03"))
    with pytest.raises(ValueError, match="must follow"):
        catalog.add_revision(_rev("a", "release", "2026-01-01"))


def test_dates_are_canonical_and_current_snapshot_cannot_roll_back() -> None:
    catalog, corpus = _loaded_corpus()
    with pytest.raises(ValueError, match="canonical"):
        catalog.add_revision(_rev("a", "compact", "20260102"))
    catalog.add_revision(_rev("a", "v2", "2026-02-01"))
    corpus.admit(corpus.stage_changes({"a": catalog.revisions_of("a")[-1]}), checks_passed=True)
    with pytest.raises(SnapshotValidationError, match="rolls a back"):
        corpus.stage_changes({"a": catalog.revisions_of("a")[0]})


def test_deleted_document_can_be_restored_by_explicit_update() -> None:
    catalog, corpus = _loaded_corpus()
    catalog.add_revision(_rev("a", "v2", "2026-02-01"))
    corpus.admit(corpus.stage_changes({"a": catalog.revisions_of("a")[-1]}), checks_passed=True)
    corpus.admit(corpus.stage_changes({}, deleted=frozenset({"a"})), checks_passed=True)
    with pytest.raises(SnapshotValidationError, match="rolls a back"):
        corpus.stage_changes({"a": catalog.revisions_of("a")[0]})
    restored = corpus.stage_changes({"a": catalog.revisions_of("a")[-1]})
    admitted = corpus.admit(restored, checks_passed=True)
    assert admitted is not None
    assert "a" in admitted.revisions
    assert "a" not in admitted.deleted


def test_concurrent_readers_see_complete_old_or_new_snapshot() -> None:
    catalog, corpus = _loaded_corpus()
    catalog.add_revision(_rev("a", "v2", "2026-02-01"))
    catalog.add_revision(_rev("b", "v2", "2026-02-01"))
    pending = corpus.stage_changes(
        {doc_id: catalog.revisions_of(doc_id)[-1] for doc_id in ("a", "b")}
    )
    start = Barrier(5)
    finish = Barrier(5)

    def read_many() -> list[tuple[int, str, str]]:
        old = corpus.serve()
        observations = [(old.generation, old.revisions["a"].revision, old.revisions["b"].revision)]
        start.wait(timeout=5)
        for _ in range(200):
            snapshot = corpus.serve()
            observations.append(
                (
                    snapshot.generation,
                    snapshot.revisions["a"].revision,
                    snapshot.revisions["b"].revision,
                )
            )
        finish.wait(timeout=5)
        new = corpus.serve()
        observations.append(
            (new.generation, new.revisions["a"].revision, new.revisions["b"].revision)
        )
        return observations

    with ThreadPoolExecutor(max_workers=4) as pool:
        readers = [pool.submit(read_many) for _ in range(4)]
        start.wait(timeout=5)
        admitted = corpus.admit(pending, checks_passed=True)
        assert admitted is not None
        finish.wait(timeout=5)
        all_observations = [row for future in readers for row in future.result()]

    assert set(all_observations) == {(1, "v1", "v1"), (2, "v2", "v2")}
    assert corpus.serve(1).revisions["a"].revision == "v1"


def test_competing_admissions_have_one_winner_and_one_stale_base() -> None:
    catalog, corpus = _loaded_corpus()
    catalog.add_revision(_rev("a", "v2", "2026-02-01"))
    catalog.add_revision(_rev("b", "v2", "2026-02-01"))
    candidates = [
        corpus.stage_changes({doc_id: catalog.revisions_of(doc_id)[-1]}) for doc_id in ("a", "b")
    ]
    start = Barrier(3)

    def admit(candidate: StagedSnapshot) -> str:
        start.wait(timeout=5)
        try:
            accepted = corpus.admit(candidate, checks_passed=True)
            assert accepted is not None
            return "accepted"
        except StaleBaseError:
            return "stale"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = [pool.submit(admit, candidate) for candidate in candidates]
        start.wait(timeout=5)
        assert sorted(future.result() for future in outcomes) == ["accepted", "stale"]
    live = corpus.serve()
    assert live.generation == 2
    assert {key: rev.revision for key, rev in live.revisions.items()} in (
        {"a": "v2", "b": "v1"},
        {"a": "v1", "b": "v2"},
    )
