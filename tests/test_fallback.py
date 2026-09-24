"""Stale fallback retrieval is caught — never silently served as current.

Milestone A exit criterion: a stale fallback lookup (old revision pulled in
because the new artifact is not ready) must be observable as stale, so the
consumer can decide to answer from current text instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from living_context.catalog import Catalog, DocumentRevision


@dataclass(frozen=True, slots=True)
class FallbackResult:
    doc_id: str
    revision: str
    content: str
    is_current: bool


def resolve_fallback(catalog: Catalog, doc_id: str, as_of: str) -> FallbackResult | None:
    """Return the newest revision at or before `as_of`, flagged for staleness.

    `is_current` is True only when the revision is the LATEST in the catalog —
    any other revision is a stale fallback and must be treated as such.
    """
    chosen = catalog.revision_at(doc_id, as_of)
    if chosen is None:
        return None
    all_revs = catalog.revisions_of(doc_id)
    is_current = bool(all_revs) and chosen.revision == all_revs[-1].revision
    return FallbackResult(
        doc_id=chosen.doc_id,
        revision=chosen.revision,
        content=chosen.content,
        is_current=is_current,
    )


def test_stale_fallback_is_flagged_not_silently_current() -> None:
    catalog = Catalog()
    catalog.add_revision(
        DocumentRevision("spec/temp", "v1", "limit 120 C", "initial", "2026-01-01")
    )
    catalog.add_revision(
        DocumentRevision("spec/temp", "v2", "limit 200 C", "correction", "2026-02-01")
    )

    stale = resolve_fallback(catalog, "spec/temp", "2026-01-15")
    assert stale is not None
    assert stale.revision == "v1"
    assert stale.content == "limit 120 C"
    assert stale.is_current is False  # stale must be visible

    current = resolve_fallback(catalog, "spec/temp", "2026-03-01")
    assert current is not None
    assert current.revision == "v2"
    assert current.is_current is True


def test_fallback_for_deleted_doc_is_absent() -> None:
    catalog = Catalog()
    assert resolve_fallback(catalog, "never-existed", "2026-05-01") is None
