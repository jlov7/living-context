"""Refresh planning — which learned artifacts must be rebuilt after an edit.

Protocol §4 arms differ ONLY in the refresh policy:
- arm 4: changed artifact only (document boundary as update boundary)
- arm 5: conservative co-training impact set (dependency-aware)
- arm 6: arm 4 + explicit TEXT FALLBACK until the new artifact is admitted

This module computes plans; it never touches a model. The atomic admission
guarantee comes from `VersionedCorpus` (Milestone A) — a plan is executed by
staging the NEW source snapshot and admitting it only after the planned
artifacts pass their checks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from living_context.catalog import DocumentRevision, StagedSnapshot
from living_context.dependencies import DependencyGraph

__all__ = ["ARM_LABELS", "RefreshPlan", "plan_refresh"]


@dataclass(frozen=True, slots=True)
class RefreshPlan:
    """Which documents need a rebuilt artifact, and whether to fall back to text."""

    arm: int
    rebuild: frozenset[tuple[str, str]]  # (doc_id, new_revision) to retrain
    retire: frozenset[str] = frozenset()  # deleted doc ids whose artifacts must be removed
    fallback_to_text: frozenset[str] = frozenset()  # doc_ids answered from fresh text
    reason: str = ""


ARM_LABELS = {
    4: "selective refresh: changed artifact only",
    5: "selective refresh: conservative co-training impact set",
    6: "selective refresh + explicit text fallback during refresh",
}


def plan_refresh(
    arm: int,
    old: StagedSnapshot,
    new: StagedSnapshot,
    present: Mapping[str, DocumentRevision],
    graph: DependencyGraph | None = None,
) -> RefreshPlan:
    """Decide the rebuild set for one source edit (old snapshot -> new snapshot).

    `present` maps every doc_id to its CURRENT pre-edit revision (what old
    artifacts were trained on). Only docs whose revision CHANGED are the seed;
    arm 5 widens the seed through the co-training graph's invalidated_by().
    """
    if arm not in ARM_LABELS:
        raise ValueError(f"unknown arm {arm}")
    if new.based_on_generation != old.generation:
        raise ValueError(
            f"new snapshot is based on generation {new.based_on_generation}, "
            f"not old generation {old.generation}"
        )
    if set(present) != set(old.revisions) or any(
        present[doc_id] != revision for doc_id, revision in old.revisions.items()
    ):
        raise ValueError("present must exactly describe the old snapshot's trained revisions")

    changed_docs = {
        doc_id
        for doc_id in set(old.revisions) | set(new.revisions)
        if old.revisions.get(doc_id) != new.revisions.get(doc_id)
    }
    changed_live = {
        (doc_id, new.revisions[doc_id].revision)
        for doc_id in changed_docs
        if doc_id in new.revisions
    }
    retired = frozenset(doc_id for doc_id in changed_docs if doc_id not in new.revisions)

    if arm == 4:
        rebuild = changed_live
        reason = f"changed-artifact-only refresh of {len(changed_live)} doc(s)"
    elif arm == 5:
        if graph is None:
            raise ValueError("arm 5 requires a dependency graph")
        old_seeds = {
            (doc_id, old.revisions[doc_id].revision)
            for doc_id in changed_docs
            if doc_id in old.revisions
        }
        impacted_old = graph.invalidated_by(old_seeds)
        impacted_docs = changed_docs | {doc_id for doc_id, _ in impacted_old}
        rebuild = {
            (doc_id, new.revisions[doc_id].revision)
            for doc_id in impacted_docs
            if doc_id in new.revisions and (doc_id in present or doc_id in changed_docs)
        }
        reason = (
            f"conservative co-training impact set: seed {len(changed_docs)}, "
            f"closure {len(rebuild)} doc(s)"
        )
    else:  # arm 6
        rebuild = changed_live
        reason = (
            f"changed-artifact-only refresh of {len(changed_live)} doc(s) "
            f"with text fallback while they rebuild"
        )

    rebuild = frozenset((d, r) for d, r in rebuild if d in present or d in changed_docs)
    fallback = frozenset(doc_id for doc_id, _ in rebuild) if arm == 6 else frozenset()
    return RefreshPlan(
        arm=arm,
        rebuild=rebuild,
        retire=retired,
        fallback_to_text=fallback,
        reason=reason,
    )
