"""Artifact manifests and the authored revision corpus load cleanly."""

from __future__ import annotations

import json
from pathlib import Path

from living_context.artifacts import ArtifactManifest, ManifestError, SourceRef

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "experiments" / "revision_sequences.jsonl"


def test_manifest_requires_a_training_target() -> None:
    m = ArtifactManifest(
        artifact_id="a1",
        sources=(SourceRef("x", "v1", "distractor"),),
        base_model_revision="base-1b",
        trained_at_generation=1,
        checksum="0123456789abcdef",
    )
    try:
        m.validate()
        raise AssertionError("manifest without a target was not rejected")
    except ManifestError:
        pass


def test_manifest_validation_passes_for_well_formed() -> None:
    m = ArtifactManifest(
        artifact_id="a1",
        sources=(SourceRef("x", "v1", "target"), SourceRef("y", "v2", "distractor")),
        base_model_revision="base-1b",
        trained_at_generation=1,
        checksum="0123456789abcdef",
    )
    m.validate()


def test_manifest_rejects_bad_role() -> None:
    m = ArtifactManifest(
        artifact_id="a1",
        sources=(SourceRef("x", "v1", "watcher"),),
        base_model_revision="base-1b",
        trained_at_generation=1,
        checksum="0123456789abcdef",
    )
    try:
        m.validate()
        raise AssertionError("bad role was not rejected")
    except ManifestError:
        pass


def test_corpus_has_24_docs_6_families_3_revisions() -> None:
    rows = [json.loads(line) for line in CORPUS.read_text().splitlines() if line.strip()]
    assert len(rows) == 72
    families = {r["family"] for r in rows}
    assert families == {"hydraulics", "telemetry", "compliance", "logistics", "supply", "controls"}
    docs = {r["doc_id"] for r in rows}
    assert len(docs) == 24
    per_doc = {}
    for r in rows:
        per_doc.setdefault(r["doc_id"], []).append(r["revision"])
    assert all(revs == ["v1", "v2", "v3"] for revs in per_doc.values())


def test_edge_kinds_are_present() -> None:
    rows = [json.loads(line) for line in CORPUS.read_text().splitlines() if line.strip()]
    kinds = {r["edit_kind"] for r in rows}
    assert {
        "initial",
        "number-change",
        "rename",
        "revoked-fact",
        "correction",
        "contradictory-update",
        "cross-reference",
        "deleted",
        "unchanged",
    } <= kinds
