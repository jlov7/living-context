"""Exact candidate and consumed-byte binding for the optional admission path."""

from dataclasses import replace
from hashlib import sha256
from typing import Any, cast

import pytest

from living_context.artifacts import ArtifactManifest, SourceRef
from living_context.catalog import (
    ArtifactAdmissionError,
    Catalog,
    DocumentRevision,
    StaleBaseError,
    VersionedCorpus,
)


def revision(doc: str, number: int) -> DocumentRevision:
    return DocumentRevision(
        doc, f"r{number}", f"{doc} value {number}", "correction", f"2026-01-{number:02d}"
    )


def fixture():
    catalog = Catalog()
    first = revision("a", 1)
    catalog.add_revision(first)
    corpus = VersionedCorpus(catalog)
    staged = corpus.stage({"a": first})
    data = bytearray(b"learned-prefix")
    manifest = ArtifactManifest(
        "a-artifact",
        (SourceRef("a", "r1", "target"),),
        "frozen-model",
        1,
        sha256(data).hexdigest(),
    )
    return catalog, corpus, staged, data, manifest


def test_checked_bytes_are_copied_and_candidate_admits() -> None:
    _, corpus, staged, data, manifest = fixture()
    receipt = corpus.verify_artifacts(
        staged, {manifest.artifact_id: manifest}, {manifest.artifact_id: data}
    )
    data[:] = b"replaced"
    assert receipt.consumed_artifacts[0][1] == b"learned-prefix"
    assert corpus.admit(staged, checks_passed=True, receipt=receipt) is not None
    assert corpus.serve().generation == 1
    with pytest.raises(ArtifactAdmissionError, match="not issued"):
        corpus.admit(corpus.stage_changes({}), checks_passed=True, receipt=receipt)


@pytest.mark.parametrize("kind", ["digest", "missing", "stale", "incomplete", "generation"])
def test_invalid_evidence_rejects_without_changing_reader(kind: str) -> None:
    catalog, corpus, staged, data, manifest = fixture()
    if kind == "digest":
        manifest = replace(manifest, checksum="0" * 64)
    elif kind == "stale":
        manifest = replace(manifest, sources=(SourceRef("a", "old", "target"),))
    elif kind == "generation":
        manifest = replace(manifest, trained_at_generation=2)
    elif kind == "incomplete":
        other = revision("b", 1)
        catalog.add_revision(other)
        staged = corpus.stage({"a": catalog.revisions_of("a")[0], "b": other})
    manifests = {} if kind == "missing" else {manifest.artifact_id: manifest}
    payloads = {} if kind == "missing" else {manifest.artifact_id: data}
    with pytest.raises(ArtifactAdmissionError):
        corpus.verify_artifacts(staged, manifests, payloads)
    assert corpus.current_generation == 0
    assert corpus.serve().revisions == {}


def test_receipt_cannot_cross_candidate_corpus_or_stale_base() -> None:
    catalog, corpus, staged, data, manifest = fixture()
    receipt = corpus.verify_artifacts(
        staged, {manifest.artifact_id: manifest}, {manifest.artifact_id: data}
    )
    other_corpus = VersionedCorpus(catalog)
    other_stage = other_corpus.stage({"a": catalog.revisions_of("a")[0]})
    with pytest.raises(ArtifactAdmissionError, match="not issued"):
        other_corpus.admit(other_stage, checks_passed=True, receipt=receipt)
    assert corpus.admit(staged, checks_passed=False, receipt=receipt) is None
    corpus.admit(corpus.stage_changes({}), checks_passed=True)
    with pytest.raises(StaleBaseError):
        corpus.admit(staged, checks_passed=True, receipt=receipt)


def test_numeric_length_is_not_artifact_bytes() -> None:
    _, corpus, staged, _data, manifest = fixture()
    with pytest.raises(ArtifactAdmissionError, match="bytes-like"):
        corpus.verify_artifacts(
            staged, {manifest.artifact_id: manifest}, cast(Any, {manifest.artifact_id: 14})
        )
    assert corpus.current_generation == 0


def test_unchanged_prior_artifact_can_be_reused_and_rejected_receipt_discarded() -> None:
    catalog = Catalog()
    a1, a2, b1 = revision("a", 1), revision("a", 2), revision("b", 1)
    for value in (a1, a2, b1):
        catalog.add_revision(value)
    corpus = VersionedCorpus(catalog)
    corpus.admit(corpus.stage({"a": a1, "b": b1}), checks_passed=True)
    candidate = corpus.stage_changes({"a": a2})
    old_bytes, new_bytes = b"existing-b", b"rebuilt-a"
    manifests = {
        "a": ArtifactManifest(
            "a", (SourceRef("a", "r2", "target"),), "model", 2, sha256(new_bytes).hexdigest()
        ),
        "b": ArtifactManifest(
            "b", (SourceRef("b", "r1", "target"),), "model", 1, sha256(old_bytes).hexdigest()
        ),
    }
    receipt = corpus.verify_artifacts(candidate, manifests, {"a": new_bytes, "b": old_bytes})
    assert corpus.admit(candidate, checks_passed=False, receipt=receipt) is None
    with pytest.raises(ArtifactAdmissionError, match="not issued"):
        corpus.admit(candidate, checks_passed=True, receipt=receipt)
    receipt = corpus.verify_artifacts(candidate, manifests, {"a": new_bytes, "b": old_bytes})
    assert corpus.admit(candidate, checks_passed=True, receipt=receipt) is not None
