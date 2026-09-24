"""Versioned source catalog with atomic snapshot admission.

Milestone A component: NO models, NO training. This is pure software that is
valuable regardless of the compute gate: a deterministic, testable store of
authored document revisions where a half-updated corpus is never observable.

Every revision is append-only and carries a stable document identity plus a
monotonic `as_of`. A `CorpusSnapshot` is frozen at stage time; it becomes the
active corpus only through `admit` after its checks pass. `serve` reads a
single immutable snapshot object, so an observer can never see a partially
updated corpus.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from secrets import token_hex
from threading import RLock
from types import MappingProxyType

from living_context.artifacts import ArtifactManifest

__all__ = [
    "ArtifactAdmissionError",
    "ArtifactAdmissionReceipt",
    "Catalog",
    "DocumentRevision",
    "SnapshotValidationError",
    "StagedSnapshot",
    "StaleBaseError",
    "VersionedCorpus",
]


@dataclass(frozen=True, slots=True)
class DocumentRevision:
    doc_id: str
    revision: str
    content: str
    edit_kind: str  # initial | number-change | rename | correction | removal | unchanged
    as_of: str


@dataclass(frozen=True, slots=True)
class StagedSnapshot:
    generation: int
    based_on_generation: int
    revisions: Mapping[str, DocumentRevision]  # live documents only (deleted excluded)
    deleted: frozenset[str]
    admitted: bool = False


class StaleBaseError(RuntimeError):
    """The staged snapshot was built against a generation that is no longer current."""


class SnapshotValidationError(ValueError):
    """A candidate snapshot is incomplete or inconsistent with its catalog."""


class ArtifactAdmissionError(SnapshotValidationError):
    """Artifact bytes or source binding do not match the staged candidate."""


@dataclass(frozen=True, slots=True)
class ArtifactAdmissionReceipt:
    """Process-local proof that a trusted caller supplied and checked immutable bytes."""

    generation: int
    based_on_generation: int
    candidate_sha256: str
    artifact_sha256: tuple[tuple[str, str], ...]
    source_sha256: tuple[tuple[str, str], ...]
    consumed_artifacts: tuple[tuple[str, bytes], ...]
    consumed_sources: tuple[tuple[str, bytes], ...]
    token: str


class Catalog:
    """Append-only archive of every authored revision, per document."""

    def __init__(self) -> None:
        self._revisions: dict[str, list[DocumentRevision]] = {}

    def add_revision(self, rev: DocumentRevision) -> None:
        if not rev.doc_id or not rev.revision:
            raise ValueError("doc_id and revision must be non-empty")
        try:
            as_of = date.fromisoformat(rev.as_of)
        except ValueError as exc:
            raise ValueError(f"as_of must be an ISO calendar date: {rev.as_of!r}") from exc
        if as_of.isoformat() != rev.as_of:
            raise ValueError(f"as_of must use canonical YYYY-MM-DD form: {rev.as_of!r}")
        history = self._revisions.setdefault(rev.doc_id, [])
        if any(existing.revision == rev.revision for existing in history):
            raise ValueError(f"revision {rev.revision!r} already exists for {rev.doc_id}")
        if history and date.fromisoformat(history[-1].as_of) >= as_of:
            raise ValueError(
                f"as_of {rev.as_of!r} must follow {history[-1].as_of!r} for {rev.doc_id}"
            )
        history.append(rev)

    def revisions_of(self, doc_id: str) -> tuple[DocumentRevision, ...]:
        return tuple(self._revisions.get(doc_id, ()))

    def revision_at(self, doc_id: str, as_of: str) -> DocumentRevision | None:
        """The revision current at `as_of`: newest with revision as_of <= requested."""
        try:
            requested = date.fromisoformat(as_of)
        except ValueError as exc:
            raise ValueError(f"as_of must be an ISO calendar date: {as_of!r}") from exc
        if requested.isoformat() != as_of:
            raise ValueError(f"as_of must use canonical YYYY-MM-DD form: {as_of!r}")
        chosen: DocumentRevision | None = None
        for rev in self._revisions.get(doc_id, ()):
            if date.fromisoformat(rev.as_of) <= requested:
                chosen = rev
            else:
                break
        return chosen

    def doc_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._revisions))


class VersionedCorpus:
    """Atomic admissions over a Catalog. Serves one immutable snapshot at a time."""

    def __init__(self, catalog: Catalog) -> None:
        self._catalog = catalog
        self._lock = RLock()
        self._generation = 0
        self._active = StagedSnapshot(
            generation=0,
            based_on_generation=0,
            revisions=MappingProxyType({}),
            deleted=frozenset(),
            admitted=True,
        )
        self._snapshots: dict[int, StagedSnapshot] = {0: self._active}
        self._latest_admitted: dict[str, DocumentRevision] = {}
        self._pending_receipts: dict[str, tuple[StagedSnapshot, ArtifactAdmissionReceipt]] = {}

    @staticmethod
    def _candidate_bytes(staged: StagedSnapshot) -> bytes:
        import json

        rows = [
            [doc_id, rev.revision, rev.content, rev.edit_kind, rev.as_of]
            for doc_id, rev in sorted(staged.revisions.items())
        ]
        return json.dumps(
            [staged.generation, staged.based_on_generation, rows, sorted(staged.deleted)],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

    @staticmethod
    def _check_consumed_bytes(
        consumed: tuple[tuple[str, bytes], ...],
        digests: tuple[tuple[str, str], ...],
    ) -> bool:
        return len(consumed) == len(digests) and all(
            key == expected and sha256(data).hexdigest() == digest
            for (key, data), (expected, digest) in zip(consumed, digests, strict=True)
        )

    def verify_artifacts(
        self,
        staged: StagedSnapshot,
        manifests: Mapping[str, ArtifactManifest],
        artifact_bytes: Mapping[str, bytes | bytearray | memoryview],
    ) -> ArtifactAdmissionReceipt:
        """Check a complete candidate and copy all consumed source/artifact bytes.

        This is an optional trusted-caller check. It authenticates neither the
        caller nor the claimed training process. Its receipt is valid only on
        this corpus instance and for this exact staged object.
        """
        with self._lock:
            if staged.admitted or staged.based_on_generation != self._generation:
                raise StaleBaseError("artifact verification requires a fresh staged candidate")
            if staged.generation != self._generation + 1:
                raise ArtifactAdmissionError("candidate generation is not next")
            self._validate_candidate(staged.revisions, staged.deleted)
            if set(manifests) != set(artifact_bytes):
                raise ArtifactAdmissionError("manifest and artifact byte sets differ")
            copied_artifacts: list[tuple[str, bytes]] = []
            target_docs: set[str] = set()
            for artifact_id, manifest in sorted(manifests.items()):
                manifest.validate()
                if manifest.artifact_id != artifact_id:
                    raise ArtifactAdmissionError("manifest artifact id mismatch")
                if not 0 < manifest.trained_at_generation <= staged.generation:
                    raise ArtifactAdmissionError("artifact generation is outside candidate history")
                if len(manifest.checksum) != 64 or any(
                    c not in "0123456789abcdef" for c in manifest.checksum
                ):
                    raise ArtifactAdmissionError("checksum must be lowercase SHA-256")
                supplied = artifact_bytes[artifact_id]
                if not isinstance(supplied, (bytes, bytearray, memoryview)):
                    raise ArtifactAdmissionError("artifact payload must be bytes-like")
                payload = bytes(supplied)
                if sha256(payload).hexdigest() != manifest.checksum:
                    raise ArtifactAdmissionError("artifact digest mismatch")
                for source in manifest.sources:
                    revision = staged.revisions.get(source.doc_id)
                    if revision is None or revision.revision != source.revision:
                        raise ArtifactAdmissionError("manifest source is absent or stale")
                    if source.role == "target":
                        target_docs.add(source.doc_id)
                copied_artifacts.append((artifact_id, payload))
            if target_docs != set(staged.revisions):
                raise ArtifactAdmissionError("candidate lacks target artifact coverage")
            copied_sources = tuple(
                (doc_id, revision.content.encode("utf-8"))
                for doc_id, revision in sorted(staged.revisions.items())
            )
            receipt = ArtifactAdmissionReceipt(
                generation=staged.generation,
                based_on_generation=staged.based_on_generation,
                candidate_sha256=sha256(self._candidate_bytes(staged)).hexdigest(),
                artifact_sha256=tuple(
                    (key, sha256(data).hexdigest()) for key, data in copied_artifacts
                ),
                source_sha256=tuple(
                    (key, sha256(data).hexdigest()) for key, data in copied_sources
                ),
                consumed_artifacts=tuple(copied_artifacts),
                consumed_sources=copied_sources,
                token=token_hex(32),
            )
            self._pending_receipts[receipt.token] = (staged, receipt)
            return receipt

    def discard_artifact_receipt(self, receipt: ArtifactAdmissionReceipt) -> None:
        """Release copied bytes for a candidate the owner will not admit."""
        with self._lock:
            pending = self._pending_receipts.get(receipt.token)
            if pending is not None and pending[1] is receipt:
                del self._pending_receipts[receipt.token]

    @property
    def current_generation(self) -> int:
        with self._lock:
            return self._generation

    def serve(self, generation: int | None = None) -> StagedSnapshot:
        """The complete active corpus. Always a fully admitted snapshot."""
        with self._lock:
            if generation is None:
                return self._active
            try:
                return self._snapshots[generation]
            except KeyError as exc:
                raise KeyError(f"generation {generation} was never admitted") from exc

    def _validate_candidate(
        self,
        revisions: Mapping[str, DocumentRevision],
        deleted: frozenset[str],
    ) -> dict[str, DocumentRevision]:
        candidate = dict(revisions)
        deleted_ids = frozenset(deleted)
        if set(candidate) & set(deleted_ids):
            overlap = sorted(set(candidate) & set(deleted_ids))
            raise SnapshotValidationError(f"live and deleted overlap: {overlap}")
        omitted = set(self._active.revisions) - set(candidate) - set(deleted_ids)
        if omitted:
            raise SnapshotValidationError(
                f"candidate omits live documents without deleting them: {sorted(omitted)}"
            )
        unknown_deleted = set(deleted_ids) - set(self._catalog.doc_ids())
        if unknown_deleted:
            raise SnapshotValidationError(
                f"cannot delete unknown documents: {sorted(unknown_deleted)}"
            )
        for doc_id, revision in candidate.items():
            if doc_id != revision.doc_id:
                raise SnapshotValidationError(
                    f"mapping key {doc_id!r} does not match revision doc_id {revision.doc_id!r}"
                )
            if revision not in self._catalog.revisions_of(doc_id):
                raise SnapshotValidationError(
                    f"revision {doc_id}@{revision.revision} is not present in the catalog"
                )
            latest_admitted = self._latest_admitted.get(doc_id)
            history = self._catalog.revisions_of(doc_id)
            if latest_admitted is not None and history.index(revision) < history.index(
                latest_admitted
            ):
                raise SnapshotValidationError(
                    f"candidate rolls {doc_id} back from {latest_admitted.revision} "
                    f"to {revision.revision}; "
                    "serve an admitted historical generation instead"
                )
        return candidate

    def stage(
        self,
        revisions: Mapping[str, DocumentRevision],
        *,
        deleted: frozenset[str] = frozenset(),
        based_on_generation: int | None = None,
    ) -> StagedSnapshot:
        """Freeze a complete candidate corpus. Nothing is visible until `admit`.

        Omitting a currently live document is only valid when its id is in
        ``deleted``. This prevents a partial update from being mistaken for a
        complete replacement snapshot.
        """
        with self._lock:
            base = self._generation if based_on_generation is None else based_on_generation
            candidate = self._validate_candidate(revisions, frozenset(deleted))
            return StagedSnapshot(
                generation=self._generation + 1,
                based_on_generation=base,
                revisions=MappingProxyType(candidate),
                deleted=frozenset(deleted),
            )

    def stage_changes(
        self,
        updates: Mapping[str, DocumentRevision],
        *,
        deleted: frozenset[str] = frozenset(),
        based_on_generation: int | None = None,
    ) -> StagedSnapshot:
        """Stage disjoint updates and deletions over the complete active snapshot."""
        with self._lock:
            overlap = sorted(set(updates) & set(deleted))
            if overlap:
                raise SnapshotValidationError(f"updates and deleted overlap: {overlap}")
            candidate = dict(self._active.revisions)
            candidate.update(updates)
            for doc_id in deleted:
                candidate.pop(doc_id, None)
            retained_deleted = self._active.deleted - set(updates)
            return self.stage(
                candidate,
                deleted=retained_deleted | deleted,
                based_on_generation=based_on_generation,
            )

    def admit(
        self,
        staged: StagedSnapshot,
        *,
        checks_passed: bool,
        receipt: ArtifactAdmissionReceipt | None = None,
    ) -> StagedSnapshot | None:
        """Make `staged` active ONLY if checks passed and the base is still current.

        On any failure the corpus is untouched — this is the atomicity guarantee:
        the caller sees either the old snapshot or the fully checked new one.
        The caller owns the checks; this API requires an actual Boolean result.
        """
        with self._lock:
            if not isinstance(checks_passed, bool):
                raise TypeError("checks_passed must be a bool")
            if not checks_passed:
                if receipt is not None:
                    self.discard_artifact_receipt(receipt)
                return None
            if staged.admitted:
                raise SnapshotValidationError("an admitted snapshot cannot be admitted again")
            if staged.based_on_generation != self._generation:
                raise StaleBaseError(
                    f"staged on generation {staged.based_on_generation}, current is {self._generation}"
                )
            if staged.generation != self._generation + 1:
                raise SnapshotValidationError(
                    f"staged generation {staged.generation}, expected {self._generation + 1}"
                )
            if receipt is not None:
                pending = self._pending_receipts.get(receipt.token)
                if pending is None or pending[0] is not staged or pending[1] is not receipt:
                    raise ArtifactAdmissionError("receipt was not issued for this candidate")
                if receipt.candidate_sha256 != sha256(self._candidate_bytes(staged)).hexdigest():
                    raise ArtifactAdmissionError("candidate changed after artifact verification")
                if not self._check_consumed_bytes(
                    receipt.consumed_artifacts, receipt.artifact_sha256
                ):
                    raise ArtifactAdmissionError("receipt artifact bytes changed")
                if not self._check_consumed_bytes(receipt.consumed_sources, receipt.source_sha256):
                    raise ArtifactAdmissionError("receipt source bytes changed")
            candidate = self._validate_candidate(staged.revisions, frozenset(staged.deleted))
            self._active = StagedSnapshot(
                generation=staged.generation,
                based_on_generation=staged.based_on_generation,
                revisions=MappingProxyType(candidate),
                deleted=frozenset(staged.deleted),
                admitted=True,
            )
            self._generation = staged.generation
            self._snapshots[self._generation] = self._active
            self._latest_admitted.update(candidate)
            self._pending_receipts.clear()
            return self._active

    def is_visible(self, doc_id: str, revision: str) -> bool:
        """True only if this exact revision is the live one in the ACTIVE corpus."""
        with self._lock:
            live = self._active.revisions.get(doc_id)
            return live is not None and live.revision == revision
