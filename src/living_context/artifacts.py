"""Typed artifact manifests — provenance for learned-context artifacts.

A manifest records WHICH source revisions trained an artifact and what role each
played (training target vs distractor). This is what makes "the old answer /
update / new answer" demonstration inspectable: an answer can name the exact
artifacts and revisions it was served from.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ArtifactManifest", "ManifestError", "SourceRef"]


@dataclass(frozen=True, slots=True)
class SourceRef:
    doc_id: str
    revision: str
    role: str  # "target" | "distractor"


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    artifact_id: str
    sources: tuple[SourceRef, ...]
    base_model_revision: str
    trained_at_generation: int
    checksum: str

    def validate(self) -> None:
        if not self.sources:
            raise ManifestError(f"artifact {self.artifact_id}: no sources")
        if not any(s.role == "target" for s in self.sources):
            raise ManifestError(f"artifact {self.artifact_id}: no training target")
        bad = {s.role for s in self.sources} - {"target", "distractor"}
        if bad:
            raise ManifestError(f"artifact {self.artifact_id}: bad role(s) {bad}")
        if len(self.checksum) < 16:
            raise ManifestError(f"artifact {self.artifact_id}: checksum too short")


class ManifestError(RuntimeError):
    pass
