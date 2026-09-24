"""A caller-owned, model-free text retrieval path over admitted generations."""

from __future__ import annotations

from dataclasses import dataclass

from living_context.catalog import VersionedCorpus
from living_context.refresh import RefreshPlan


@dataclass(frozen=True, slots=True)
class TextRead:
    doc_id: str
    revision: str
    generation: int
    content: str
    current: bool


class TextReader:
    """CLI application reader; every response names its admitted generation."""

    def __init__(self, corpus: VersionedCorpus) -> None:
        self._corpus = corpus

    def read(self, doc_id: str, *, generation: int | None = None) -> TextRead | None:
        snapshot = self._corpus.serve(generation)
        revision = snapshot.revisions.get(doc_id)
        if revision is None:
            return None
        return TextRead(
            doc_id,
            revision.revision,
            snapshot.generation,
            revision.content,
            snapshot.generation == self._corpus.current_generation,
        )

    def fallback(self, doc_id: str, plan: RefreshPlan, *, generation: int) -> TextRead | None:
        """Read admitted text when a plan names this document for fallback.

        The caller must pass the generation it believes current. A pending source
        candidate cannot be presented as current until it is admitted.
        """
        if doc_id not in plan.fallback_to_text:
            raise ValueError("document is not in the fallback plan")
        if doc_id in plan.retire:
            raise ValueError("retired document cannot use text fallback")
        if generation != self._corpus.current_generation:
            raise ValueError("requested fallback generation is stale")
        wanted = {revision for doc, revision in plan.rebuild if doc == doc_id}
        if len(wanted) != 1:
            raise ValueError("fallback plan lacks one exact target revision")
        result = self.read(doc_id, generation=generation)
        if result is None or result.revision not in wanted:
            raise ValueError("planned fallback revision is not admitted")
        if not result.current:
            raise ValueError("requested fallback generation became stale")
        return result
