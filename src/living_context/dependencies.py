"""Observed training dependency graph.

Records which document revisions were present together during artifact
training. An edge (a -> b) means: training on `b`'s revision also co-trained
`a`'s revision in the same batch, so a later change to one may touch knowledge
that appears to live in the other. Milestone A stores the authored graph
deterministically; Milestone B instruments it from actual runs.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DependencyGraph", "TrainingEdge"]


@dataclass(frozen=True, slots=True)
class TrainingEdge:
    left_doc: str
    left_revision: str
    right_doc: str
    right_revision: str
    batch_id: str


class DependencyGraph:
    """Bipartite-by-batch co-training edges, queryable both directions."""

    def __init__(self) -> None:
        self._edges: list[TrainingEdge] = []
        self._by_left: dict[tuple[str, str], list[TrainingEdge]] = {}

    def add(self, edge: TrainingEdge) -> None:
        self._edges.append(edge)
        self._by_left.setdefault((edge.left_doc, edge.left_revision), []).append(edge)

    def co_trained(self, doc_id: str, revision: str) -> set[tuple[str, str]]:
        """Every (other_doc, other_revision) co-trained with this revision."""
        out: set[tuple[str, str]] = set()
        for e in self._by_left.get((doc_id, revision), ()):
            out.add((e.right_doc, e.right_revision))
        for e in self._edges:
            if e.right_doc == doc_id and e.right_revision == revision:
                out.add((e.left_doc, e.left_revision))
        return out

    def invalidated_by(self, changed: set[tuple[str, str]]) -> set[tuple[str, str]]:
        """The transitive closure of artifacts whose knowledge may be affected."""
        frontier: set[tuple[str, str]] = set(changed)
        seen: set[tuple[str, str]] = set(changed)
        while frontier:
            doc_id, revision = frontier.pop()
            for dep in self.co_trained(doc_id, revision):
                if dep not in seen:
                    seen.add(dep)
                    frontier.add(dep)
        return seen
