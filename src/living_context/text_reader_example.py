"""An identified local CLI caller for admitted text fallback; no model use."""

from __future__ import annotations

from living_context.catalog import Catalog, DocumentRevision, VersionedCorpus
from living_context.refresh import plan_refresh
from living_context.text_reader import TextReader


def walkthrough() -> list[str]:
    source = Catalog()
    first = DocumentRevision("policy", "r1", "Limit: 10", "initial", "2026-01-01")
    second = DocumentRevision("policy", "r2", "Limit: 20", "number-change", "2026-01-02")
    source.add_revision(first)
    source.add_revision(second)
    corpus = VersionedCorpus(source)
    current = corpus.admit(corpus.stage({"policy": first}), checks_passed=True)
    assert current is not None
    reader = TextReader(corpus)
    pending = corpus.stage_changes({"policy": second})
    plan = plan_refresh(6, current, pending, current.revisions)
    before = reader.read("policy")
    assert before is not None
    admitted = corpus.admit(pending, checks_passed=True)
    assert admitted is not None
    after = reader.fallback("policy", plan, generation=admitted.generation)
    assert after is not None
    deletion = corpus.stage_changes({}, deleted=frozenset({"policy"}))
    corpus.admit(deletion, checks_passed=True)
    return [
        f"before admission: generation={before.generation} revision={before.revision} text={before.content}",
        f"after admission: generation={after.generation} revision={after.revision} text={after.content}",
        f"after deletion: current={reader.read('policy')}",
    ]


if __name__ == "__main__":
    for line in walkthrough():
        print(line)
