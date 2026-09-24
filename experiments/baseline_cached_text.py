"""Historical text-retrieval arm with a rebuilt in-memory document map.

The essential baseline this project must beat honestly: ordinary retrieval
done WELL. It reads the CURRENT revision of each document from the catalog
(fresh indexing), retrieves by lexical match against the question, and answers
with the frozen reader model given the retrieved text in context.

The document map is built once, but retrieved text is prefixed again on every
generation. This script does not measure a persistent prompt or KV cache. The
historical learned arm used only three of these five questions and a different
generation limit, so the retained arms are not a matched comparison.

Canary discipline: every accepted answer is a fabricated fact the base model
cannot know; we separate "base model happened to answer" from "the retrieved
text carried the answer" via a control query WITHOUT retrieval context.

Run: uv run python experiments/baseline_cached_text.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mlx_lm import load
from mlx_lm.generate import generate  # type: ignore[attr-defined]

from living_context.catalog import Catalog, DocumentRevision
from living_context.evaluation import answer_matches

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts" / "evidence"
MODEL_ID = "mlx-community/Llama-3.2-1B-Instruct-4bit"

# Canary Q/A authored in the corpus: fabricated values the base model cannot know.
CANARY_QA = [
    {"question": "What is the flow rate of the H-1 pump at 165 bar?", "answer": "55"},
    {"question": "What is the rating of the V-2 valve?", "answer": "200"},
    {"question": "What is the capacity of the T-4 tank?", "answer": "45"},
    {"question": "What is the release status of batch B-120?", "answer": "revoked"},
    {"question": "Which alloy does spec SP-1 require?", "answer": "AL-7075"},
]


def _load_corpus() -> Catalog:
    """Load the authored revision sequence into a Catalog, current = v3."""
    catalog = Catalog()
    for line in (ROOT / "experiments" / "revision_sequences.jsonl").read_text().splitlines():
        row = json.loads(line)
        catalog.add_revision(
            DocumentRevision(
                doc_id=row["doc_id"],
                revision=row["revision"],
                content=row["content"],
                edit_kind=row["edit_kind"],
                as_of=row["as_of"],
            )
        )
    return catalog


def _index_current(catalog: Catalog) -> dict[str, str]:
    """Fresh index: the LATEST revision of every document."""
    out: dict[str, str] = {}
    for doc_id in catalog.doc_ids():
        rev = catalog.revisions_of(doc_id)[-1]
        out[doc_id] = rev.content
    return out


def _score(question: str, doc_text: str) -> int:
    """Lexical retrieval score: number of question terms present in the doc."""
    terms = {t.lower() for t in question.replace("?", "").split() if len(t) > 2}
    blob = doc_text.lower()
    return sum(1 for t in terms if t in blob)


def _retrieve(index: dict[str, str], question: str) -> str | None:
    scored = [(-_score(question, text), doc_id) for doc_id, text in index.items()]
    if not scored:
        return None
    scored.sort()
    return index[scored[0][1]]


def _answer_with_context(model, tokenizer, doc_text: str, question: str) -> str:
    prompt = f"Document:\n{doc_text}\n\nQuestion: {question}\nAnswer:"
    return generate(model, tokenizer, prompt=prompt, max_tokens=24)


def _answer_bare(model, tokenizer, question: str) -> str:
    return generate(model, tokenizer, prompt=f"Question: {question}\nAnswer:", max_tokens=24)


def _covers(text: str, answer: str) -> bool:
    return answer_matches(text, answer)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=0)  # placeholder parity with artifact arm
    args = ap.parse_args()
    _ = args

    catalog = _load_corpus()
    index = _index_current(catalog)

    model, tokenizer = load(MODEL_ID)
    model.eval()

    rows = []
    for qa in CANARY_QA:
        doc_text = _retrieve(index, qa["question"])
        t0 = time.monotonic()
        with_ctx = _answer_with_context(model, tokenizer, doc_text or "", qa["question"])
        t1 = time.monotonic()
        bare = _answer_bare(model, tokenizer, qa["question"])
        t2 = time.monotonic()
        rows.append(
            {
                "question": qa["question"],
                "answer": qa["answer"],
                "retrieved_doc": doc_text,
                "with_context_answer": with_ctx.strip()[:120],
                "with_context_has_answer": _covers(with_ctx, qa["answer"]),
                "bare_answer": bare.strip()[:120],
                "bare_has_answer": _covers(bare, qa["answer"]),
                "retrieval_gen_s": round(t1 - t0, 3),
                "bare_gen_s": round(t2 - t1, 3),
            }
        )
        print(
            f"{qa['question']!r}: ctx_has={rows[-1]['with_context_has_answer']} "
            f"bare_has={rows[-1]['bare_has_answer']} "
            f"(retrieve(+gen) {rows[-1]['retrieval_gen_s']}s, bare {rows[-1]['bare_gen_s']}s)"
        )

    summary = {
        "model": MODEL_ID,
        "arm": "1-cached-text-fresh-index",
        "indexed_docs": len(index),
        "retrieved_and_answered": sum(r["with_context_has_answer"] for r in rows),
        "bare_answered_without_context": sum(r["bare_has_answer"] for r in rows),
        "retained": True,
        "rows": rows,
    }
    path = EVIDENCE / f"baseline-cached-text-{int(time.time())}.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"retained: {path}")
    print(
        f"RESULT with_context={summary['retrieved_and_answered']}/{len(rows)} "
        f"bare={summary['bare_answered_without_context']}/{len(rows)} "
        f"-> canary discipline holds: {(summary['bare_answered_without_context'] < summary['retrieved_and_answered'])}"
    )


if __name__ == "__main__":
    main()
