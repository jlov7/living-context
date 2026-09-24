"""Milestone C — one document edit through every refresh arm, end to end.

The demonstration the project exists to make:
    old answer -> edit -> refresh -> new answer, with an unrelated question
    provably unaffected, using the REAL measured learned path from Milestone B
    with complete-snapshot atomic admission from the repaired offline core.

Edit: hydraulics/h-tank v2 ("Volume 60 L") -> v3 ("Volume 45 L").

Arms (protocol §4):
- arm 4: changed artifact only
- arm 5: conservative co-training impact set (h-tank + h-pump co-trained)
- arm 6: arm 4 + explicit text fallback while the new artifact is being built

Atomicity is enforced by the harness: the new snapshot is admitted only after
the planned refresh artifacts pass their canary checks; during the refresh the
OLD corpus is what any observer sees.

Run: uv run python experiments/refresh_arms_demo.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx_lm import load
from mlx_lm.generate import generate  # type: ignore[attr-defined]

from living_context.catalog import Catalog, DocumentRevision, VersionedCorpus
from living_context.dependencies import DependencyGraph, TrainingEdge
from living_context.evaluation import answer_matches
from living_context.refresh import plan_refresh

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts" / "evidence"
MODEL_ID = "mlx-community/Llama-3.2-1B-Instruct-4bit"

PREFIX = 64
STEPS = 400
SEED = 7

EDIT = {
    "doc_id": "hydraulics/h-tank",
    "question": "What is the volume of the T-4 tank?",
    "old_answer": "60",
    "new_answer": "45",
}
CO_TRAINED = ("hydraulics/h-tank", "hydraulics/h-pump")  # batch b7 in the graph
UNRELATED_QUESTION = "What is the flow rate of the H-1 pump at 165 bar?"
UNRELATED_ANSWER = "55"
# A cross-document question needs BOTH a revised doc (h-tank) and an
# unrevised one (h-pump). It can only be answered once the refresh is done.
CROSS_DOC_QUESTION = "What is the flow of the H-1 pump at 165 bar and the volume of the T-4 tank?"
CROSS_DOC_ANSWERS = ("55", "45")

FACTS = {
    "hydraulics/h-tank": "T-4 tank. Volume 45 L, ambient fill only. Drain at bottom.",
    "hydraulics/h-pump": "H-1 pump. Flow 55 L/min at 165 bar. Inlet size 25 mm.",
    "hydraulics/h-valve": "V-2 valve. Rated 200 bar, pilot-operated.",
}


class PrefixArtifact(nn.Module):
    def __init__(self, n_prefix: int, dim: int) -> None:
        super().__init__()
        init = mx.random.normal((n_prefix, dim)) * 0.02
        self.weights = init.astype(mx.float32)


def hidden_dim(model: nn.Module) -> int:
    return model.model.embed_tokens(mx.arange(3)).shape[-1]


def _covers(text: str, answer: str) -> bool:
    return answer_matches(text, answer)


def train_artifact(
    model,
    tokenizer,
    doc_text: str,
    question: str,
    answer: str,
    n_prefix: int,
    n_steps: int,
    seed: int,
) -> tuple[PrefixArtifact, float]:
    """Train one artifact so the BARE question is answered from the artifact only."""
    mx.random.seed(seed)
    h = hidden_dim(model)
    artifact = PrefixArtifact(n_prefix, h)
    train_prompt = f"Document: {doc_text}\n\nQuestion: {question}\nAnswer:"
    q_ids = tokenizer.encode(train_prompt)
    a_ids = tokenizer.encode(answer)
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None and a_ids and a_ids[0] == bos:
        a_ids = a_ids[1:]
    full_ids = q_ids + a_ids
    input_ids = mx.array(full_ids)[None, :]
    q_len = len(q_ids)
    L = len(full_ids)

    def loss_fn() -> mx.array:
        emb = model.model.embed_tokens(input_ids)
        emb = mx.concatenate([artifact.weights[None, :, :], emb], axis=1)
        logits = model(inputs=input_ids, input_embeddings=emb)
        pre = logits[:, :-1, :]
        pre = pre[:, n_prefix - 1 : n_prefix - 1 + L, :]
        target = input_ids[0]
        target_logits = mx.take_along_axis(pre, target[None, :, None], axis=-1).squeeze(-1)
        token_losses = mx.logsumexp(pre, axis=-1) - target_logits
        valid = mx.arange(L) >= q_len
        return mx.sum(mx.where(valid[None, :], token_losses, 0.0)) / mx.sum(valid)

    loss_and_grad = nn.value_and_grad(artifact, loss_fn)
    opt = optim.Adam(learning_rate=1e-2)
    model.eval()
    start = time.monotonic()
    for _ in range(n_steps):
        _, grads = loss_and_grad()
        opt.update(artifact, grads)
        mx.eval(artifact.weights, opt.state)
    train_s = time.monotonic() - start
    return artifact, train_s


def answer_with_artifact(model, tokenizer, artifact, question: str, max_tokens: int = 16) -> str:
    gen_prompt = f"Question: {question}\nAnswer:"
    ids = tokenizer.encode(gen_prompt)
    cur = list(ids)
    out = []
    for _ in range(max_tokens):
        emb = model.model.embed_tokens(mx.array(cur)[None, :])
        emb = mx.concatenate([artifact.weights[None, :, :], emb], axis=1)
        logits = model(inputs=mx.array(cur)[None, :], input_embeddings=emb)
        nxt = int(mx.argmax(logits[0, -1, :], axis=-1))
        out.append(nxt)
        if nxt == tokenizer.eos_token_id:
            break
        cur = cur + [nxt]
    return tokenizer.decode(out)


def answer_text(model, tokenizer, doc_text: str, question: str, max_tokens: int = 48) -> str:
    prompt = f"Document: {doc_text}\n\nQuestion: {question}\nAnswer:"
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)


def build_corpus() -> tuple[Catalog, VersionedCorpus, dict[str, str]]:
    """Admit the corpus at v2 (pre-edit). v3 is staged later as the edit."""
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
    corpus = VersionedCorpus(catalog)
    v2_live = {}
    for doc_id in catalog.doc_ids():
        revs = catalog.revisions_of(doc_id)
        if revs and len(revs) >= 2 and revs[1].edit_kind != "deleted":
            v2_live[doc_id] = revs[1]  # v2: the pre-edit current revision
        elif revs and revs[0].edit_kind != "deleted":
            v2_live[doc_id] = revs[0]
    corpus.admit(corpus.stage(v2_live), checks_passed=True)
    current = {doc_id: catalog.revisions_of(doc_id)[-1].content for doc_id in catalog.doc_ids()}
    return catalog, corpus, current


def main() -> None:
    model, tokenizer = load(MODEL_ID)
    model.eval()
    EVIDENCE.mkdir(parents=True, exist_ok=True)

    catalog, corpus, current = build_corpus()
    graph = DependencyGraph()
    # Dependencies describe the artifacts that exist before the edit.
    graph.add(TrainingEdge("hydraulics/h-tank", "v2", "hydraulics/h-pump", "v2", "b7"))

    # ---------------- phase 1: OLD CORPUS (v2 facts) --------------------------
    old_art = train_artifact(
        model,
        tokenizer,
        "T-4 tank. Volume 60 L, ambient fill only. Drain at bottom.",
        EDIT["question"],
        "60",
        PREFIX,
        STEPS,
        SEED,
    )[0]
    old_answer = answer_with_artifact(model, tokenizer, old_art, EDIT["question"])
    old_ok = _covers(old_answer, "60")
    print(f"OLD corpus (v2): answer={old_answer.strip()[:40]!r} correct={old_ok}")

    pump_art = train_artifact(
        model,
        tokenizer,
        FACTS["hydraulics/h-pump"],
        UNRELATED_QUESTION,
        UNRELATED_ANSWER,
        PREFIX,
        STEPS,
        SEED,
    )[0]
    pump_before = answer_with_artifact(model, tokenizer, pump_art, UNRELATED_QUESTION)
    pump_before_ok = _covers(pump_before, UNRELATED_ANSWER)
    print(f"unrelated h-pump before edit: correct={pump_before_ok}")

    # ---------------- phase 2: THE EDIT (v2 -> v3) ----------------------------
    # New snapshot staged but NOT yet admitted: observers still see the old.
    staged_v3 = corpus.stage_changes(
        {"hydraulics/h-tank": catalog.revisions_of("hydraulics/h-tank")[-1]}
    )
    assert not staged_v3.admitted
    still_old = corpus.serve()
    assert (
        still_old.revisions["hydraulics/h-tank"].content
        != catalog.revisions_of("hydraulics/h-tank")[-1].content
    )

    # ---------------- phase 3: refresh arms -----------------------------------
    # The plans differ ONLY in rebuild policy; the v2->v3 admission is ONE
    # atomic edit shared by all arms. Plans are computed BEFORE admission so
    # every arm reasons from the same pre-edit old snapshot. `present` is the
    # FULL old snapshot (every doc and its current revision) so arm 5's
    # co-training closure can include unaffected docs like h-pump.
    old_snapshot = corpus.serve()
    present = dict(old_snapshot.revisions)
    plans = {arm: plan_refresh(arm, old_snapshot, staged_v3, present, graph) for arm in (4, 5, 6)}
    arms_out = {}
    arm_checks: dict[int, bool] = {}
    for arm, plan in plans.items():
        rebuilt = {}
        for doc_id, _rev in sorted(plan.rebuild):
            if doc_id == EDIT["doc_id"]:
                question = EDIT["question"]
                answer = EDIT["new_answer"]
            elif doc_id == "hydraulics/h-pump":
                question = UNRELATED_QUESTION
                answer = UNRELATED_ANSWER
            else:
                raise RuntimeError(f"no refresh check declared for {doc_id}")
            art, _ = train_artifact(
                model,
                tokenizer,
                staged_v3.revisions[doc_id].content,
                question,
                answer,
                PREFIX,
                STEPS,
                SEED,
            )
            rebuilt[doc_id] = art

        if EDIT["doc_id"] in rebuilt:
            grand = answer_with_artifact(
                model, tokenizer, rebuilt[EDIT["doc_id"]], EDIT["question"]
            )
            grand_ok = _covers(grand, EDIT["new_answer"])
        else:
            grand, grand_ok = "UNREFRESHED", False
        unaffected = answer_with_artifact(model, tokenizer, pump_art, UNRELATED_QUESTION)
        unaffected_ok = _covers(unaffected, UNRELATED_ANSWER)
        rebuilt_checks = []
        for doc_id, artifact in rebuilt.items():
            if doc_id == EDIT["doc_id"]:
                rebuilt_checks.append(grand_ok)
            else:
                rebuilt_checks.append(
                    _covers(
                        answer_with_artifact(model, tokenizer, artifact, UNRELATED_QUESTION),
                        UNRELATED_ANSWER,
                    )
                )

        if arm == 6:
            fallback = answer_text(model, tokenizer, current[EDIT["doc_id"]], EDIT["question"])
            fallback_ok = _covers(fallback, EDIT["new_answer"])
        else:
            fallback, fallback_ok = "", False
        arm_checks[arm] = grand_ok and unaffected_ok and all(rebuilt_checks)

        arms_out[arm] = {
            "plan_reason": plan.reason,
            "rebuild_docs": sorted(d for d, _ in plan.rebuild),
            "fallback_to_text": sorted(plan.fallback_to_text),
            "new_answer": grand.strip()[:60],
            "new_answer_correct": grand_ok,
            "unaffected_correct": unaffected_ok,
            "fallback_answer": fallback.strip()[:60],
            "fallback_correct": fallback_ok,
            "refresh_checks_passed": arm_checks[arm],
        }
        print(
            f"arm {arm}: {plan.reason} -> new_correct={grand_ok} "
            f"unaffected_correct={unaffected_ok} fallback_correct={fallback_ok}"
        )

    admitted = corpus.admit(staged_v3, checks_passed=all(arm_checks.values()))
    for arm_result in arms_out.values():
        arm_result["corpus_after_admission_generation"] = corpus.current_generation
    if admitted is not None:
        cross_doc = answer_text(
            model,
            tokenizer,
            " ".join(
                admitted.revisions[doc_id].content
                for doc_id in ("hydraulics/h-pump", "hydraulics/h-tank")
            ),
            CROSS_DOC_QUESTION,
        )
        cross_doc_ok = all(_covers(cross_doc, a) for a in CROSS_DOC_ANSWERS)
    else:
        cross_doc = "NOT RUN: refresh checks failed; candidate snapshot was not admitted"
        cross_doc_ok = False
    print(f"cross-document text-context check after admission: correct={cross_doc_ok}")

    summary = {
        "model": MODEL_ID,
        "edit": EDIT,
        "phase1_old_answer": old_answer.strip()[:60],
        "phase1_old_correct": old_ok,
        "phase2_atomic_staging_held": True,
        "candidate_admitted": admitted is not None,
        "unrelated_before_correct": pump_before_ok,
        "arms": arms_out,
        "cross_document_answer": cross_doc.strip()[:80],
        "cross_document_correct": cross_doc_ok,
        "cross_document_question": CROSS_DOC_QUESTION,
        "cross_document_path": "text context after source-snapshot admission",
        "unrelated_question": UNRELATED_QUESTION,
        "unrelated_answer": UNRELATED_ANSWER,
        "co_training_batch": "b7",
    }
    path = EVIDENCE / f"refresh-arms-demo-{int(time.time())}.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"retained: {path}")


if __name__ == "__main__":
    main()
