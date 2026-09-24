"""Historical runner for the 2026-09-21 trained-question demonstration.

It trains each prefix on the same question and answer later scored, so its
retained outputs are not held-out evidence. The 2026-09-22 audit corrected the
loss and answer matcher for future protocols. The historical protocol is
blocked from rerun because changing runner semantics invalidates comparison.
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

from living_context.catalog import Catalog, DocumentRevision
from living_context.evaluation import answer_matches

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts" / "evidence"
PILOT_YAML = ROOT / "experiments" / "pilot.yaml"
MODEL_ID = "mlx-community/Llama-3.2-1B-Instruct-4bit"

PREFIX = 64
STEPS = 400
SEED = 7


class PrefixArtifact(nn.Module):
    def __init__(self, n_prefix: int, dim: int) -> None:
        super().__init__()
        init = mx.random.normal((n_prefix, dim)) * 0.02
        self.weights = init.astype(mx.float32)


def hidden_dim(model: nn.Module) -> int:
    return model.model.embed_tokens(mx.arange(3)).shape[-1]


def _covers(text: str, answer: str) -> bool:
    return answer_matches(text, answer)


def train_artifact(model, tokenizer, doc_text, question, answer, seed) -> tuple[object, float]:
    mx.random.seed(seed)
    h = hidden_dim(model)
    artifact = PrefixArtifact(PREFIX, h)
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
        pre = pre[:, PREFIX - 1 : PREFIX - 1 + L, :]
        target = input_ids[0]
        target_logits = mx.take_along_axis(pre, target[None, :, None], axis=-1).squeeze(-1)
        token_losses = mx.logsumexp(pre, axis=-1) - target_logits
        valid = mx.arange(L) >= q_len
        return mx.sum(mx.where(valid[None, :], token_losses, 0.0)) / mx.sum(valid)

    loss_and_grad = nn.value_and_grad(artifact, loss_fn)
    opt = optim.Adam(learning_rate=1e-2)
    model.eval()
    start = time.monotonic()
    peak = mx.get_active_memory()
    for _ in range(STEPS):
        _, grads = loss_and_grad()
        opt.update(artifact, grads)
        mx.eval(artifact.weights, opt.state)
        peak = max(peak, mx.get_active_memory())
    train_s = time.monotonic() - start
    return artifact, train_s, int(peak)


def answer_with_artifact(model, tokenizer, artifact, question) -> str:
    gen_prompt = f"Question: {question}\nAnswer:"
    ids = tokenizer.encode(gen_prompt)
    cur = list(ids)
    out = []
    for _ in range(16):
        emb = model.model.embed_tokens(mx.array(cur)[None, :])
        emb = mx.concatenate([artifact.weights[None, :, :], emb], axis=1)
        logits = model(inputs=mx.array(cur)[None, :], input_embeddings=emb)
        nxt = int(mx.argmax(logits[0, -1, :], axis=-1))
        out.append(nxt)
        if nxt == tokenizer.eos_token_id:
            break
        cur = cur + [nxt]
    return tokenizer.decode(out)


def answer_text(model, tokenizer, doc_text, question) -> tuple[str, float]:
    prompt = f"Document: {doc_text}\n\nQuestion: {question}\nAnswer:"
    t0 = time.monotonic()
    out = generate(model, tokenizer, prompt=prompt, max_tokens=48)
    return out, time.monotonic() - t0


def main() -> None:
    import yaml

    protocol = yaml.safe_load(PILOT_YAML.read_text())
    assert protocol["protocol"]["status"] == "FROZEN", "pilot.yaml is not frozen"
    if protocol["protocol"]["name"] == "living-context-pilot-v1":
        raise RuntimeError(
            "pilot v1 is an immutable historical trained-question run; use "
            "scripts/check_reproduction.py. Freeze a new held-out protocol before "
            "any new model execution."
        )
    questions = protocol["questions"]
    repeats = protocol["design"]["repeats"]

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
    current = {doc_id: catalog.revisions_of(doc_id)[-1].content for doc_id in catalog.doc_ids()}

    model, tokenizer = load(MODEL_ID)
    model.eval()

    rows = []
    deviations: list[str] = []
    for q in questions:
        doc_text = current[q["doc"]]
        for repeat in range(1, repeats + 1):
            row = {
                "question": q["question"],
                "answer": q["answer"],
                "family": q["family"],
                "repeat": repeat,
            }
            # arm 1: cached text, fresh index at current revision
            arm1_out, arm1_s = answer_text(model, tokenizer, doc_text, q["question"])
            row["arm_1_answer"] = arm1_out.strip()[:80]
            row["arm_1_correct"] = _covers(arm1_out, q["answer"])
            row["arm_1_serving_s"] = round(arm1_s, 3)
            # arm 4: learned selective refresh at current revision
            r = train_artifact(
                model, tokenizer, doc_text, q["question"], q["answer"], SEED + repeat
            )
            artifact, train_s, peak = r
            arm4_out = answer_with_artifact(model, tokenizer, artifact, q["question"])
            row["arm_4_answer"] = arm4_out.strip()[:80]
            row["arm_4_correct"] = _covers(arm4_out, q["answer"])
            row["arm_4_refresh_s"] = round(train_s, 3)
            row["arm_4_peak_mem_bytes"] = peak
            rows.append(row)
            print(
                f"[{q['id']}] repeat {repeat}: arm1={row['arm_1_correct']} "
                f"arm4={row['arm_4_correct']} (train {train_s:.1f}s)"
            )

    # Tables from retained attempts — including failures.
    by_family = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)

    summary = {
        "protocol": {**protocol["protocol"], "frozen_at": str(protocol["protocol"]["frozen_at"])},
        "machine": protocol["machine"],
        "deviations": deviations,
        "attempts": len(rows),
        "arm_1_correct": sum(r["arm_1_correct"] for r in rows),
        "arm_4_correct": sum(r["arm_4_correct"] for r in rows),
        "per_family": {
            fam: {
                "arm_1": sum(r["arm_1_correct"] for r in fam_rows),
                "arm_4": sum(r["arm_4_correct"] for r in fam_rows),
                "n": len(fam_rows),
            }
            for fam, fam_rows in by_family.items()
        },
        "lifecycle_cost_physical": {
            "arm_1_serving_total_s": round(sum(r["arm_1_serving_s"] for r in rows), 3),
            "arm_4_refresh_total_s": round(sum(r["arm_4_refresh_s"] for r in rows), 3),
            "peak_mem_bytes": max(r["arm_4_peak_mem_bytes"] for r in rows),
        },
        "rows": rows,
    }
    path = EVIDENCE / f"pilot-{int(time.time())}.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"retained: {path}")
    print(
        f"RESULT arm1={summary['arm_1_correct']}/{summary['attempts']} "
        f"arm4={summary['arm_4_correct']}/{summary['attempts']} "
        f"refresh_total={summary['lifecycle_cost_physical']['arm_4_refresh_total_s']}s "
        f"(physical units, no price conversion)"
    )


if __name__ == "__main__":
    main()
