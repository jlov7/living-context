"""Historical embedding-soft-prefix arm (arm 3 proxy).

The retained run trained each prefix on the evaluated Q/A and used a defective
objective and substring scorer. This source now contains corrected
cross-entropy and boundary matching for a future, newly frozen protocol; the
historical JSON remains descriptive only. The prefix is an embedding prefix,
not a persisted per-layer KV cache.

Documented difference, not hidden: this arm trains ONE soft-prefix per canary
fact (full learned refresh) — the expensive, straightforward maintenance path.
It does NOT yet attempt selective refresh or dependency-aware updates (arms
4–6, Milestone C).

Run: uv run python experiments/arm_learned_context.py
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

from living_context.evaluation import answer_matches

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts" / "evidence"
MODEL_ID = "mlx-community/Llama-3.2-1B-Instruct-4bit"

CANARY_QA = [
    {"question": "What is the flow rate of the H-1 pump at 165 bar?", "answer": "55"},
    {"question": "What is the rating of the V-2 valve?", "answer": "200"},
    {"question": "What is the capacity of the T-4 tank?", "answer": "45"},
]

PREFIX_MAX = 64
STEPS = 400
SEED = 7


class PrefixArtifact(nn.Module):
    def __init__(self, n_prefix: int, dim: int) -> None:
        super().__init__()
        init = mx.random.normal((n_prefix, dim)) * 0.02
        self.weights = init.astype(mx.float32)


def hidden_dim(model: nn.Module) -> int:
    tr = mx.arange(3)
    return model.model.embed_tokens(tr).shape[-1]


def _covers(text: str, answer: str) -> bool:
    return answer_matches(text, answer)


def run_learned(model, tokenizer, qa: dict, n_prefix: int, n_steps: int, seed: int) -> dict:
    mx.random.seed(seed)
    h = hidden_dim(model)
    artifact = PrefixArtifact(n_prefix, h)

    # The prefix is the ONLY carrier of the fact: the generation prompt here is
    # the BARE question, exactly like Milestone 0's canary. This is what makes
    # the arm informative — a trained prefix that answers the bare question has
    # actually absorbed the fact from the training document.
    doc_lines = {
        "What is the flow rate of the H-1 pump at 165 bar?": "H-1 pump. Flow 55 L/min at 165 bar. Inlet size 25 mm.",
        "What is the rating of the V-2 valve?": "V-2 valve. Rated 200 bar, pilot-operated.",
        "What is the capacity of the T-4 tank?": "T-4 tank. Volume 45 L, ambient fill only. Drain at bottom.",
    }
    doc_text = doc_lines[qa["question"]]
    train_prompt = f"Document: {doc_text}\n\nQuestion: {qa['question']}\nAnswer:"
    gen_prompt = "Question: {}\nAnswer:".format(qa["question"])

    q_ids = tokenizer.encode(train_prompt)
    a_ids = tokenizer.encode(qa["answer"])
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
    peak = mx.get_active_memory()
    for _ in range(n_steps):
        _, grads = loss_and_grad()
        opt.update(artifact, grads)
        mx.eval(artifact.weights, opt.state)
        peak = max(peak, mx.get_active_memory())
    train_s = time.monotonic() - start

    base = generate(model, tokenizer, prompt=gen_prompt, max_tokens=24)
    t0 = time.monotonic()
    with_artifact = decode_greedy(model, tokenizer, artifact.weights, gen_prompt, max_tokens=16)
    serve_s = time.monotonic() - t0

    base_ok = _covers(base, qa["answer"])
    with_ok = _covers(with_artifact, qa["answer"])
    return {
        "question": qa["question"],
        "answer": qa["answer"],
        "baseline_answer": base.strip()[:80],
        "baseline_has_answer": base_ok,
        "artifact_answer": with_artifact.strip()[:80],
        "artifact_has_answer": with_ok,
        "train_s": round(train_s, 3),
        "serve_s": round(serve_s, 3),
        "peak_mem_bytes": int(peak),
    }


def decode_greedy(model, tokenizer, prefix, prompt: str, max_tokens: int) -> str:
    ids = tokenizer.encode(prompt)
    cur = list(ids)
    out = []
    for _ in range(max_tokens):
        emb = model.model.embed_tokens(mx.array(cur)[None, :])
        emb = mx.concatenate([prefix[None, :, :], emb], axis=1)
        logits = model(inputs=mx.array(cur)[None, :], input_embeddings=emb)
        nxt = int(mx.argmax(logits[0, -1, :], axis=-1))
        out.append(nxt)
        if nxt == tokenizer.eos_token_id:
            break
        cur = cur + [nxt]
    return tokenizer.decode(out)


def main() -> None:
    model, tokenizer = load(MODEL_ID)
    model.eval()
    EVIDENCE.mkdir(parents=True, exist_ok=True)

    rows = []
    for qa in CANARY_QA:
        r = run_learned(model, tokenizer, qa, PREFIX_MAX, STEPS, SEED)
        print(
            f"{qa['question']!r}: base={r['baseline_has_answer']} "
            f"artifact={r['artifact_has_answer']} train={r['train_s']}s "
            f"peak={r['peak_mem_bytes'] / 1e9:.2f}GB"
        )
        rows.append(r)

    summary = {
        "model": MODEL_ID,
        "arm": "3-full-learned-refresh",
        "n_prefix": PREFIX_MAX,
        "n_steps": STEPS,
        "seed": SEED,
        "answered_with_artifact": sum(r["artifact_has_answer"] for r in rows),
        "answered_bare": sum(r["baseline_has_answer"] for r in rows),
        "rows": rows,
    }
    path = EVIDENCE / f"arm-learned-context-{int(time.time())}.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"retained: {path}")
    print(
        f"RESULT learned-arm={summary['answered_with_artifact']}/{len(rows)} "
        f"bare={summary['answered_bare']}/{len(rows)}"
    )


if __name__ == "__main__":
    main()
