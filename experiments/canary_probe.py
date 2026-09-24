"""Historical soft-prefix canary runner, corrected for future protocols.

Historical feasibility scope, not part of the maintained replay:
- This is a local soft-prefix probe, not a Cartridges reproduction.
- The mechanism question was whether a compact trainable prefix
  artifact could demonstrably change a frozen base model's output on this machine,
  reproducibly, and at what cost?

Mechanism: a trainable soft-prefix consumed via `input_embeddings`. The base
model weights are frozen. The artifact is a small learned prefix embedding that
gets prepended to the query; it is trained on one canary fact the base model
cannot know, then tested on a question that requires that fact.

The retained runs used a defective objective and substring scorer, so they show
output influence rather than validated fact recall. This source now uses
cross-entropy for any future authorized run. The artifact is an embedding soft
prefix; the model recomputes activations at inference.

Usage:
    .venv/bin/python experiments/canary_probe.py --prefix 32 --steps 300
    .venv/bin/python experiments/canary_probe.py --sweep
"""

from __future__ import annotations

import argparse
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
CANARY_ANSWER = "QX-7743-KB"
QUESTION = "What is the allocation code for the Kessler Battery?"


class PrefixArtifact(nn.Module):
    """The learnable artifact: a trainable prefix of embedding vectors.

    The ONLY trainable thing in the whole probe. Everything else is frozen.
    """

    def __init__(self, n_prefix: int, dim: int) -> None:
        super().__init__()
        init = mx.random.normal((n_prefix, dim)) * 0.02
        self.weights = init.astype(mx.float32)


def load_frozen() -> tuple[nn.Module, object]:
    model, tokenizer = load(MODEL_ID)
    model.freeze()
    return model, tokenizer


def hidden_dim(model: nn.Module) -> int:
    tr = mx.arange(3)
    return model.model.embed_tokens(tr).shape[-1]


def run_canary(
    model: nn.Module, tokenizer: object, n_prefix: int, n_steps: int, seed: int = 7
) -> dict:
    """Train one prefix artifact on the canary fact; measure its influence.

    Returns a retained-evidence dict with timings and pass/fail.
    """
    mx.random.seed(seed)
    h = hidden_dim(model)
    artifact = PrefixArtifact(n_prefix, h)

    q_ids = tokenizer.encode(QUESTION)
    a_ids = tokenizer.encode(CANARY_ANSWER)
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None and a_ids and a_ids[0] == bos:
        a_ids = a_ids[1:]  # a mid-sequence BOS (added by encode) corrupts target alignment
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

    base = generate(model, tokenizer, prompt=QUESTION, max_tokens=24)
    start = time.monotonic()
    with_artifact = decode_greedy(model, tokenizer, artifact.weights, QUESTION, max_tokens=16)
    serve_s = time.monotonic() - start

    base_ok = answer_matches(base, CANARY_ANSWER)
    with_ok = answer_matches(with_artifact, CANARY_ANSWER)
    return {
        "model": MODEL_ID,
        "n_prefix": n_prefix,
        "n_steps": n_steps,
        "seed": seed,
        "hidden_dim": h,
        "baseline_answer": base.strip()[:80],
        "baseline_has_canary": base_ok,
        "with_artifact_answer": with_artifact.strip()[:80],
        "with_artifact_has_canary": with_ok,
        "canary_pass": (not base_ok) and with_ok,
        "train_s": round(train_s, 3),
        "serve_s": round(serve_s, 3),
        "peak_mem_bytes": int(peak),
    }


def decode_greedy(model, tokenizer, prefix, prompt: str, max_tokens: int) -> str:
    """Greedy decode with the trained prefix prepended; frozen sampling loop."""
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=int, default=32)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    model, tokenizer = load_frozen()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if args.sweep:
        points = []
        for np_ in (8, 32, 128, 512):
            print(f"--- prefix={np_} steps={args.steps} ---", flush=True)
            r = run_canary(model, tokenizer, np_, args.steps, args.seed)
            print(
                f"    pass={r['canary_pass']} train={r['train_s']}s "
                f"peak={r['peak_mem_bytes'] / 1e9:.2f}GB "
                f"base={r['baseline_has_canary']} with={r['with_artifact_has_canary']}",
                flush=True,
            )
            points.append(r)
        out = {"points": points}
    else:
        r = run_canary(model, tokenizer, args.prefix, args.steps, args.seed)
        out = r
        print(json.dumps(r, indent=2))

    path = EVIDENCE / f"canary-{int(time.time())}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"retained: {path}")


if __name__ == "__main__":
    main()
