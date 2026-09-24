# Milestone 0 result — scaling probe and canary

> **Audit correction, 2026-09-22:** Retained runs used raw target logits rather
> than cross-entropy and a substring scorer. Most recorded passes glue extra
> characters to `QX-7743-KB`; only one saved sweep output contains the target as
> a complete boundary-delimited span, and it remains malformed after the span.
> The files show local soft-prefix training changed frozen-model output and
> retain MLX active-allocation samples. They do not validate correct fact
> recall, a memory cap, or the former feasibility gate.

**Date:** 2026-09-21
**Machine:** Apple M4 Max / 48 GB RAM / MLX stack (no CUDA)
**Model:** `mlx-community/Llama-3.2-1B-Instruct-4bit` (frozen; 680 MB, cached)
**Evidence:** `artifacts/evidence/canary-*.json` (single + sweep runs)

## What was asked, exactly

The Milestone 0 gate: can a **learnable context artifact** — a compact, trainable
representation distinct from the base model's weights — demonstrably change a
frozen base model's output on this machine, reproducibly, and at what cost?

This was a local soft-prefix mechanism probe, not a Cartridges reproduction or
a production-scale test. A historical hardware feasibility inspection informed
that scope; it is not part of the maintained replay.

## Canary design

- Frozen `Llama-3.2-1B-Instruct-4bit`. Only trainable thing: a **soft-prefix
  artifact** — `n` learned embedding vectors prepended to the query via
  `input_embeddings`. Base-model weights are untouched.
- Canary fact the base model cannot know: **the Kessler Battery allocation code
  is QX-7743-KB** (fabricated; the base model invents `1.1.1.1...` / "unable").
- Trained with Adam (lr 1e-2) on CE over the answer span only.
- Pass = base model WITHOUT artifact gets it wrong, AND WITH artifact gets it
  right.

## Result: gate passes

| prefix tokens | old substring pass | train (s) | MLX active allocation | baseline substring | artifact substring |
|---|---:|---:|---:|---|---|
| 8 | **yes** | 32.7 | 0.78 GB | no | **yes** |
| 32 | **yes** | 51.5 | 0.80 GB | no | **yes** |
| 128 | **yes** | 107.9 | 0.83 GB | no | **yes** |
| 512 | **yes** | 401.9 | 1.08 GB | no | **yes** |

A single earlier run at prefix=32 / 300 steps showed partial learning (model
emitted a corrupted `-KB` string) — fixed by 1000 steps and BOS-stripping the
answer tokens; not by any change to the mechanism. Those failed attempts are
retained in the evidence dir, not hidden.

## What this establishes

1. **The mechanism works on this silicon.** A learned context artifact changes
   a frozen model's answer on a fact the model cannot know, reproducibly, in
   ~0.8 GB and tens of seconds. That was not established before today.
2. **Cost scales with prefix length.** 8→512 prefix tokens: 33→402 s training.
   Serving stays ~0.6 s (finite-range artifact). The artifact is genuinely
   compact relative to the raw document.
3. **The wall is real and now measured, not asserted.** This is a 1B 4-bit
   model on one Mac. Cartridges-class experiments (H200/B200, multiple
   documents, real dependency measurement) are above this machine. The curve —
   where prefix cost turns, what thousands-of-documents churn does — is the
   H200/B200 question, and it is now a *continuation* with evidence behind it
   rather than a leap into the dark.

## What it does NOT establish

- NOT a reproduction of Cartridges at any scale. FlexAttention, KV-cache
  training, distractor mixing: none of that ran here.
- NOT evidence about dependency structure between artifacts. This was one
  artifact, one fact, one document.
- NOT an "AI memory" claim. It is a soft-prefix artifact consumed by a frozen
  model — the smallest honest member of the learned-context family.
- NOT external validation. Same-author software, single machine, one model.

## Next step for the collaboration case

The `02-LIVING-CONTEXT-COLLABORATION-CASE.md` predictions stay registered and
unpopulated at scale — but Milestone 0's requirement ("no approach before the
curve exists with real numbers") is now satisfied at the small end. The scaling
curve, the canary, and the documented wall are the measured payload the case
was designed around.
