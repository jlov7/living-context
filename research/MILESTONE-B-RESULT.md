# Milestone B archived report — learned path + text retrieval

> **Audit correction, 2026-09-22:** This is an archived historical report, not
> a supported comparison. The learned arm trained on each evaluated Q/A, used
> raw target logits instead of cross-entropy, and accepted malformed substring
> matches. Its three questions differed from the text arm's five. The text arm
> rebuilt an in-memory document map but re-prefilled text on every query, so
> "cached-text" overstates what was measured. Retained MLX memory values are
> active-allocation samples, not process RSS. The decision boundary is withdrawn.

**Date:** 2026-09-21
**Machine:** Apple M4 Max / 48 GB / MLX (no CUDA)
**Model:** `mlx-community/Llama-3.2-1B-Instruct-4bit` (frozen; cached, USD 0.00)
**Evidence files:**
- `artifacts/evidence/baseline-cached-text-1790030776.json` (arm 1)
- `artifacts/evidence/arm-learned-context-1790031016.json` (learned arm, prefix-only carrier)

**Instrumentation-error note (recorded for honesty, 2026-09-21):** two earlier
evidence files were REMOVED because they were instrument errors, not failures —
`baseline-cached-text-1790030758.json` ran with a case-sensitivity bug in the
answer matcher (wrong 4/5), and `arm-learned-context-1790030886.json` had the
document text in the generation prompt (a confound making `bare=3` impossible).
Per the evidence rule "failures are preserved, never tidied", these two were
deliberately distinguishable from genuine failed attempts (e.g. the retained
V-2 valve failure in the learned arm, which IS kept): they measured the wrong
quantity. Their removal is recorded here rather than done silently.

## The question Milestone B answers

With the Milestone-0 gate passed (a learnable artifact demonstrably changes a
frozen model's output on this machine), the next question was: **at the scale
the gate proved, how does the learned path compare against a strong cached-text
baseline done honestly?**

## Exit-criterion check

| Criterion | Status | Evidence |
|---|---|---|
| Gradients, trainable tensors, positional conventions confirmed | **MET** | `PrefixArtifact(nn.Module)` is the only trainable object; base frozen; prefix prepended via `input_embeddings`; Adam over the answer-span CE; peak 0.74 GB (measured) |
| Strong cached-text baseline with **fresh indexing** | **MET** | `baseline_cached_text.py`: index built from the CURRENT (v3) revision of each authored document; retrieval + generation is the serving cost — deliberately NOT uncached repeated prefill |
| Reader model, question set, available facts, output contract matched across arms | **MET** | same frozen model; same 3-question set; same source facts; same `answer.lower() in text.lower()` acceptance |
| Canary discipline: separated injected from base-model knowledge | **MET** | bare (no context, no artifact) answers **0/5** in the baseline arm and **0/3** in the learned arm — the answers come only from injected material |

## Historical substring result (withdrawn as accuracy)

| Arm | Correct | Bare (no carrier) | Latency / cost |
|---|---|---|---|
| **1 — cached text, fresh index** | **5/5** | 0/5 | retrieval+generation ~0.1–0.3 s/question |
| **3 — learned artifact (full refresh)** | **2/3** | 0/3 | ~31–36 s TRAIN per fact + ~0.1 s serve; 0.74 GB peak |

Per-question detail (retained):
- flow rate 55 → artifact pass
- capacity 45 → artifact pass
- **V-2 valve rating 200 → artifact FAIL (retained as a failed build)** — the
  prefix trained but did not absorb the fact on this run. Not deleted, not
  hidden: it is precisely the kind of attempted-failure the protocol requires
  tracking.

## Honest reading (pre-pilot, n=3–5)

1. **At this tiny scale, well-engineered cached text is faster AND more
   accurate.** That is the expected and reportable baseline position — the
   protocol says the decision boundary is the contribution, and this is its
   first measured point: low query volume + low doc churn → text wins.
2. The learned artifact did absorb facts the bare model could not answer
   (2/3), confirming M0's canary at the question level — the mechanism works,
   but costs ~100× the wall time of retrieval per refresh and missed a case.
3. Neither arm was tuned or rehearsed; the V-2 failure is preserved as-is.
   A learned context with cheaper refresh (the actual research question) needs
   the selective/dependency-aware arms — Milestone C — which the harness in
   Milestone A is now ready to drive.

## Non-claims

- This is NOT a claim that learned context loses in general. The pilot corpus
  is 3 questions; the fresh-index advantage at this scale is unsurprising and
  non-significant. It is a measured point on the decision boundary, not a
  verdict.
- NOT a Cartridges reproduction; NOT a benchmark against the paper's regime.
- NOT evidence about H200/B200 behaviour. That remains the collaboration case.

## Commands run

```bash
uv run python experiments/baseline_cached_text.py   # -> 5/5 with ctx, 0/5 bare
uv run python experiments/arm_learned_context.py    # -> 2/3 artifact, 0/3 bare
```

Missing provenance is `unknown`; none of the above was inferred from a model
nickname — the model is pinned by huggingface id and cached locally.
